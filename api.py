"""
PRODUCTION FASTAPI MICROSERVICE — XGBoost-Powered Risk Scoring
===============================================================
REST API for Real-Time Fraud Detection. Risk scores are computed by
the trained XGBoost model via predict_proba (not rule-based heuristics).
A rule-based signal_breakdown is provided alongside for UI explainability.

Endpoints:
- GET  /healthz        : Health check & readiness probe.
- GET  /metrics        : Operational statistics (calls, avg latency, verdicts).
- POST /api/v1/score   : Real-time single event risk scoring (<15ms).
- POST /api/v1/batch   : Batch scoring on list of events.
- GET  /api/v1/drift   : Latest PSI and Covariate Shift status.

Documentation:
- Interactive Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
"""

import os
import time
import json
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predict import FraudRiskEngine, FEATURE_COLS
from app import HTML_PAGE

# Global Engine instance
engine: Optional[FraudRiskEngine] = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")

# Telemetry stats
stats = {
    "total_requests": 0,
    "verdicts": {"NEW USER (GENUINE)": 0, "SUSPICIOUS (STEP-UP)": 0, "REPEATING USER (LIKELY ABUSE)": 0},
    "total_latency_ms": 0.0
}

def get_engine() -> FraudRiskEngine:
    global engine
    if engine is None:
        redis_url = os.environ.get("REDIS_URL")
        print(f"[FastAPI] Initializing FraudRiskEngine (Redis: {redis_url or 'in-memory fallback'})...")
        engine = FraudRiskEngine(warm_start=True, redis_url=redis_url)
    return engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_engine()
    print("[FastAPI] FraudRiskEngine ready for live inference.")
    yield
    print("[FastAPI] Shutting down FraudRiskEngine.")

app = FastAPI(
    title="Fraud Detection ML Model - Real-Time Risk Microservice",
    description="Low-latency (<15ms) real-time risk scoring engine for signup abuse detection.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- PYDANTIC SCHEMAS -----------------
class SignupEventRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Unique account identifier (auto-generated if omitted)")
    name: str = Field(..., description="Full user name", json_schema_extra={"example": "David Smith"})
    email: str = Field(..., description="User signup email address", json_schema_extra={"example": "david.smith@gmail.com"})
    ip_address: str = Field(..., description="Connecting IPv4 address", json_schema_extra={"example": "203.0.113.45"})
    device_id: str = Field(..., description="Hardware fingerprint hash", json_schema_extra={"example": "dev_macbook_pro_99"})
    payment_token: str = Field(..., description="Tokenized payment card token", json_schema_extra={"example": "pm_barclays_card_99"})
    area: Optional[str] = Field("london", description="City / Region", json_schema_extra={"example": "london"})
    device_os: Optional[str] = Field("macos", description="Operating system", json_schema_extra={"example": "macos"})
    payment_country: Optional[str] = Field(None, description="BIN issuing country code", json_schema_extra={"example": "GB"})
    signup_time: Optional[str] = Field(None, description="Event timestamp", json_schema_extra={"example": "2026-07-15 14:30:00"})

class RiskScoreResponse(BaseModel):
    user_id: Optional[str] = None
    risk_score: float
    verdict: str
    recommended_action: str
    severity: Optional[str] = None
    model_confidence_pct: float
    model_probability: Optional[float] = None
    decision_threshold: float
    latency_ms: float
    signal_breakdown: Dict[str, float]
    raw_features: Dict[str, Any]

class BatchPredictionRequest(BaseModel):
    events: List[SignupEventRequest]

class BatchPredictionResponse(BaseModel):
    total_scored: int
    summary_verdicts: Dict[str, int]
    avg_latency_ms: float
    results: List[RiskScoreResponse]

# ----------------- MIDDLEWARE -----------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
    return response

# ----------------- WEB GUI & ROOT ROUTES -----------------
@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def get_dashboard():
    """Interactive Real-Time Fraud Assessment Web Dashboard."""
    return HTMLResponse(content=HTML_PAGE)

@app.get("/visuals/{file_path:path}", tags=["Dashboard"])
def get_visual_asset(file_path: str):
    """Serves model evaluation and explainability PNG charts."""
    full_path = os.path.join(VISUALS_DIR, file_path.replace("/", os.sep))
    if os.path.exists(full_path) and os.path.isfile(full_path):
        return FileResponse(full_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Visual asset not found")

@app.post("/api/score", tags=["Dashboard"])
def web_score(event: Dict[str, Any]):
    """Frontend endpoint for web dashboard."""
    eng = get_engine()
    return eng.score_event(event, update_state=True)

@app.post("/api/score-batch", tags=["Dashboard"])
def web_score_batch(rows: List[Dict[str, Any]]):
    """Frontend batch endpoint for web dashboard."""
    eng = get_engine()
    results = []
    for row in rows:
        res = eng.score_event(row, update_state=True)
        results.append({
            "user_id": res["user_id"],
            "name": row.get("name", res["user_id"]),
            "email": row.get("email", ""),
            "risk_score": res["risk_score"],
            "verdict": res["verdict"],
            "recommended_action": res["recommended_action"],
            "severity": res["severity"],
            "confidence": res["model_confidence_pct"],
            "top_signal": list(res["signal_breakdown"].keys())[0] if res["signal_breakdown"] else "",
        })
    return results

# ----------------- ENDPOINTS -----------------
@app.get("/healthz", status_code=status.HTTP_200_OK, tags=["System"])
def health_check():
    """Kubernetes liveness and readiness probe."""
    eng = get_engine()
    return {
        "status": "healthy",
        "service": "Fraud Detection ML Model Risk Engine",
        "model_loaded": eng is not None,
        "feature_count": len(FEATURE_COLS)
    }

@app.get("/metrics", tags=["System"])
def get_metrics():
    """Operational monitoring statistics."""
    avg_lat = (stats["total_latency_ms"] / max(1, stats["total_requests"]))
    return {
        "total_requests_processed": stats["total_requests"],
        "verdict_distribution": stats["verdicts"],
        "average_inference_latency_ms": round(avg_lat, 2)
    }

@app.post("/api/v1/score", response_model=RiskScoreResponse, tags=["Inference"])
def score_signup(event: SignupEventRequest):
    """
    Evaluates a single signup event in real-time (<15ms).
    Computes graph linkage, rolling velocity, and 3-band verdict.
    """
    eng = get_engine()
    t0 = time.perf_counter()
    payload = event.model_dump()
    if not payload.get("user_id"):
        payload["user_id"] = f"u_{abs(hash(payload.get('email', ''))) % 10000000:07d}"
    result = eng.score_event(payload, update_state=True)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    # Telemetry
    stats["total_requests"] += 1
    stats["verdicts"][result["verdict"]] = stats["verdicts"].get(result["verdict"], 0) + 1
    stats["total_latency_ms"] += latency_ms
    
    result["latency_ms"] = round(latency_ms, 2)
    return result

@app.post("/api/v1/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def batch_score_signups(batch: BatchPredictionRequest):
    """Batch scores a list of signup events."""
    eng = get_engine()
    t0 = time.perf_counter()
    results = []
    verdict_counts = {}
    
    for evt in batch.events:
        res = eng.score_event(evt.model_dump(), update_state=True)
        v = res["verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        res["latency_ms"] = 0.0
        results.append(res)
        
    total_time = (time.perf_counter() - t0) * 1000.0
    avg_lat = total_time / max(1, len(batch.events))
    
    return {
        "total_scored": len(batch.events),
        "summary_verdicts": verdict_counts,
        "avg_latency_ms": round(avg_lat, 2),
        "results": results
    }

@app.get("/api/v1/drift", tags=["Monitoring"])
def get_drift_status():
    """Retrieves latest PSI and Covariate Shift audit report."""
    drift_file = os.path.join(os.path.dirname(__file__), "results", "drift_analysis.json")
    if os.path.exists(drift_file):
        with open(drift_file, "r") as f:
            return json.load(f)
    return {"status": "No drift audit found. Run scripts/07_drift_monitor.py first."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Fraud Detection ML Model Production Microservice on http://0.0.0.0:{port} ...")
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
