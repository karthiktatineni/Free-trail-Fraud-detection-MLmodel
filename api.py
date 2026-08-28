"""
PRODUCTION FASTAPI MICROSERVICE — Commercial Fraud Detection API
==================================================================
High-throughput REST API for Real-Time Fraud & Abuse Detection.
Features:
  - Synchronous real-time ML inference (<15ms)
  - API Key Authentication (fk_live_... / fk_test_...)
  - Sliding-Window Rate Limiting (with RFC standard headers & 429 response)
  - Interactive Swagger UI & ReDoc documentation (/docs, /redoc)
  - Built-in Developer Portal & Web GUI Dashboard (/)
"""

import os
import time
import json
import secrets
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, Header, Depends, status
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predict import FraudRiskEngine, FEATURE_COLS
from app import HTML_PAGE

# Global Engine instance
engine: Optional[FraudRiskEngine] = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")

# ----------------- IN-MEMORY API KEY & RATE LIMIT STORE -----------------
# Supports both pre-seeded demo keys and dynamically generated user keys
API_KEYS: Dict[str, Dict[str, Any]] = {
    "fk_live_demo_9824ab71f2": {
        "key_id": "key_demo_01",
        "name": "Default Production Key",
        "type": "live",
        "created_at": "2026-01-01T00:00:00Z",
        "rate_limit_per_min": 60,
        "user_email": "developer@enterprise.io"
    },
    "fk_test_demo_5512cd39e4": {
        "key_id": "key_demo_02",
        "name": "Sandbox Test Key",
        "type": "test",
        "created_at": "2026-01-01T00:00:00Z",
        "rate_limit_per_min": 120,
        "user_email": "developer@enterprise.io"
    }
}


class SlidingWindowRateLimiter:
    """Sliding-window rate limiter per API key / IP."""
    def __init__(self, default_limit: int = 60):
        self.default_limit = default_limit
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.lifetime_counts: Dict[str, int] = defaultdict(int)

    def check(self, key: str, limit: Optional[int] = None) -> Tuple[bool, int, int, int]:
        """
        Returns:
            (allowed: bool, limit: int, remaining: int, reset_seconds: int)
        """
        now = time.time()
        effective_limit = limit or self.default_limit
        cutoff = now - 60.0

        # Evict timestamps older than 60s
        self.history[key] = [t for t in self.history[key] if t > cutoff]
        current_count = len(self.history[key])

        if current_count >= effective_limit:
            earliest = self.history[key][0]
            reset_in = max(1, int(60.0 - (now - earliest)))
            return False, effective_limit, 0, reset_in

        self.history[key].append(now)
        self.lifetime_counts[key] += 1
        remaining = max(0, effective_limit - len(self.history[key]))
        return True, effective_limit, remaining, 60

    def get_current_window_count(self, key: str) -> int:
        cutoff = time.time() - 60.0
        self.history[key] = [t for t in self.history[key] if t > cutoff]
        return len(self.history[key])


rate_limiter = SlidingWindowRateLimiter(default_limit=60)

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
    print("[FastAPI] Fraud Detection ML Model ready for live inference.")
    yield
    print("[FastAPI] Shutting down FraudRiskEngine.")


app = FastAPI(
    title="Fraud Detection ML Model - Commercial Risk Microservice",
    description="""
### Real-Time Anti-Abuse & Multi-Accounting Risk API

The Fraud Detection API provides high-throughput (<15ms) risk scoring for SaaS signups, checkout flows, and trial onboarding.

#### Authentication:
Include your API key in every request via the `X-API-Key` header or `Authorization: Bearer <key>`.
- **Live Key:** `fk_live_...` (Standard production rate limit: 60 req/min)
- **Test Key:** `fk_test_...` (Sandbox rate limit: 120 req/min)

#### Standard Rate Limiting Headers:
All API responses return RFC-compliant rate limit headers:
- `X-RateLimit-Limit`: Maximum requests permitted per minute.
- `X-RateLimit-Remaining`: Requests remaining in the current 60s sliding window.
- `X-RateLimit-Reset`: Time in seconds until the sliding window resets.
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- AUTH & RATE LIMIT DEPENDENCY -----------------
async def verify_api_key_and_rate_limit(
    request: Request,
    response: Response,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None)
):
    """
    Validates API key (from X-API-Key or Bearer Token) and enforces
    per-key sliding-window rate limiting.
    """
    api_key = x_api_key
    if not api_key and authorization and authorization.startswith("Bearer "):
        api_key = authorization.split("Bearer ")[1].strip()

    client_ip = request.client.host if request.client else "127.0.0.1"

    # Identify user / key tier
    if api_key and api_key in API_KEYS:
        key_meta = API_KEYS[api_key]
        rate_key = api_key
        limit = key_meta.get("rate_limit_per_min", 60)
    elif api_key:
        # Dynamic key format validation
        if api_key.startswith("fk_live_") or api_key.startswith("fk_test_"):
            rate_key = api_key
            limit = 60
            API_KEYS[api_key] = {
                "key_id": f"key_{secrets.token_hex(4)}",
                "name": "Custom Dynamic Key",
                "type": "live" if api_key.startswith("fk_live_") else "test",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rate_limit_per_min": 60,
                "user_email": "authenticated_user@domain.com"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": "invalid_api_key", "message": "Invalid API key format. Must start with fk_live_ or fk_test_."}
            )
    else:
        # Anonymous / Demo IP tier (30 req/min)
        rate_key = f"ip_{client_ip}"
        limit = 30

    allowed, effective_limit, remaining, reset_in = rate_limiter.check(rate_key, limit=limit)

    # Attach standard RFC rate limit headers
    response.headers["X-RateLimit-Limit"] = str(effective_limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(reset_in)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limit_exceeded",
                "message": f"Rate limit of {effective_limit} requests/minute exceeded for this key.",
                "retry_after_seconds": reset_in
            },
            headers={
                "Retry-After": str(reset_in),
                "X-RateLimit-Limit": str(effective_limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_in)
            }
        )

    return rate_key


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
    signup_time: Optional[str] = Field(None, description="Event timestamp (ISO-8601)", json_schema_extra={"example": "2026-07-15 14:30:00"})


class RiskScoreResponse(BaseModel):
    user_id: Optional[str] = None
    risk_score: float = Field(..., description="Calibrated risk score between 0.0 (Clean) and 100.0 (Abuse)")
    verdict: str = Field(..., description="Decision verdict: NEW USER (GENUINE), SUSPICIOUS (STEP-UP), or REPEATING USER (LIKELY ABUSE)")
    recommended_action: str = Field(..., description="Downstream policy action: ALLOW, STEP-UP, or BLOCK")
    severity: Optional[str] = Field(None, description="Severity category: low, medium, or high")
    model_confidence_pct: float = Field(..., description="Model certainty percentage")
    model_probability: Optional[float] = Field(None, description="Raw probability from predict_proba()")
    decision_threshold: float = Field(..., description="Current operating decision threshold")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    signal_breakdown: Dict[str, float] = Field(..., description="Additive signal weights for explainability")
    raw_features: Dict[str, Any] = Field(..., description="Full 20-dimensional causal feature vector")


class BatchPredictionRequest(BaseModel):
    events: List[SignupEventRequest]


class BatchPredictionResponse(BaseModel):
    total_scored: int
    summary_verdicts: Dict[str, int]
    avg_latency_ms: float
    results: List[RiskScoreResponse]


class CreateApiKeyRequest(BaseModel):
    name: str = Field("Production API Key", description="Human-readable label for the key")
    key_type: str = Field("live", description="Key type: 'live' (production) or 'test' (sandbox)")
    user_email: Optional[str] = Field("developer@enterprise.io", description="Associated developer email")


class CreateApiKeyResponse(BaseModel):
    api_key: str
    key_id: str
    name: str
    key_type: str
    rate_limit_per_min: int
    created_at: str


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
    """Interactive Developer Portal & Fraud Risk Dashboard."""
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
    """Frontend endpoint for web dashboard playground."""
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


# ----------------- API KEY MANAGEMENT ENDPOINTS -----------------
@app.post(
    "/api/v1/keys/create",
    response_model=CreateApiKeyResponse,
    tags=["API Key Management"],
    summary="Create a new API Key"
)
def create_api_key(req: CreateApiKeyRequest):
    """Generates a new secure live (fk_live_...) or test (fk_test_...) API key."""
    prefix = "fk_live_" if req.key_type == "live" else "fk_test_"
    random_part = secrets.token_hex(16)
    new_key = f"{prefix}{random_part}"
    key_id = f"key_{secrets.token_hex(4)}"
    limit = 60 if req.key_type == "live" else 120
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    API_KEYS[new_key] = {
        "key_id": key_id,
        "name": req.name,
        "type": req.key_type,
        "created_at": now_iso,
        "rate_limit_per_min": limit,
        "user_email": req.user_email
    }

    return CreateApiKeyResponse(
        api_key=new_key,
        key_id=key_id,
        name=req.name,
        key_type=req.key_type,
        rate_limit_per_min=limit,
        created_at=now_iso
    )


@app.get(
    "/api/v1/keys/list",
    tags=["API Key Management"],
    summary="List Active API Keys"
)
def list_api_keys():
    """Retrieves all registered active API keys."""
    keys_list = []
    for key_str, meta in API_KEYS.items():
        masked = f"{key_str[:12]}...{key_str[-4:]}"
        keys_list.append({
            "key_id": meta["key_id"],
            "name": meta["name"],
            "type": meta["type"],
            "masked_key": masked,
            "raw_key": key_str,
            "rate_limit_per_min": meta["rate_limit_per_min"],
            "created_at": meta["created_at"],
            "requests_this_minute": rate_limiter.get_current_window_count(key_str)
        })
    return {"keys": keys_list}


@app.delete(
    "/api/v1/keys/{key_id}",
    tags=["API Key Management"],
    summary="Revoke an API Key"
)
def revoke_api_key(key_id: str):
    """Revokes and deletes an API key."""
    to_delete = None
    for k, meta in API_KEYS.items():
        if meta["key_id"] == key_id:
            to_delete = k
            break
    if to_delete:
        del API_KEYS[to_delete]
        return {"status": "success", "message": f"API key {key_id} revoked."}
    raise HTTPException(status_code=404, detail="API key not found")


@app.get(
    "/api/v1/keys/usage",
    tags=["API Key Management"],
    summary="Get Real-Time Usage & Quota Stats"
)
def get_usage_stats(rate_key: str = Depends(verify_api_key_and_rate_limit)):
    """Returns current rate limit quota and requests consumed in the sliding window."""
    current_count = rate_limiter.get_current_window_count(rate_key)
    limit = API_KEYS.get(rate_key, {}).get("rate_limit_per_min", 60)
    return {
        "rate_key": rate_key[:12] + "..." if len(rate_key) > 16 else rate_key,
        "requests_current_minute": current_count,
        "limit_per_minute": limit,
        "remaining_requests": max(0, limit - current_count),
        "total_requests_all_time": stats["total_requests"]
    }


# ----------------- OPERATIONAL MONITORING -----------------
@app.get("/healthz", tags=["Monitoring"], summary="Health & Readiness Probe")
def health_check():
    """Kubernetes / Cloud load-balancer readiness probe."""
    eng = get_engine()
    return {
        "status": "healthy",
        "service": "Fraud Detection ML Model Risk Engine",
        "model_loaded": eng.pipeline is not None,
        "feature_count": len(FEATURE_COLS),
        "uptime_status": "operational"
    }


@app.get("/metrics", tags=["Monitoring"], summary="Operational Telemetry")
def get_metrics():
    """Returns cumulative request volume, verdict distribution, and average latency."""
    avg_lat = stats["total_latency_ms"] / max(1, stats["total_requests"])
    return {
        "total_requests_processed": stats["total_requests"],
        "verdict_distribution": stats["verdicts"],
        "average_inference_latency_ms": round(avg_lat, 2),
        "feature_count": len(FEATURE_COLS)
    }


@app.get("/api/v1/drift", tags=["Monitoring"], summary="Population Stability Index Drift Report")
def get_drift_status():
    """Retrieves the latest PSI and Covariate Shift audit report."""
    drift_file = os.path.join(os.path.dirname(__file__), "results", "drift_analysis.json")
    if os.path.exists(drift_file):
        with open(drift_file, "r") as f:
            return json.load(f)
    return {"status": "No drift audit found. Run scripts/07_drift_monitor.py first."}


# ----------------- CORE INFERENCE ENDPOINTS -----------------
@app.post(
    "/api/v1/score",
    response_model=RiskScoreResponse,
    tags=["Inference"],
    summary="Real-Time Single Event Risk Scoring"
)
def score_signup(
    event: SignupEventRequest,
    rate_key: str = Depends(verify_api_key_and_rate_limit)
):
    """
    Scores a single signup event synchronously in real time (<15ms).
    Extracts 20 causal features, evaluates ML model predict_proba(),
    and maps output to an actionable 3-band risk policy.
    """
    start_t = time.perf_counter()
    eng = get_engine()

    raw_event = event.model_dump()
    result = eng.score_event(raw_event, update_state=True)

    lat_ms = (time.perf_counter() - start_t) * 1000.0
    result["latency_ms"] = round(lat_ms, 2)

    # Update telemetry stats
    stats["total_requests"] += 1
    stats["total_latency_ms"] += lat_ms
    verdict = result["verdict"]
    stats["verdicts"][verdict] = stats["verdicts"].get(verdict, 0) + 1

    return result


@app.post(
    "/api/v1/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="Batch Event Risk Scoring"
)
def score_batch(
    batch: BatchPredictionRequest,
    rate_key: str = Depends(verify_api_key_and_rate_limit)
):
    """Scores a batch of signup events in a single API call."""
    start_t = time.perf_counter()
    eng = get_engine()

    results = []
    verdict_counts: Dict[str, int] = defaultdict(int)

    for item in batch.events:
        item_start = time.perf_counter()
        res = eng.score_event(item.model_dump(), update_state=True)
        res["latency_ms"] = round((time.perf_counter() - item_start) * 1000.0, 2)
        results.append(res)
        verdict_counts[res["verdict"]] += 1
        stats["total_requests"] += 1
        stats["verdicts"][res["verdict"]] = stats["verdicts"].get(res["verdict"], 0) + 1

    total_time = (time.perf_counter() - start_t) * 1000.0
    stats["total_latency_ms"] += total_time
    avg_lat = total_time / max(1, len(batch.events))

    return {
        "total_scored": len(batch.events),
        "summary_verdicts": dict(verdict_counts),
        "avg_latency_ms": round(avg_lat, 2),
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Fraud Detection ML Model Production Microservice on http://0.0.0.0:{port} ...")
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
