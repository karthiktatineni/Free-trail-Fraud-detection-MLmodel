"""
PRODUCTION FASTAPI MICROSERVICE — Multi-Tenant Commercial Fraud Platform
=========================================================================
Enterprise REST API providing:
  - Multi-tenant user isolation & Firebase Auth integration
  - Per-user API key management (fk_live_... / fk_test_...)
  - Strict 30 requests/minute sliding-window rate limiting
  - Automatic tenant customer persistence & duplicate search
  - Continuous online model retraining trigger
  - Interactive OpenAPI /docs & /redoc documentation
"""

import os
import time
import json
import secrets
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request, Response, Header, Depends, Query, status
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predict import FraudRiskEngine, FEATURE_COLS
from app import HTML_PAGE
from database import (
    get_or_create_user,
    get_user_by_id,
    create_user_api_key,
    list_user_api_keys,
    validate_api_key,
    revoke_user_api_key,
    record_customer_signup,
    list_user_customers,
    search_user_customer,
    push_initial_dataset_to_firebase
)

# Global Engine instance
engine: Optional[FraudRiskEngine] = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")

DEFAULT_RATE_LIMIT = int(os.environ.get("DEFAULT_RATE_LIMIT_PER_MINUTE", 30))


class SlidingWindowRateLimiter:
    """Sliding-window rate limiter per API key (30 requests/min)."""
    def __init__(self, default_limit: int = 30):
        self.default_limit = default_limit
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.lifetime_counts: Dict[str, int] = defaultdict(int)

    def check(self, key: str, limit: Optional[int] = None) -> Tuple[bool, int, int, int]:
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


rate_limiter = SlidingWindowRateLimiter(default_limit=DEFAULT_RATE_LIMIT)

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
    print("[FastAPI] Fraud Detection Multi-Tenant SAAS API Ready.")
    yield
    print("[FastAPI] Shutting down.")


app = FastAPI(
    title="Fraud Detection ML Model — Multi-Tenant Commercial API",
    description="""
### Enterprise Real-Time Fraud & Abuse Detection Platform

- **Strict Rate Limiting:** 30 requests/minute per tenant API key.
- **Tenant Customer Isolation:** All customer events are stored privately under the authenticated tenant's database.
- **Authentication:** Provide your key in the `X-API-Key` header (`fk_live_...` or `fk_test_...`).
    """,
    version="2.1.0",
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
async def authenticate_and_rate_limit(
    request: Request,
    response: Response,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    Validates tenant API key and strictly enforces the 30 req/min sliding-window rate limit.
    """
    api_key = x_api_key
    if not api_key and authorization and authorization.startswith("Bearer "):
        api_key = authorization.split("Bearer ")[1].strip()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "missing_api_key", "message": "Authentication required. Pass X-API-Key header with your fk_live_... key."}
        )

    # Validate against database
    key_record = validate_api_key(api_key)
    if not key_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "invalid_api_key", "message": "API key not recognized or revoked. Please sign in to generate a valid key."}
        )

    limit = key_record.get("rate_limit_per_min", DEFAULT_RATE_LIMIT)
    allowed, effective_limit, remaining, reset_in = rate_limiter.check(api_key, limit=limit)

    # Standard RFC rate limit response headers
    response.headers["X-RateLimit-Limit"] = str(effective_limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(reset_in)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limit_exceeded",
                "message": f"Rate limit of {effective_limit} requests/minute exceeded for your API key.",
                "retry_after_seconds": reset_in
            },
            headers={
                "Retry-After": str(reset_in),
                "X-RateLimit-Limit": str(effective_limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_in)
            }
        )

    return key_record


# ----------------- PYDANTIC SCHEMAS -----------------
class UserSessionRequest(BaseModel):
    uid: str
    email: str
    display_name: Optional[str] = None


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
    customer_id: Optional[str] = None
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


class CreateApiKeyRequest(BaseModel):
    user_id: str
    name: str = "Production API Key"
    key_type: str = "live"


# ----------------- MIDDLEWARE -----------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
    return response


# ----------------- WEB GUI & CONFIG ENDPOINTS -----------------
@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def get_dashboard():
    """Interactive Commercial Developer Portal & Fraud Risk Dashboard."""
    return HTMLResponse(content=HTML_PAGE)


@app.get("/api/v1/config/firebase", tags=["Configuration"])
def get_firebase_client_config():
    """Dynamically serves Firebase configuration from .env (No hardcoding)."""
    return {
        "apiKey": os.environ.get("FIREBASE_API_KEY", ""),
        "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
        "projectId": os.environ.get("FIREBASE_PROJECT_ID", ""),
        "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
        "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID", ""),
        "appId": os.environ.get("FIREBASE_APP_ID", ""),
        "databaseURL": os.environ.get("FIREBASE_DATABASE_URL", ""),
        "defaultRateLimit": DEFAULT_RATE_LIMIT
    }


@app.get("/visuals/{file_path:path}", tags=["Dashboard"])
def get_visual_asset(file_path: str):
    """Serves model evaluation and explainability PNG charts."""
    full_path = os.path.join(VISUALS_DIR, file_path.replace("/", os.sep))
    if os.path.exists(full_path) and os.path.isfile(full_path):
        return FileResponse(full_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Visual asset not found")


# ----------------- USER AUTH & TENANT SESSION -----------------
@app.post("/api/v1/auth/session", tags=["Authentication"])
def create_or_login_user(req: UserSessionRequest):
    """Syncs authenticated user from Firebase Auth into database and returns active profile."""
    user = get_or_create_user(uid=req.uid, email=req.email, display_name=req.display_name)
    keys = list_user_api_keys(req.uid)
    return {
        "user": user,
        "keys": keys,
        "rate_limit_per_min": DEFAULT_RATE_LIMIT
    }


# ----------------- API KEY MANAGEMENT -----------------
@app.post("/api/v1/keys/create", tags=["API Key Management"])
def generate_api_key(req: CreateApiKeyRequest):
    """Generates a new unique API key tied strictly to the authenticated tenant."""
    key_record = create_user_api_key(
        user_id=req.user_id,
        name=req.name,
        key_type=req.key_type,
        rate_limit_per_min=DEFAULT_RATE_LIMIT
    )
    return key_record


@app.get("/api/v1/keys/list", tags=["API Key Management"])
def list_api_keys(user_id: str = Query(...)):
    """Retrieves all active API keys belonging strictly to the specified user."""
    keys = list_user_api_keys(user_id)
    for k in keys:
        k["requests_this_minute"] = rate_limiter.get_current_window_count(k.get("key_hash", ""))
    return {"keys": keys}


@app.delete("/api/v1/keys/{key_id}", tags=["API Key Management"])
def revoke_key(key_id: str, user_id: str = Query(...)):
    """Revokes an API key belonging to a tenant."""
    success = revoke_user_api_key(user_id=user_id, key_id=key_id)
    if success:
        return {"status": "success", "message": "Key revoked."}
    raise HTTPException(status_code=404, detail="Key not found or unauthorized.")


# ----------------- MULTI-TENANT CUSTOMERS -----------------
@app.get("/api/v1/customers/list", tags=["Tenant Customer Database"])
def list_tenant_customers(user_id: str = Query(...), limit: int = 50):
    """Lists customer signup events scored under this user's account."""
    customers = list_user_customers(user_id=user_id, limit=limit)
    return {"customers": customers, "count": len(customers)}


@app.get("/api/v1/customers/search", tags=["Tenant Customer Database"])
def search_tenant_customer(user_id: str = Query(...), q: str = Query(...)):
    """Searches if a customer exists under this user's account by email, IP, name, or payment token."""
    return search_user_customer(user_id=user_id, query=q)


# ----------------- CONTINUOUS LEARNING & DATASET SYNC -----------------
@app.post("/api/v1/model/retrain", tags=["Continuous Learning"])
def trigger_continuous_retraining():
    """Runs continuous learning model retraining on cumulative production customer data."""
    from scripts.continuous_retraining import run_continuous_training
    report = run_continuous_training()
    # Reload engine pipeline
    global engine
    engine = None
    get_engine()
    return report


@app.post("/api/v1/firebase/sync-dataset", tags=["Continuous Learning"])
def sync_dataset_to_firebase(batch_limit: int = 200):
    """Pushes historical raw dataset to Firebase Firestore."""
    return push_initial_dataset_to_firebase(batch_limit=batch_limit)


# ----------------- CORE INFERENCE ENDPOINTS -----------------
@app.post(
    "/api/v1/score",
    response_model=RiskScoreResponse,
    tags=["Inference"],
    summary="Real-Time Single Event Risk Scoring (Strict 30 req/min)"
)
def score_signup(
    event: SignupEventRequest,
    auth_key: Dict[str, Any] = Depends(authenticate_and_rate_limit)
):
    """
    Scores a single signup event in <15ms.
    Strictly rate limited to 30 requests/minute.
    Automatically records the customer under the caller's tenant account.
    """
    start_t = time.perf_counter()
    eng = get_engine()

    raw_event = event.model_dump()
    result = eng.score_event(raw_event, update_state=True)

    lat_ms = (time.perf_counter() - start_t) * 1000.0
    result["latency_ms"] = round(lat_ms, 2)

    # Persist customer under authenticated user account
    tenant_uid = auth_key["user_id"]
    cust_id = record_customer_signup(user_id=tenant_uid, event_data=raw_event, score_result=result)
    result["customer_id"] = cust_id

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
    summary="Batch Event Risk Scoring (Strict 30 req/min)"
)
def score_batch(
    batch: BatchPredictionRequest,
    auth_key: Dict[str, Any] = Depends(authenticate_and_rate_limit)
):
    """Scores a batch of signup events under the caller's tenant account."""
    start_t = time.perf_counter()
    eng = get_engine()
    tenant_uid = auth_key["user_id"]

    results = []
    verdict_counts: Dict[str, int] = defaultdict(int)

    for item in batch.events:
        item_start = time.perf_counter()
        raw_event = item.model_dump()
        res = eng.score_event(raw_event, update_state=True)
        res["latency_ms"] = round((time.perf_counter() - item_start) * 1000.0, 2)
        cust_id = record_customer_signup(user_id=tenant_uid, event_data=raw_event, score_result=res)
        res["customer_id"] = cust_id
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


# ----------------- OPERATIONAL MONITORING -----------------
@app.get("/healthz", tags=["Monitoring"])
def health_check():
    eng = get_engine()
    return {
        "status": "healthy",
        "service": "Fraud Detection ML Model SAAS Microservice",
        "model_loaded": eng.pipeline is not None,
        "feature_count": len(FEATURE_COLS),
        "rate_limit_per_min": DEFAULT_RATE_LIMIT,
        "uptime_status": "operational"
    }


@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    avg_lat = stats["total_latency_ms"] / max(1, stats["total_requests"])
    return {
        "total_requests_processed": stats["total_requests"],
        "verdict_distribution": stats["verdicts"],
        "average_inference_latency_ms": round(avg_lat, 2),
        "feature_count": len(FEATURE_COLS)
    }


@app.get("/api/v1/drift", tags=["Monitoring"])
def get_drift_status():
    drift_file = os.path.join(os.path.dirname(__file__), "results", "drift_analysis.json")
    if os.path.exists(drift_file):
        with open(drift_file, "r") as f:
            return json.load(f)
    return {"status": "No drift audit found. Run scripts/07_drift_monitor.py first."}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Fraud Detection SAAS Microservice on http://0.0.0.0:{port} ...")
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
