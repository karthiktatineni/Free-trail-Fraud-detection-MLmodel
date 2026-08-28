"""
Fraud Detection ML Model — Python Client SDK
=============================================
Lightweight, zero-dependency Python client for integrating real-time
fraud risk scoring into SaaS signup flows, payment gateways, and backend services.

Usage:
    from client import FraudDetectionClient

    # Initialize with your API Key
    client = FraudDetectionClient(base_url="http://localhost:8000", api_key="fk_live_demo_9824ab71f2")
    
    # 1. Single Event Real-Time Scoring:
    res = client.score_signup(
        name="David Smith",
        email="david.smith@gmail.com",
        ip_address="203.0.113.45",
        device_id="dev_macbook_pro_99",
        payment_token="pm_barclays_card_99",
        area="london",
        device_os="macos"
    )
    print(f"Verdict: {res['verdict']} | Score: {res['risk_score']}/100 | Action: {res['recommended_action']}")

    # 2. Batch Scoring:
    batch_res = client.score_batch([
        {"name": "User A", "email": "a@gmail.com", "ip_address": "1.1.1.1", "device_id": "d1", "payment_token": "p1", "area": "mumbai"},
        {"name": "User B", "email": "b@yopmail.com", "ip_address": "2.2.2.2", "device_id": "d2", "payment_token": "p2", "area": "delhi"}
    ])
    print(f"Scored {batch_res['total_scored']} events.")
"""

import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional


class FraudDetectionClient:
    """Client for the Fraud Detection REST API Microservice."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = "fk_live_demo_9824ab71f2",
        timeout: float = 5.0,
        max_retries_on_rate_limit: int = 2
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries_on_rate_limit

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "FraudDetectionClient-Python/2.0.0"
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _post(self, endpoint: str, payload: Dict[str, Any], attempt: int = 0) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            if e.code == 429 and attempt < self.max_retries:
                # Handle rate limit backoff
                retry_after = 2.0
                try:
                    err_json = json.loads(err_body)
                    retry_after = float(err_json.get("detail", {}).get("retry_after_seconds", 2.0))
                except Exception:
                    pass
                time.sleep(retry_after)
                return self._post(endpoint, payload, attempt=attempt + 1)

            raise RuntimeError(f"API Error ({e.code}): {err_body}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to Fraud API at {self.base_url}: {e.reason}") from e

    def _get(self, endpoint: str) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            raise RuntimeError(f"API Error ({e.code}): {err_body}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to Fraud API at {self.base_url}: {e.reason}") from e

    def healthz(self) -> Dict[str, Any]:
        """Checks API health and readiness probe."""
        return self._get("/healthz")

    def get_metrics(self) -> Dict[str, Any]:
        """Retrieves operational telemetry and verdict counts."""
        return self._get("/metrics")

    def get_usage(self) -> Dict[str, Any]:
        """Retrieves current rate limit quota and requests consumed in the window."""
        return self._get("/api/v1/keys/usage")

    def get_drift_report(self) -> Dict[str, Any]:
        """Retrieves latest PSI drift monitoring report."""
        return self._get("/api/v1/drift")

    def score_signup(
        self,
        name: str,
        email: str,
        ip_address: str,
        device_id: str,
        payment_token: str,
        area: str = "mumbai",
        device_os: str = "android",
        payment_country: Optional[str] = None,
        user_id: Optional[str] = None,
        signup_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Scores a single signup event synchronously in real time (<15ms).
        
        Returns:
            Dict with: risk_score (0-100), verdict, recommended_action, severity,
                       model_confidence_pct, model_probability, latency_ms, signal_breakdown.
        """
        payload = {
            "name": name,
            "email": email,
            "ip_address": ip_address,
            "device_id": device_id,
            "payment_token": payment_token,
            "area": area,
            "device_os": device_os,
            "payment_country": payment_country,
            "user_id": user_id,
            "signup_time": signup_time
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        return self._post("/api/v1/score", payload)

    def score_batch(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Scores a batch of signup events in a single API call."""
        return self._post("/api/v1/batch", {"events": events})


if __name__ == "__main__":
    print("Testing FraudDetectionClient with API Key...")
    client = FraudDetectionClient("http://localhost:8000", api_key="fk_live_demo_9824ab71f2")
    try:
        health = client.healthz()
        print("API Health:", health)
    except ConnectionError:
        print("Server offline. Start with `py api.py` to run live tests.")
