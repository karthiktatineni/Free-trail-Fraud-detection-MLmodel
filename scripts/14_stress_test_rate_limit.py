"""
API Rate Limit & Stress Test Script
===================================
Tests the live production endpoint with concurrent burst requests
to verify that the 30 req/min sliding-window rate limiter triggers HTTP 429
and reports response latencies, status codes, and error payloads.
"""

import os
import time
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

load_dotenv()

URL = os.environ.get("FRAUD_API_URL", "https://free-trail-fraud-detection-mlmodel.onrender.com/api/v1/score")
API_KEY = os.environ.get("TEST_API_KEY") or os.environ.get("FRAUD_API_KEY", "")

if not API_KEY:
    print("[WARNING] No TEST_API_KEY found in .env or environment variables.")
    print("Please set TEST_API_KEY in your .env file before running.")

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

BASE_PAYLOAD = {
    "name": "Sarah Miller",
    "email": "sarah.miller@gmail.com",
    "ip_address": "198.51.100.24",
    "device_id": "dev_macbook_pro_m2_99",
    "payment_token": "pm_visa_auth_8821",
    "area": "new york"
}


def send_request(req_id: int) -> Dict[str, Any]:
    payload = dict(BASE_PAYLOAD)
    payload["email"] = f"sarah.miller+{req_id}@gmail.com"
    
    start_t = time.perf_counter()
    try:
        res = requests.post(URL, json=payload, headers=HEADERS, timeout=15)
        latency = round((time.perf_counter() - start_t) * 1000.0, 2)
        
        try:
            body = res.json()
        except Exception:
            body = res.text
            
        return {
            "id": req_id,
            "status_code": res.status_code,
            "latency_ms": latency,
            "headers": dict(res.headers),
            "response": body,
            "error": None
        }
    except Exception as e:
        latency = round((time.perf_counter() - start_t) * 1000.0, 2)
        return {
            "id": req_id,
            "status_code": None,
            "latency_ms": latency,
            "headers": {},
            "response": None,
            "error": str(e)
        }


def run_stress_test(total_requests: int = 40, concurrency: int = 20):
    print("=" * 70)
    print("FRAUD DETECTION API RATE LIMIT & STRESS TEST")
    print(f"Target URL   : {URL}")
    print(f"API Key      : {API_KEY[:16]}...")
    print(f"Requests     : {total_requests} concurrent requests")
    print(f"Concurrency  : {concurrency} workers")
    print("=" * 70)

    start_all = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, i + 1) for i in range(total_requests)]
        for f in as_completed(futures):
            results.append(f.result())

    total_time = round(time.time() - start_all, 2)
    results.sort(key=lambda x: x["id"])

    # Aggregate Statistics
    status_counts = {}
    rate_limited_count = 0
    success_count = 0
    error_count = 0
    latencies = []

    print("\n--- INDIVIDUAL REQUEST LOGS ---")
    for r in results:
        code = r["status_code"]
        lat = r["latency_ms"]
        latencies.append(lat)
        status_counts[code] = status_counts.get(code, 0) + 1

        if code == 200:
            success_count += 1
            verdict = r["response"].get("verdict", "OK") if isinstance(r["response"], dict) else "OK"
            score = r["response"].get("risk_score", "--") if isinstance(r["response"], dict) else "--"
            print(f"[{r['id']:02d}] 200 OK | Latency: {lat:6.1f}ms | Verdict: {verdict} (Score: {score})")
        elif code == 429:
            rate_limited_count += 1
            detail = r["response"].get("detail", r["response"]) if isinstance(r["response"], dict) else r["response"]
            print(f"[{r['id']:02d}] 429 RATE LIMITED | Latency: {lat:6.1f}ms | Detail: {detail}")
        else:
            error_count += 1
            err_msg = r["error"] or r["response"]
            print(f"[{r['id']:02d}] ERROR ({code}) | Latency: {lat:6.1f}ms | Message: {err_msg}")

    print("\n" + "=" * 70)
    print("TEST SUMMARY & VERIFICATION REPORT")
    print("=" * 70)
    print(f"Total Requests Sent   : {total_requests}")
    print(f"Total Execution Time  : {total_time}s ({round(total_requests / max(total_time, 0.001), 1)} req/s)")
    print(f"Successful (HTTP 200) : {success_count}")
    print(f"Rate Limited (HTTP 429): {rate_limited_count}")
    print(f"Other Errors          : {error_count}")
    if latencies:
        print(f"Average Latency       : {round(sum(latencies)/len(latencies), 1)}ms")
        print(f"Min / Max Latency     : {min(latencies)}ms / {max(latencies)}ms")
    print("Status Code Breakdown :", status_counts)
    
    if rate_limited_count > 0:
        print("\n[VERIFIED] Rate limiting is ACTIVE and successfully blocking bursts > 30 req/min (HTTP 429).")
    else:
        print("\n[NOTE] No 429 status returned. Check if the deployed server has rate limiting middleware enabled.")
    print("=" * 70)


if __name__ == "__main__":
    run_stress_test(total_requests=40, concurrency=20)
