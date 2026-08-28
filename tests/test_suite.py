"""
AUTOMATED PRODUCTION TEST SUITE
================================
Unit and Integration Tests for:
1. Incremental Union-Find Disjoint-Set Graph
2. Causal Feature Store & Sliding 24-Hour Velocity
3. Precision SLA & Cost-Optimized Decision Threshold Invariants
4. Population Stability Index (PSI) Drift Math
5. Model-Driven Scoring & Redis Feature Store Fallback
6. FastAPI Microservice Endpoints
"""

import os
import sys
import time
import json
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from predict import IncrementalUnionFind, FraudRiskEngine, FEATURE_COLS
import importlib
drift_module = importlib.import_module("scripts.07_drift_monitor")
calculate_psi = drift_module.calculate_psi
redis_module = importlib.import_module("scripts.17_redis_feature_store")
RedisFeatureStore = redis_module.RedisFeatureStore
InMemorySortedSet = redis_module.InMemorySortedSet


class TestUnionFind(unittest.TestCase):
    """Tests for Incremental Union-Find Graph Component Tracking."""
    
    def test_single_node(self):
        uf = IncrementalUnionFind()
        self.assertEqual(uf.get_component_size("node_A"), 1)

    def test_connected_component_size(self):
        uf = IncrementalUnionFind()
        uf.union("card_1", "dev_1")
        uf.union("dev_1", "subnet_1")
        self.assertEqual(uf.get_component_size("card_1"), 3)
        self.assertEqual(uf.get_component_size("dev_1"), 3)
        self.assertEqual(uf.get_component_size("subnet_1"), 3)

    def test_independent_clusters(self):
        uf = IncrementalUnionFind()
        uf.union("c1", "d1")
        uf.union("c2", "d2")
        uf.union("d2", "s2")
        
        self.assertEqual(uf.get_component_size("c1"), 2)
        self.assertEqual(uf.get_component_size("c2"), 3)

    def test_cycle_handling(self):
        uf = IncrementalUnionFind()
        uf.union("A", "B")
        uf.union("B", "C")
        uf.union("C", "A")  # Cycle
        self.assertEqual(uf.get_component_size("A"), 3)


class TestCausalFeatureStore(unittest.TestCase):
    """Tests for Causal Temporal Guarantees & Sliding Window Eviction."""

    def test_sliding_window_eviction(self):
        store = RedisFeatureStore(redis_url=None)
        t0 = 100000.0
        store.record_sliding_event("vel:test", "e1", t0 - 1000)
        store.record_sliding_event("vel:test", "e2", t0 - 90000)

        v_24h = store.get_sliding_velocity("vel:test", t0, window_seconds=86400)
        self.assertEqual(v_24h, 1, "Expired event was not evicted from 24h window!")

    def test_zero_lookahead_causality(self):
        engine = FraudRiskEngine(warm_start=False)
        event = {
            "name": "Jane Doe",
            "email": "jane.doe@gmail.com",
            "ip_address": "198.51.100.1",
            "device_id": "dev_test_causal",
            "payment_token": "pm_test_causal",
            "area": "london"
        }
        res1 = engine.score_event(event, update_state=True)
        self.assertEqual(res1["raw_features"]["payment_reuse_count"], 0)
        self.assertEqual(res1["raw_features"]["graph_component_size"], 1)

        res2 = engine.score_event(event, update_state=False)
        self.assertEqual(res2["raw_features"]["payment_reuse_count"], 1)


class TestModelDrivenScoring(unittest.TestCase):
    """Tests that model predict_proba drives the risk score and fallback works."""

    def test_model_probability_present_and_drives_score(self):
        engine = FraudRiskEngine(warm_start=False)
        event = {
            "name": "Test User",
            "email": "test@gmail.com",
            "ip_address": "1.2.3.4",
            "device_id": "d_test",
            "payment_token": "p_test",
            "area": "mumbai"
        }
        res = engine.score_event(event, update_state=False)
        self.assertIn("model_probability", res["raw_features"])
        if engine.pipeline is not None:
            self.assertIsNotNone(res["raw_features"]["model_probability"])
            self.assertAlmostEqual(res["risk_score"], res["raw_features"]["model_probability"] * 100.0, places=1)

    def test_redis_fallback_when_unreachable(self):
        store = RedisFeatureStore(redis_url="redis://invalid-host:6379/0")
        self.assertFalse(store.is_live_redis)
        c = store.increment_lifetime_counter("test", "id1")
        self.assertEqual(c, 1)

    def test_graceful_fallback_no_model(self):
        engine = FraudRiskEngine(model_path="nonexistent_model.joblib", warm_start=False)
        self.assertIsNone(engine.pipeline)
        event = {
            "name": "Fallback User",
            "email": "user@gmail.com",
            "ip_address": "10.0.0.1",
            "device_id": "dev_fb",
            "payment_token": "pm_fb",
            "area": "delhi"
        }
        res = engine.score_event(event, update_state=False)
        self.assertIn("verdict", res)
        self.assertIn("risk_score", res)


class TestThresholdAndMetrics(unittest.TestCase):
    """Tests that model threshold meets PRD Precision SLA."""

    def test_threshold_satisfies_precision_sla(self):
        test_csv = os.path.join(BASE_DIR, "data", "processed", "test_set.csv")
        metrics_json = os.path.join(BASE_DIR, "results", "final_metrics.json")
        self.assertTrue(os.path.exists(test_csv), "test_set.csv must exist!")
        self.assertTrue(os.path.exists(metrics_json), "final_metrics.json must exist!")
        
        with open(metrics_json, "r") as f:
            metrics = json.load(f)
        threshold = float(metrics["decision_threshold"])
        
        df = pd.read_csv(test_csv)
        engine = FraudRiskEngine(warm_start=False)
        X = df[FEATURE_COLS].values
        probs = engine.pipeline.predict_proba(X)[:, 1]
        
        preds = (probs >= threshold).astype(int)
        y = df["is_repeat_user"].values
        
        tp = np.sum((preds == 1) & (y == 1))
        fp = np.sum((preds == 1) & (y == 0))
        precision = tp / max(1, (tp + fp))
        recall = tp / np.sum(y == 1)

        self.assertGreaterEqual(precision, 0.75, f"Precision SLA violated! Got: {precision:.4f}")
        self.assertGreaterEqual(recall, 0.90, f"Recall too low! Got: {recall:.4f}")


class TestDriftMath(unittest.TestCase):
    """Tests Population Stability Index calculation accuracy."""

    def test_identical_distributions_have_zero_psi(self):
        arr = np.random.normal(loc=10.0, scale=2.0, size=1000)
        psi = calculate_psi(arr, arr)
        self.assertAlmostEqual(psi, 0.0, places=2)

    def test_shifted_distributions_flag_drift(self):
        arr_base = np.random.normal(loc=0.0, scale=1.0, size=1000)
        arr_shifted = np.random.normal(loc=3.0, scale=1.0, size=1000)
        psi = calculate_psi(arr_base, arr_shifted)
        self.assertGreater(psi, 0.25, "Shifted distribution should have PSI >= 0.25!")


class TestFastAPIEndpoints(unittest.TestCase):
    """Tests FastAPI microservice endpoints with Multi-Tenant Auth and 30 req/min Rate Limiting."""

    def setUp(self):
        import secrets
        from fastapi.testclient import TestClient
        from api import app, rate_limiter
        import api as _api
        rate_limiter.history.clear()
        # Reset engine so tests don't inherit stale production warm-start state
        _api.engine = None
        _api.engine = FraudRiskEngine(warm_start=False)
        self.client = TestClient(app)
        self.uid = f"usr_test_{secrets.token_hex(4)}"
        # Register a test tenant
        resp = self.client.post("/api/v1/auth/session", json={
            "uid": self.uid,
            "email": f"{self.uid}@enterprise.io",
            "display_name": "Test Tenant"
        })
        self.assertEqual(resp.status_code, 200)
        # Generate API key on-demand
        key_resp = self.client.post("/api/v1/keys/create", json={
            "user_id": self.uid,
            "name": "Test Tenant Key",
            "key_type": "live"
        })
        self.assertEqual(key_resp.status_code, 200)
        self.tenant_key = key_resp.json()["api_key"]
        self.headers = {"X-API-Key": self.tenant_key}

    def test_healthz_endpoint(self):
        resp = self.client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "healthy")
        self.assertEqual(resp.json()["rate_limit_per_min"], 30)

    def test_score_genuine_user(self):
        import secrets as _s
        _run_id = _s.token_hex(6)
        payload = {
            "name": "David Smith",
            "email": f"david.smith.{_run_id}@gmail.com",
            "ip_address": "203.0.113.45",
            "device_id": f"dev_unique_mac_{_run_id}",
            "payment_token": f"pm_unique_card_{_run_id}",
            "area": "london",
            "payment_country": "GB",
            "signup_time": "2026-07-15 14:30:00"
        }
        resp = self.client.post("/api/v1/score", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["verdict"], "NEW USER (GENUINE)")
        self.assertEqual(data["recommended_action"], "ALLOW")
        self.assertLess(data["risk_score"], 25.0)
        self.assertIn("customer_id", data)

    def test_score_fraud_syndicate(self):
        payload = {
            "name": "Akash Verma",
            "email": "akash.verma404+trial3@guerrillamail.com",
            "ip_address": "88.189.145.12",
            "device_id": "460f1adf042934c1",
            "payment_token": "pm_9d3f935e045d",
            "area": "delhi"
        }
        resp = self.client.post("/api/v1/score", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["verdict"], "REPEATING USER (LIKELY ABUSE)")
        self.assertEqual(data["recommended_action"], "BLOCK / REQUIRE PAYMENT")
        self.assertGreaterEqual(data["risk_score"], 10.0)

    def test_zero_shot_unseen_attacker(self):
        engine_cold = FraudRiskEngine(warm_start=False)
        payload = {
            "name": "New Attacker",
            "email": "attacker+trial1@mailinator.com",
            "ip_address": "198.51.100.99",
            "device_id": "dev_fresh_attacker",
            "payment_token": "pm_fresh_card",
            "area": "mumbai",
            "payment_country": "US"
        }
        res = engine_cold.score_event(payload, update_state=False)
        self.assertIn(res["verdict"], ["SUSPICIOUS (STEP-UP)", "REPEATING USER (LIKELY ABUSE)"])

    def test_api_key_creation_and_listing(self):
        # Create a new API key for this user
        resp = self.client.post("/api/v1/keys/create", json={"user_id": self.uid, "name": "New Key", "key_type": "live"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["api_key"].startswith("fk_live_"))
        self.assertEqual(data["rate_limit_per_min"], 30)

        # List keys
        list_resp = self.client.get(f"/api/v1/keys/list?user_id={self.uid}")
        self.assertEqual(list_resp.status_code, 200)
        keys = list_resp.json()["keys"]
        self.assertGreaterEqual(len(keys), 1)

    def test_rate_limit_headers_and_429_enforcement(self):
        create_resp = self.client.post("/api/v1/keys/create", json={"user_id": self.uid, "name": "Rate Limit Test", "key_type": "test"})
        test_key = create_resp.json()["api_key"]

        payload = {
            "name": "Rate Test User",
            "email": "rate@gmail.com",
            "ip_address": "1.1.1.1",
            "device_id": "d_rate",
            "payment_token": "p_rate",
            "area": "london"
        }

        # First request should return 200 with strict 30 rate limit headers
        resp = self.client.post("/api/v1/score", json=payload, headers={"X-API-Key": test_key})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("x-ratelimit-limit"), "30")
        self.assertIn("x-ratelimit-remaining", resp.headers)
        self.assertIn("x-ratelimit-reset", resp.headers)

        # Saturate 30 req/min limit
        from api import rate_limiter
        rate_limiter.history[test_key] = [time.time()] * 35

        exceeded_resp = self.client.post("/api/v1/score", json=payload, headers={"X-API-Key": test_key})
        self.assertEqual(exceeded_resp.status_code, 429)
        self.assertIn("rate_limit_exceeded", exceeded_resp.json()["detail"]["error"])
        self.assertIn("retry_after_seconds", exceeded_resp.json()["detail"])


class TestLegalComplianceAndFooter(unittest.TestCase):
    """Tests for Single Source of Truth Legal Docs, API Endpoints, and Footer UI."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from api import app
        from legal_loader import load_legal_documents, get_legal_document
        cls.client = TestClient(app)
        cls.load_legal_documents = staticmethod(load_legal_documents)
        cls.get_legal_document = staticmethod(get_legal_document)

    def test_canonical_legal_documents_exist_and_parse(self):
        docs = self.load_legal_documents(include_content=True)
        self.assertEqual(len(docs), 7, "Expected 7 canonical legal documents in /legal")

        expected_slugs = ["privacy", "terms", "cookies", "refund", "aup", "dpa", "disclaimer"]
        loaded_slugs = [d["slug"] for d in docs]
        self.assertEqual(loaded_slugs, expected_slugs)

        for doc in docs:
            self.assertTrue(len(doc["title"]) > 0)
            self.assertTrue(len(doc["version"]) > 0)
            self.assertTrue(len(doc["effective_date"]) > 0)
            self.assertTrue(len(doc["summary"]) > 0)
            self.assertTrue(len(doc["content"]) > 100)

    def test_get_individual_legal_document_and_aliases(self):
        privacy_doc = self.get_legal_document("privacy")
        self.assertIsNotNone(privacy_doc)
        self.assertEqual(privacy_doc["slug"], "privacy")
        self.assertEqual(privacy_doc["title"], "Privacy Policy")

        # Test alias resolution
        aliased = self.get_legal_document("privacy-policy")
        self.assertIsNotNone(aliased)
        self.assertEqual(aliased["slug"], "privacy")

        # Test non-existent slug
        missing = self.get_legal_document("nonexistent_policy")
        self.assertIsNone(missing)

    def test_fastapi_legal_endpoints(self):
        # List all documents
        resp = self.client.get("/api/v1/legal/documents")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("documents", data)
        self.assertEqual(len(data["documents"]), 7)

        # Fetch individual document
        for slug in ["privacy", "terms", "cookies", "refund", "aup", "dpa", "disclaimer"]:
            doc_resp = self.client.get(f"/api/v1/legal/{slug}")
            self.assertEqual(doc_resp.status_code, 200)
            doc_data = doc_resp.json()
            self.assertEqual(doc_data["slug"], slug)
            self.assertIn("content", doc_data)

        # 404 on missing document
        missing_resp = self.client.get("/api/v1/legal/fake_document_slug")
        self.assertEqual(missing_resp.status_code, 404)

    def test_legal_direct_page_routes(self):
        for route in ["/legal", "/privacy", "/terms", "/cookies", "/refund", "/aup", "/dpa", "/disclaimer"]:
            resp = self.client.get(route)
            self.assertEqual(resp.status_code, 200)
            self.assertIn("text/html", resp.headers.get("content-type", ""))

    def test_html_page_footer_and_legal_center(self):
        from app import HTML_PAGE
        self.assertIn("site-footer", HTML_PAGE)
        self.assertIn("karthik tatineni", HTML_PAGE)
        self.assertIn("legal-modal", HTML_PAGE)
        self.assertIn("cookie-consent-banner", HTML_PAGE)
        self.assertIn("openLegalDoc", HTML_PAGE)
        self.assertIn("hasAnalyticsConsent", HTML_PAGE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
