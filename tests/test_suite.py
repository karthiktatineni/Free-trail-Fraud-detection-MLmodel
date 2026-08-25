"""
AUTOMATED PRODUCTION TEST SUITE
================================
Unit and Integration Tests for:
1. Incremental Union-Find Disjoint-Set Graph
2. Causal Feature Store & Sliding 24-Hour Velocity
3. Precision SLA & Decision Threshold Invariants
4. Population Stability Index (PSI) Drift Math
5. FastAPI Microservice Endpoints
"""

import os
import sys
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
from scripts.redis_feature_store import RedisFeatureStore

class TestUnionFind(unittest.TestCase):
    """Tests for Incremental Union-Find Graph Component Tracking."""
    
    def test_single_node(self):
        uf = IncrementalUnionFind()
        self.assertEqual(uf.get_component_size("node_A"), 1)

    def test_connected_component_size(self):
        uf = IncrementalUnionFind()
        uf.union("card_1", "dev_1")
        uf.union("dev_1", "subnet_1")
        # All three are in one component
        self.assertEqual(uf.get_component_size("card_1"), 3)
        self.assertEqual(uf.get_component_size("dev_1"), 3)
        self.assertEqual(uf.get_component_size("subnet_1"), 3)

    def test_independent_clusters(self):
        uf = IncrementalUnionFind()
        # Cluster 1 (size 2)
        uf.union("c1", "d1")
        # Cluster 2 (size 3)
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
        store = RedisFeatureStore()
        t0 = 100000.0
        # Event 1 at t0 - 1000s (within 24h)
        store.record_sliding_event("vel:test", "e1", t0 - 1000)
        # Event 2 at t0 - 90000s (older than 24h: 86400s)
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
        # First query should see 0 prior counts
        res1 = engine.score_event(event, update_state=True)
        self.assertEqual(res1["raw_features"]["payment_reuse_count"], 0)
        self.assertEqual(res1["raw_features"]["graph_component_size"], 1)

        # Second query should see exactly 1 prior occurrence
        res2 = engine.score_event(event, update_state=False)
        self.assertEqual(res2["raw_features"]["payment_reuse_count"], 1)


class TestThresholdAndMetrics(unittest.TestCase):
    """Tests that model threshold meets PRD Precision SLA."""

    def test_threshold_satisfies_precision_sla(self):
        test_csv = os.path.join(BASE_DIR, "data", "processed", "test_set.csv")
        self.assertTrue(os.path.exists(test_csv), "test_set.csv must exist!")
        df = pd.read_csv(test_csv)
        
        engine = FraudRiskEngine(warm_start=False)
        X = df[FEATURE_COLS].values
        probs = engine.pipeline.predict_proba(X)[:, 1]
        
        threshold = 0.060
        preds = (probs >= threshold).astype(int)
        y = df["is_repeat_user"].values
        
        tp = np.sum((preds == 1) & (y == 1))
        fp = np.sum((preds == 1) & (y == 0))
        precision = tp / (tp + fp)
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
    """Tests FastAPI microservice endpoints."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from api import app
        self.client = TestClient(app)

    def test_healthz_endpoint(self):
        resp = self.client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "healthy")

    def test_score_genuine_user(self):
        payload = {
            "name": "David Smith",
            "email": "david.smith@gmail.com",
            "ip_address": "203.0.113.45",
            "device_id": "dev_unique_mac_101",
            "payment_token": "pm_unique_card_101",
            "area": "london",
            "payment_country": "GB",
            "signup_time": "2026-07-15 14:30:00"
        }
        resp = self.client.post("/api/v1/score", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["verdict"], "NEW / GENUINE")
        self.assertEqual(data["recommended_action"], "ALLOW")
        self.assertLess(data["risk_score"], 3.3)

    def test_score_fraud_syndicate(self):
        payload = {
            "name": "Sanjay Nair",
            "email": "sanjay.nair+trial4@mailinator.com",
            "ip_address": "39.173.180.190",
            "device_id": "f21faa72fe17c06d",
            "payment_token": "pm_424776171fe7",
            "area": "ahmedabad"
        }
        resp = self.client.post("/api/v1/score", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["verdict"], "REPEAT / LIKELY ABUSE")
        self.assertEqual(data["recommended_action"], "BLOCK / REQUIRE PAYMENT")
        self.assertGreaterEqual(data["risk_score"], 6.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
