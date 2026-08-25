"""
INFERENCE ENGINE - Real-Time & Batch Scoring for Unseen Data
============================================================
Provides real-time scoring for single signup events or batch CSV prediction.
Maintains causal entity state (union-find graph linkage, rolling velocity windows,
and identity reuse counters) across events.

Usage:
  - CLI Single Prediction:
      py scripts/predict.py --name "John Doe" --email "john@mailinator.com" --ip "192.168.1.10" --device "dev_abc123" --payment "pm_987654" --area "mumbai"
  - CLI Batch Prediction:
      py scripts/predict.py --csv path/to/unseen_signups.csv --output path/to/predictions.csv
"""

import os
import json
import argparse
import hashlib
from difflib import SequenceMatcher
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
METRICS_PATH = os.path.join(RESULTS_DIR, "final_metrics.json")
TRAIN_DATA_PATH = os.path.join(DATA_RAW_DIR, "raw_signup_events.csv")

DISPOSABLE_DOMAINS = {
    "mailinator.com", "tempmail.com", "10minutemail.com",
    "guerrillamail.com", "yopmail.com"
}
FREE_DOMAINS = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com"}

AREA_TO_COUNTRY = {
    "mumbai": "IN", "delhi": "IN", "bangalore": "IN", "hyderabad": "IN",
    "chennai": "IN", "pune": "IN", "kolkata": "IN", "ahmedabad": "IN",
    "new_york": "US", "san_francisco": "US", "london": "GB",
    "toronto": "CA", "singapore": "SG", "dubai": "AE"
}

FEATURE_COLS = [
    "payment_reuse_count", "ip_reuse_count", "ip_subnet_reuse_count",
    "device_reuse_count", "device_signups_last_hour",
    "payment_signups_last_24h", "device_signups_last_24h", "subnet_signups_last_24h",
    "name_similarity_score",
    "is_disposable_email_domain", "is_free_email_domain",
    "email_local_has_digits", "email_local_has_plus_tag",
    "payment_ip_country_mismatch",
    "graph_component_size", "attrs_reused_count",
    "signup_hour", "is_odd_hour",
    "area_freq", "device_os_freq",
]


class FraudRiskEngine:
    def __init__(self, model_path=MODEL_PATH, metrics_path=METRICS_PATH, warm_start=True):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run training pipeline first.")
        self.pipeline = joblib.load(model_path)
        
        self.threshold = 0.055
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                self.threshold = metrics.get("decision_threshold", 0.055)
        
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            self.importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            self.importances = np.abs(clf.coef_[0])
        else:
            self.importances = np.ones(len(FEATURE_COLS))
        self.importance_dict = dict(zip(FEATURE_COLS, self.importances))
        
        self.parent = {}
        self.size = {}
        self.seen_payment = {}
        self.seen_ip = {}
        self.seen_subnet = {}
        self.seen_device = {}
        self.seen_name = {}
        self.hour_bucket_device = {}
        self.window_24h_payment = {}
        self.window_24h_device = {}
        self.window_24h_ip_subnet = {}
        
        self.area_counts = {
            "mumbai": 0.075, "delhi": 0.073, "bangalore": 0.074, "hyderabad": 0.072,
            "chennai": 0.071, "pune": 0.070, "kolkata": 0.069, "ahmedabad": 0.070,
            "new_york": 0.072, "san_francisco": 0.071, "london": 0.073,
            "toronto": 0.070, "singapore": 0.072, "dubai": 0.068
        }
        self.os_counts = {"android": 0.20, "ios": 0.20, "windows": 0.20, "macos": 0.20, "linux": 0.20}
        
        if warm_start and os.path.exists(TRAIN_DATA_PATH):
            self._warm_start_from_history(TRAIN_DATA_PATH)

    def _find(self, x):
        self.parent.setdefault(x, x)
        self.size.setdefault(x, 1)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def _union(self, a, b):
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def _component_size(self, x):
        return self.size[self._find(x)]

    def _count_and_prune(self, window_dict, key, now):
        lst = window_dict.get(key, [])
        lst = [ts for ts in lst if (now - ts).total_seconds() <= 24 * 3600]
        cnt = len(lst)
        window_dict[key] = lst
        return cnt

    def _warm_start_from_history(self, history_csv):
        df_hist = pd.read_csv(history_csv, parse_dates=["signup_time"])
        df_hist = df_hist.sort_values("signup_time").reset_index(drop=True)
        print(f"Warm-starting feature store with {len(df_hist)} historical events...")
        for _, row in df_hist.iterrows():
            pay = str(row["payment_token"])
            ip = str(row["ip_address"])
            subnet = ".".join(ip.split(".")[:3])
            dev = str(row["device_id"])
            name_norm = str(row["name"]).lower().strip()
            t = row["signup_time"]
            hour_bucket = t.floor("h")
            key = (dev, hour_bucket)

            self.seen_payment[pay] = self.seen_payment.get(pay, 0) + 1
            self.seen_ip[ip] = self.seen_ip.get(ip, 0) + 1
            self.seen_subnet[subnet] = self.seen_subnet.get(subnet, 0) + 1
            self.seen_device[dev] = self.seen_device.get(dev, 0) + 1
            self.seen_name[name_norm] = self.seen_name.get(name_norm, 0) + 1
            self.hour_bucket_device[key] = self.hour_bucket_device.get(key, 0) + 1
            self.window_24h_payment.setdefault(pay, []).append(t)
            self.window_24h_device.setdefault(dev, []).append(t)
            self.window_24h_ip_subnet.setdefault(subnet, []).append(t)
            self._union(pay, dev)
            self._union(dev, subnet)
        print("Feature store warmed up successfully.")

    def extract_features(self, event, update_state=True):
        name = str(event.get("name", "")).strip()
        name_norm = "".join([c for c in name.lower() if c.isalpha() or c == " "]).strip()
        email = str(event.get("email", "")).strip().lower()
        email_domain = email.split("@")[1] if "@" in email else str(event.get("email_domain", "")).lower()
        email_local = email.split("@")[0] if "@" in email else ""

        ip = str(event.get("ip_address", "127.0.0.1")).strip()
        subnet = ".".join(ip.split(".")[:3])
        dev = str(event.get("device_id", "dev_unknown")).strip()
        pay = str(event.get("payment_token", "pm_unknown")).strip()
        area = str(event.get("area", "mumbai")).strip().lower()
        os_name = str(event.get("device_os", "android")).strip().lower()

        t_val = event.get("signup_time")
        if t_val is None or pd.isna(t_val):
            t = pd.Timestamp.now()
        else:
            t = pd.Timestamp(t_val)
        hour_bucket = t.floor("h")
        key = (dev, hour_bucket)

        payment_reuse_count = self.seen_payment.get(pay, 0)
        ip_reuse_count = self.seen_ip.get(ip, 0)
        subnet_reuse_count = self.seen_subnet.get(subnet, 0)
        device_reuse_count = self.seen_device.get(dev, 0)

        payment_signups_24h = self._count_and_prune(self.window_24h_payment, pay, t)
        device_signups_24h = self._count_and_prune(self.window_24h_device, dev, t)
        subnet_signups_24h = self._count_and_prune(self.window_24h_ip_subnet, subnet, t)

        device_signups_1h = self.hour_bucket_device.get(key, 0)

        best_sim = 0.0
        if self.seen_name:
            candidates = list(self.seen_name.keys())[-300:]
            for cand in candidates:
                s = SequenceMatcher(None, name_norm, cand).ratio()
                if s > best_sim:
                    best_sim = s
                if best_sim > 0.97:
                    break

        is_disp = int(email_domain in DISPOSABLE_DOMAINS)
        is_free = int(email_domain in FREE_DOMAINS)
        email_has_digits = int(any(c.isdigit() for c in email_local))
        email_has_plus = int("+" in email_local)

        ip_country = AREA_TO_COUNTRY.get(area, "IN")
        payment_country = event.get("payment_country", ip_country)
        geo_mismatch = int(ip_country != payment_country)

        graph_size = max(
            self._component_size(pay),
            self._component_size(dev),
            self._component_size(subnet)
        )

        attrs_reused = (
            int(payment_reuse_count > 0) +
            int(subnet_reuse_count > 0) +
            int(device_reuse_count > 0) +
            int(best_sim > 0.85)
        )

        signup_hour = t.hour
        is_odd = int(signup_hour in [0, 1, 2, 3, 4, 5])

        area_freq = self.area_counts.get(area, 0.07)
        dev_freq = self.os_counts.get(os_name, 0.20)

        feat_dict = {
            "payment_reuse_count": payment_reuse_count,
            "ip_reuse_count": ip_reuse_count,
            "ip_subnet_reuse_count": subnet_reuse_count,
            "device_reuse_count": device_reuse_count,
            "device_signups_last_hour": device_signups_1h,
            "payment_signups_last_24h": payment_signups_24h,
            "device_signups_last_24h": device_signups_24h,
            "subnet_signups_last_24h": subnet_signups_24h,
            "name_similarity_score": round(best_sim, 3),
            "is_disposable_email_domain": is_disp,
            "is_free_email_domain": is_free,
            "email_local_has_digits": email_has_digits,
            "email_local_has_plus_tag": email_has_plus,
            "payment_ip_country_mismatch": geo_mismatch,
            "graph_component_size": graph_size,
            "attrs_reused_count": attrs_reused,
            "signup_hour": signup_hour,
            "is_odd_hour": is_odd,
            "area_freq": area_freq,
            "device_os_freq": dev_freq,
        }

        if update_state:
            self.seen_payment[pay] = payment_reuse_count + 1
            self.seen_ip[ip] = ip_reuse_count + 1
            self.seen_subnet[subnet] = subnet_reuse_count + 1
            self.seen_device[dev] = device_reuse_count + 1
            self.seen_name[name_norm] = self.seen_name.get(name_norm, 0) + 1
            self.hour_bucket_device[key] = device_signups_1h + 1
            self.window_24h_payment.setdefault(pay, []).append(t)
            self.window_24h_device.setdefault(dev, []).append(t)
            self.window_24h_ip_subnet.setdefault(subnet, []).append(t)
            self._union(pay, dev)
            self._union(dev, subnet)

        return feat_dict

    def score_event(self, event, update_state=True):
        feat_dict = self.extract_features(event, update_state=update_state)
        feature_vector = np.array([[feat_dict[col] for col in FEATURE_COLS]])
        
        proba = float(self.pipeline.predict_proba(feature_vector)[0, 1])
        risk_score = round(min(proba * 100, 100.0), 1)
        
        threshold_pct = self.threshold * 100
        if risk_score < 0.55 * threshold_pct:
            verdict = "NEW / GENUINE"
            action = "ALLOW"
            severity = "low"
        elif risk_score < threshold_pct:
            verdict = "SUSPICIOUS -- MANUAL REVIEW"
            action = "STEP-UP / MANUAL REVIEW"
            severity = "medium"
        else:
            verdict = "REPEAT / LIKELY ABUSE"
            action = "BLOCK / REQUIRE PAYMENT"
            severity = "high"
            
        confidence = round(float(max(proba, 1 - proba) * 100), 1)
        
        signal_breakdown = {}
        for feat in FEATURE_COLS:
            val = float(feat_dict[feat])
            imp = float(self.importance_dict[feat])
            contribution = round(val * imp * 100, 2)
            signal_breakdown[feat] = contribution
            
        signal_breakdown = dict(sorted(signal_breakdown.items(), key=lambda x: abs(x[1]), reverse=True))

        return {
            "user_id": event.get("user_id", "unseen_user"),
            "risk_score": risk_score,
            "verdict": verdict,
            "recommended_action": action,
            "severity": severity,
            "model_confidence_pct": confidence,
            "decision_threshold": round(float(threshold_pct), 1),
            "raw_features": feat_dict,
            "signal_breakdown": signal_breakdown,
        }

    def score_batch_csv(self, input_csv_path, output_csv_path=None):
        df_unseen = pd.read_csv(input_csv_path)
        print(f"Scoring {len(df_unseen)} unseen events from {input_csv_path}...")
        results = []
        for _, row in df_unseen.iterrows():
            res = self.score_event(row.to_dict(), update_state=True)
            results.append({
                "user_id": res["user_id"],
                "risk_score": res["risk_score"],
                "verdict": res["verdict"],
                "recommended_action": res["recommended_action"],
                "confidence_pct": res["model_confidence_pct"],
                "top_risk_signal": list(res["signal_breakdown"].keys())[0],
                "top_signal_score": list(res["signal_breakdown"].values())[0],
            })
        res_df = pd.DataFrame(results)
        merged = pd.concat([df_unseen.reset_index(drop=True), res_df], axis=1)
        if output_csv_path:
            merged.to_csv(output_csv_path, index=False)
            print(f"Saved batch scored predictions to {output_csv_path}")
        return merged


def main():
    parser = argparse.ArgumentParser(description="Real-Time Risk Scoring Engine for Unseen Signups")
    parser.add_argument("--name", type=str, default="Aarav Sharma", help="User self-reported name")
    parser.add_argument("--email", type=str, default="aarav.trial1@mailinator.com", help="Signup email")
    parser.add_argument("--ip", type=str, default="39.173.180.200", help="Signup IP address")
    parser.add_argument("--device", type=str, default="f21faa72fe17c06d", help="Device/Browser hash")
    parser.add_argument("--payment", type=str, default="pm_424776171fe7", help="Payment instrument token")
    parser.add_argument("--area", type=str, default="ahmedabad", help="City / Region")
    parser.add_argument("--os", type=str, default="android", help="Device OS")
    parser.add_argument("--csv", type=str, default=None, help="Path to input CSV for batch scoring")
    parser.add_argument("--output", type=str, default=None, help="Path to output CSV for batch results")
    args = parser.parse_args()

    engine = FraudRiskEngine(warm_start=True)

    if args.csv:
        out = args.output or os.path.join(BASE_DIR, "data", "processed", "unseen_scored_output.csv")
        res = engine.score_batch_csv(args.csv, out)
        print("\nFirst 5 predictions:")
        print(res[["user_id", "risk_score", "verdict", "recommended_action"]].head())
    else:
        sample_event = {
            "name": args.name,
            "email": args.email,
            "ip_address": args.ip,
            "device_id": args.device,
            "payment_token": args.payment,
            "area": args.area,
            "device_os": args.os,
            "signup_time": pd.Timestamp.now().isoformat()
        }
        res = engine.score_event(sample_event)
        print("\n" + "="*60)
        print("REAL-TIME FRAUD RISK ASSESSMENT")
        print("="*60)
        print(f"Risk Score:          {res['risk_score']} / 100")
        print(f"Verdict:             {res['verdict']}")
        print(f"Recommended Action:  {res['recommended_action']}")
        print(f"Confidence:          {res['model_confidence_pct']}%")
        print(f"Decision Threshold:  {res['decision_threshold']}")
        print("\nTop Contributing Risk Signals:")
        for k, v in list(res["signal_breakdown"].items())[:5]:
            print(f"  - {k:<28}: {v:>6.2f}")
        print("="*60)


if __name__ == "__main__":
    main()
