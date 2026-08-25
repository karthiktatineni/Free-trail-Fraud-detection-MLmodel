"""
CALIBRATED MULTI-SIGNAL FRAUD RISK SCORING ENGINE
=================================================
Calculates intuitive, transparent 0-100 Risk Score based on additive signals:
1. Payment Method Reuse (+30 pts max)
2. Device ID Reuse (+20 pts max)
3. Accounts Created by Device in 1 Hour (+10 pts max)
4. IP Address Reuse (+15 pts max) & Subnet (+5 pts max)
5. Email Domain / Disposable / Plus-Tag (+10 pts max)
6. Name Similarity with Existing Accounts (+5 pts max)
7. Area / Payment BIN Country Mismatch (+5 pts max)

Verdict Calibration:
- Score  0 - 25 : NEW USER (GENUINE)        -> Action: ALLOW
- Score 25 - 50 : SUSPICIOUS (STEP-UP)       -> Action: REQUIRE 2FA / CAPTCHA
- Score 50 - 100: REPEATING USER (ABUSE)     -> Action: BLOCK / REQUIRE PAYMENT
"""

import os
import json
import argparse
from difflib import SequenceMatcher
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
TRAIN_DATA_PATH = os.path.join(DATA_RAW_DIR, "raw_signup_events.csv")

DISPOSABLE_DOMAINS = {
    "mailinator.com", "tempmail.com", "10minutemail.com",
    "guerrillamail.com", "yopmail.com", "throwawaymail.com", "trashmail.com",
    "sharklasers.com", "getairmail.com", "maildrop.cc", "dispostable.com",
    "fakemailgenerator.com", "emailondeck.com", "mohmal.com", "crazymailing.com"
}
FREE_DOMAINS = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com", "protonmail.com"}

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


class IncrementalUnionFind:
    """Disjoint-Set Union (Union-Find) with path compression."""
    def __init__(self):
        self.parent = {}
        self.size = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        self.size.setdefault(x, 1)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def get_component_size(self, x: str) -> int:
        return self.size[self.find(x)]


class FraudRiskEngine:
    def __init__(self, model_path=MODEL_PATH, warm_start=True):
        self.pipeline = None
        if os.path.exists(model_path):
            try:
                self.pipeline = joblib.load(model_path)
            except Exception:
                pass
        
        self.graph = IncrementalUnionFind()
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
        return self.graph.find(x)

    def _union(self, a, b):
        self.graph.union(a, b)

    def _component_size(self, x):
        return self.graph.get_component_size(x)

    def _count_and_prune(self, window_dict, key, now):
        lst = window_dict.get(key, [])
        lst = [ts for ts in lst if (now - ts).total_seconds() <= 24 * 3600]
        cnt = len(lst)
        window_dict[key] = lst
        return cnt

    def _warm_start_from_history(self, history_csv):
        df_hist = pd.read_csv(history_csv, parse_dates=["signup_time"])
        df_hist = df_hist.sort_values("signup_time").reset_index(drop=True)
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
        if self.seen_name and name_norm:
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
        """
        Computes calibrated 0-100 Risk Score with both Zero-Shot Intrinsic
        and Causal Historical Entity Linkage signals.
        """
        feat_dict = self.extract_features(event, update_state=update_state)
        signal_breakdown = {}
        
        # --- HISTORICAL REUSE SIGNALS ---
        # A. Payment Method Reuse (Weight: 30)
        pay_reuse = feat_dict["payment_reuse_count"]
        pay_pts = min(30.0, pay_reuse * 30.0)
        signal_breakdown["same_payment_method"] = round(pay_pts, 1)

        # B. Device ID Reuse (Weight: 20)
        dev_reuse = feat_dict["device_reuse_count"]
        dev_pts = min(20.0, dev_reuse * 20.0)
        signal_breakdown["device_used_to_login"] = round(dev_pts, 1)

        # C. Accounts Created by Device in 1 Hour (Weight: 10)
        dev_1h = feat_dict["device_signups_last_hour"]
        dev_1h_pts = min(10.0, dev_1h * 5.0)
        signal_breakdown["device_hourly_velocity"] = round(dev_1h_pts, 1)

        # D. IP Address Reuse (Weight: 15) & Subnet (Weight: 5)
        ip_reuse = feat_dict["ip_reuse_count"]
        ip_pts = min(15.0, ip_reuse * 15.0)
        signal_breakdown["ip_address_reuse"] = round(ip_pts, 1)
        
        subnet_reuse = feat_dict["ip_subnet_reuse_count"]
        subnet_pts = min(5.0, (1.0 if subnet_reuse >= 3 else 0.0) * 5.0)
        signal_breakdown["ip_subnet_reuse"] = round(subnet_pts, 1)

        # --- ZERO-SHOT INTRINSIC SIGNALS (No Prior History Required) ---
        # E. Disposable / Burner Email Domain (Weight: 20 max for zero-shot accuracy)
        disp_email = feat_dict["is_disposable_email_domain"]
        plus_tag = feat_dict["email_local_has_plus_tag"]
        email_digits = feat_dict["email_local_has_digits"]
        email_pts = (20.0 if disp_email else 0.0) + (5.0 if plus_tag else 0.0) + (5.0 if email_digits and not plus_tag else 0.0)
        signal_breakdown["email_domain_risk"] = min(20.0, round(email_pts, 1))

        # F. Name Similarity Score (Weight: 5)
        name_sim = feat_dict["name_similarity_score"]
        name_pts = 5.0 if name_sim >= 0.85 else 0.0
        signal_breakdown["name_similarity"] = round(name_pts, 1)

        # G. Area / BIN Country Mismatch Zero-Shot Check (Weight: 15 max)
        geo_mismatch = feat_dict["payment_ip_country_mismatch"]
        geo_pts = 15.0 if geo_mismatch else 0.0
        signal_breakdown["area_geo_mismatch"] = round(geo_pts, 1)

        # Total Additive Risk Score
        total_score = sum(signal_breakdown.values())
        risk_score = round(min(100.0, max(0.0, total_score)), 1)

        # Verdict and Action Calibration
        if risk_score < 25.0:
            verdict = "NEW USER (GENUINE)"
            action = "ALLOW"
            severity = "low"
            confidence = round(100.0 - risk_score, 1)
        elif risk_score < 50.0:
            verdict = "SUSPICIOUS (STEP-UP)"
            action = "STEP-UP / MANUAL REVIEW"
            severity = "medium"
            confidence = round(max(risk_score, 100.0 - risk_score), 1)
        else:
            verdict = "REPEATING USER (LIKELY ABUSE)"
            action = "BLOCK / REQUIRE PAYMENT"
            severity = "high"
            confidence = round(risk_score, 1)

        signal_breakdown = dict(sorted(signal_breakdown.items(), key=lambda x: x[1], reverse=True))

        return {
            "user_id": event.get("user_id", "unseen_user"),
            "risk_score": risk_score,
            "verdict": verdict,
            "recommended_action": action,
            "severity": severity,
            "model_confidence_pct": confidence,
            "decision_threshold": 50.0,
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
        df_out = pd.DataFrame(results)
        if output_csv_path:
            df_out.to_csv(output_csv_path, index=False)
            print(f"Saved predictions to {output_csv_path}")
        return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time & Batch Fraud Risk Scoring CLI")
    parser.add_argument("--name", type=str, default="Rahul Sharma", help="User Name")
    parser.add_argument("--email", type=str, default="rahul.sharma@gmail.com", help="Email Address")
    parser.add_argument("--ip", type=str, default="103.21.124.50", help="IP Address")
    parser.add_argument("--device", type=str, default="dev_pixel_99", help="Device Fingerprint Hash")
    parser.add_argument("--payment", type=str, default="pm_visa_4412", help="Payment Card Token")
    parser.add_argument("--area", type=str, default="mumbai", help="City / Region")
    parser.add_argument("--csv", type=str, default=None, help="Path to input CSV for batch prediction")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to output CSV")
    args = parser.parse_args()

    engine = FraudRiskEngine(warm_start=True)

    if args.csv:
        engine.score_batch_csv(args.csv, args.output)
    else:
        sample_event = {
            "name": args.name,
            "email": args.email,
            "ip_address": args.ip,
            "device_id": args.device,
            "payment_token": args.payment,
            "area": args.area
        }
        result = engine.score_event(sample_event, update_state=True)
        print("\n" + "=" * 60)
        print(f"FRAUD RISK ASSESSMENT RESULT: {result['verdict']}")
        print("=" * 60)
        print(f"Risk Score          : {result['risk_score']} / 100.0")
        print(f"Model Confidence    : {result['model_confidence_pct']}%")
        print(f"Recommended Action  : {result['recommended_action']}")
        print(f"Decision Threshold  : {result['decision_threshold']}")
        print("-" * 60)
        print("TOP CONTRIBUTING RISK SIGNALS:")
        for sig, val in result['signal_breakdown'].items():
            if val > 0:
                print(f"  * {sig:<28}: +{val:.1f} pts")
        print("=" * 60)
