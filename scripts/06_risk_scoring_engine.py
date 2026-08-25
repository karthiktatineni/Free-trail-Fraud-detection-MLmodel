"""
STEP 6 - Risk Scoring Engine
==============================
Per PRD Section 8:
  - Load trained model and compute risk score (0-100) for each event
  - Three-band verdict system:
    * < 0.55 * threshold  -> NEW / GENUINE (ALLOW)
    * 0.55 * threshold to threshold -> SUSPICIOUS (STEP-UP / REVIEW)
    * >= threshold -> REPEAT / LIKELY ABUSE (BLOCK / DEMAND PAYMENT)
  - Per-signal explanation (feature value * global feature importance)
  - Demo output on 5 sample events (2 genuine, 3 abuse)

Input:  models/best_model.joblib
        data/processed/features_v2.csv
        results/final_metrics.json
Output: results/risk_scoring_demo.json
        data/processed/scored_dataset.csv
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for d in [RESULTS_DIR, DATA_PROCESSED_DIR]:
    os.makedirs(d, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
FEATURES_PATH = os.path.join(DATA_PROCESSED_DIR, "features_v2.csv")
METRICS_PATH = os.path.join(RESULTS_DIR, "final_metrics.json")

pipeline = joblib.load(MODEL_PATH)
df = pd.read_csv(FEATURES_PATH)

with open(METRICS_PATH) as f:
    metrics = json.load(f)

threshold = metrics["decision_threshold"]

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

clf = pipeline.named_steps["clf"]
if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
elif hasattr(clf, "coef_"):
    importances = np.abs(clf.coef_[0])
else:
    importances = np.ones(len(FEATURE_COLS))

importance_dict = dict(zip(FEATURE_COLS, importances))


def score_event(row):
    """Score a single signup event and return structured risk output per PRD Section 8."""
    X = row[FEATURE_COLS].values.reshape(1, -1)
    proba = pipeline.predict_proba(X)[0, 1]
    risk_score = min(proba * 100, 100.0)

    threshold_pct = threshold * 100
    if risk_score < 0.55 * threshold_pct:
        verdict = "NEW / GENUINE"
    elif risk_score < threshold_pct:
        verdict = "SUSPICIOUS -- MANUAL REVIEW"
    else:
        verdict = "REPEAT / LIKELY ABUSE"

    signal_breakdown = {}
    for feat in FEATURE_COLS:
        val = float(row[feat])
        imp = float(importance_dict[feat])
        signal_breakdown[feat] = round(val * imp * 100, 2)

    signal_breakdown = dict(sorted(
        signal_breakdown.items(), key=lambda x: abs(x[1]), reverse=True
    ))

    return {
        "risk_score": round(float(risk_score), 1),
        "verdict": verdict,
        "model_confidence_pct": round(float(max(proba, 1 - proba) * 100), 1),
        "decision_threshold": round(float(threshold * 100), 1),
        "signal_breakdown": signal_breakdown,
    }


# Demo: Score 5 Sample Events
genuine_samples = df[df["is_repeat_user"] == 0].sample(2, random_state=42)
abuse_samples = df[df["is_repeat_user"] == 1].sample(3, random_state=42)
demo_df = pd.concat([genuine_samples, abuse_samples])

demo_results = []
for _, row in demo_df.iterrows():
    result = score_event(row)
    result["ground_truth"] = "New/Genuine" if row["is_repeat_user"] == 0 else "Repeat/Abuse"
    result["user_id"] = row["user_id"]
    demo_results.append(result)

demo_path = os.path.join(RESULTS_DIR, "risk_scoring_demo.json")
with open(demo_path, "w") as f:
    json.dump(demo_results, f, indent=2)

print(f"{'='*60}")
print("RISK SCORING ENGINE -- DEMO OUTPUT")
print(f"{'='*60}")

for r in demo_results:
    print(f"\n  User: {r['user_id']}")
    print(f"  Risk Score: {r['risk_score']}")
    print(f"  Verdict: {r['verdict']}")
    print(f"  Confidence: {r['model_confidence_pct']}%")
    print(f"  Ground Truth: {r['ground_truth']}")
    top_signals = list(r["signal_breakdown"].items())[:5]
    print(f"  Top signals: {dict(top_signals)}")

print(f"\nSaved to {demo_path}")

print("\nScoring entire dataset...")
all_proba = pipeline.predict_proba(df[FEATURE_COLS].values)[:, 1]
df["risk_score"] = np.minimum(all_proba * 100, 100.0)
df["verdict"] = pd.cut(
    df["risk_score"],
    bins=[-1, 0.55 * threshold * 100, threshold * 100, 101],
    labels=["NEW / GENUINE", "SUSPICIOUS", "REPEAT / LIKELY ABUSE"]
)

print(f"\nVerdict distribution:")
print(df["verdict"].value_counts().to_string())

print(f"\nMean risk score by class:")
print(df.groupby("is_repeat_user")["risk_score"].mean().to_string())

scored_path = os.path.join(DATA_PROCESSED_DIR, "scored_dataset.csv")
df.to_csv(scored_path, index=False)
print(f"\nSaved full scored dataset to {scored_path}")

print(f"\n{'='*60}")
print("RISK SCORING COMPLETE")
print(f"{'='*60}")
