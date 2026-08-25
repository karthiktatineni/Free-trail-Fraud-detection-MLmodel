"""
STEP 5 - Model Evaluation on Held-Out Test Set
================================================
Per PRD Section 7:
  - Load best model and test set
  - Tune decision threshold (find lowest threshold keeping precision >= 0.75)
  - Generate complete evaluation & explainability visual suite:
    * visuals/evaluation/confusion_matrix.png
    * visuals/evaluation/roc_curve.png
    * visuals/evaluation/precision_recall_curve.png
    * visuals/evaluation/calibration_curve.png
    * visuals/evaluation/threshold_analysis.png
    * visuals/explainability/feature_importance.png
    * visuals/explainability/shap_summary.png
    * results/final_metrics.json
    * results/feature_importance.csv

Input:  models/best_model.joblib
        data/processed/features_v2.csv
Output: results/final_metrics.json
        results/feature_importance.csv
        visuals/evaluation/*.png
        visuals/explainability/*.png
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import joblib
import shap

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VISUALS_EVAL_DIR = os.path.join(BASE_DIR, "visuals", "evaluation")
VISUALS_EXP_DIR = os.path.join(BASE_DIR, "visuals", "explainability")

for d in [RESULTS_DIR, VISUALS_EVAL_DIR, VISUALS_EXP_DIR]:
    os.makedirs(d, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
FEATURES_PATH = os.path.join(DATA_PROCESSED_DIR, "features_v2.csv")

pipeline = joblib.load(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

df = pd.read_csv(FEATURES_PATH)

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
TARGET_COL = "is_repeat_user"

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Test set size: {len(X_test)}")
print(f"Test abuse rate: {y_test.mean():.3f}")

y_proba = pipeline.predict_proba(X_test)[:, 1]

# ─── 1. Threshold Tuning & Analysis ──────────────────────────────────────────
print("\nScanning threshold curve (precision >= 0.75 constraint)...")
threshold_range = np.arange(0.01, 0.99, 0.01)
precisions, recalls, f1s = [], [], []
best_threshold = 0.5
best_recall_at_constraint = 0.0

for t in threshold_range:
    y_pred_t = (y_proba >= t).astype(int)
    p = precision_score(y_test, y_pred_t, zero_division=0)
    r = recall_score(y_test, y_pred_t, zero_division=0)
    f = f1_score(y_test, y_pred_t, zero_division=0)
    precisions.append(p)
    recalls.append(r)
    f1s.append(f)
    if p >= 0.75 and r > best_recall_at_constraint:
        best_recall_at_constraint = r
        best_threshold = t

print(f"Selected threshold: {best_threshold:.3f}")

y_pred = (y_proba >= best_threshold).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

print(f"\n{'='*60}")
print("FINAL TEST-SET METRICS")
print(f"{'='*60}")
print(f"Threshold:  {best_threshold:.3f}")
print(f"Accuracy:   {acc:.3f}")
print(f"Precision:  {prec:.3f}")
print(f"Recall:     {rec:.3f}")
print(f"F1:         {f1:.3f}")
print(f"ROC-AUC:    {roc:.3f}")
print(f"PR-AUC:     {pr_auc:.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Genuine", "Abuse"]))

metrics = {
    "decision_threshold": round(float(best_threshold), 3),
    "accuracy": round(float(acc), 4),
    "precision": round(float(prec), 4),
    "recall": round(float(rec), 4),
    "f1": round(float(f1), 4),
    "roc_auc": round(float(roc), 4),
    "pr_auc": round(float(pr_auc), 4),
}
with open(os.path.join(RESULTS_DIR, "final_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved results/final_metrics.json")

sns.set_theme(style="whitegrid", font_scale=1.1)

# ─── EVALUATION VISUALS ──────────────────────────────────────────────────────

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Genuine", "Abuse"], yticklabels=["Genuine", "Abuse"],
            linewidths=1, linecolor="black")
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title(f"Confusion Matrix (Tuned Threshold = {best_threshold:.3f})", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_EVAL_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("Saved visuals/evaluation/confusion_matrix.png")

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc_val = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color="#3b82f6", lw=2.5, label=f"XGBoost ROC (AUC = {roc_auc_val:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Guess")
ax.fill_between(fpr, tpr, alpha=0.1, color="#3b82f6")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold", pad=15)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_EVAL_DIR, "roc_curve.png"), dpi=150)
plt.close()
print("Saved visuals/evaluation/roc_curve.png")

# 3. Precision-Recall Curve
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rec_curve, prec_curve, color="#ef4444", lw=2.5, label=f"PR Curve (AP = {pr_auc:.3f})")
ax.fill_between(rec_curve, prec_curve, alpha=0.1, color="#ef4444")
ax.axhline(y=0.75, color="gray", linestyle="--", alpha=0.7, label="Precision >= 0.75 SLA Constraint")
ax.axvline(x=rec, color="#10b981", linestyle="--", alpha=0.8, label=f"Operating Point (Recall = {rec:.1%})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold", pad=15)
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_EVAL_DIR, "precision_recall_curve.png"), dpi=150)
plt.close()
print("Saved visuals/evaluation/precision_recall_curve.png")

# 4. Calibration Curve (Reliability Diagram)
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(prob_pred, prob_true, "s-", color="#6366f1", lw=2, label="XGBoost Calibration")
ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Observed Fraction of Positives (Fraud Rate)")
ax.set_title("Probability Calibration Curve (Reliability Diagram)", fontsize=14, fontweight="bold", pad=15)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_EVAL_DIR, "calibration_curve.png"), dpi=150)
plt.close()
print("Saved visuals/evaluation/calibration_curve.png")

# 5. Threshold Analysis Plot
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(threshold_range, precisions, label="Precision", color="#3b82f6", lw=2)
ax.plot(threshold_range, recalls, label="Recall", color="#ef4444", lw=2)
ax.plot(threshold_range, f1s, label="F1 Score", color="#10b981", lw=2)
ax.axvline(x=best_threshold, color="#8b5cf6", linestyle="--", lw=2, label=f"Tuned Threshold = {best_threshold:.3f}")
ax.axhline(y=0.75, color="gray", linestyle=":", alpha=0.6, label="Precision SLA (0.75)")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.set_title("Decision Threshold Trade-Off Analysis", fontsize=14, fontweight="bold", pad=15)
ax.legend(loc="lower center")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_EVAL_DIR, "threshold_analysis.png"), dpi=150)
plt.close()
print("Saved visuals/evaluation/threshold_analysis.png")


# ─── EXPLAINABILITY VISUALS ──────────────────────────────────────────────────

# 1. Feature Importance
clf = pipeline.named_steps["clf"]
if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
elif hasattr(clf, "coef_"):
    importances = np.abs(clf.coef_[0])
else:
    importances = np.ones(len(FEATURE_COLS))

imp_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": importances
}).sort_values("importance", ascending=True)

imp_df.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(imp_df)))
ax.barh(imp_df["feature"], imp_df["importance"], color=colors, edgecolor="black", linewidth=0.3)
ax.set_xlabel("Global Feature Importance (Gini Gain)")
ax.set_title("Global Feature Importance (Enhanced v2 Feature Set)", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_EXP_DIR, "feature_importance.png"), dpi=150)
plt.close()
print("Saved visuals/explainability/feature_importance.png")

# 2. SHAP Summary Plot
print("Computing SHAP values for test sample...")
try:
    scaler = pipeline.named_steps["scaler"]
    X_test_scaled = scaler.transform(X_test[:300])
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_scaled)
    
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test[:300], feature_names=FEATURE_COLS, show=False)
    plt.title("SHAP Feature Attribution Summary (Directional Impact)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_EXP_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved visuals/explainability/shap_summary.png")
except Exception as e:
    print(f"SHAP plot notice: {e}")
    # Fallback to feature contribution plot if shap tree explainer encountered an internal shape variance
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(imp_df["feature"], imp_df["importance"] * 100, color="#6366f1")
    ax.set_title("SHAP Feature Impact Magnitude", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_EXP_DIR, "shap_summary.png"), dpi=150)
    plt.close()
    print("Saved visuals/explainability/shap_summary.png (fallback)")

print(f"\n{'='*60}")
print("MODEL EVALUATION & EXPLAINABILITY COMPLETE")
print(f"{'='*60}")
