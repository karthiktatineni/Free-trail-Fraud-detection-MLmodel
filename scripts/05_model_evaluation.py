"""
STEP 5 - Model Evaluation: Threshold Tuning, Calibration & Fairness
====================================================================
  - Threshold selection on VALIDATION set only (not test set)
  - Cost-based threshold optimization (same C_FN/C_FP as model selection)
  - Calibration decision rule: ship CalibratedClassifierCV if Brier improves
  - Single evaluation on untouched TEST set with bootstrap CIs
  - Subgroup fairness analysis with minimum sample-size gate (n≥50)
  - Full visual suite: confusion matrix, ROC, PR, calibration, threshold

Input:  models/best_model.joblib
        data/processed/val_set.csv
        data/processed/test_set.csv
        data/processed/full_dataset_with_features.csv
Output: results/final_metrics.json
        results/fairness_analysis.json
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

from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    confusion_matrix, classification_report, brier_score_loss,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
import joblib

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VISUALS_EVAL_DIR = os.path.join(BASE_DIR, "visuals", "evaluation")
VISUALS_EXP_DIR = os.path.join(BASE_DIR, "visuals", "explainability")

for d in [RESULTS_DIR, VISUALS_EVAL_DIR, VISUALS_EXP_DIR]:
    os.makedirs(d, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
VAL_PATH = os.path.join(DATA_PROCESSED_DIR, "val_set.csv")
TEST_PATH = os.path.join(DATA_PROCESSED_DIR, "test_set.csv")
FULL_PATH = os.path.join(DATA_PROCESSED_DIR, "full_dataset_with_features.csv")

# --- Business Cost Parameters (same as 04_model_training.py) ---
COST_FN = 5.0
COST_FP = 1.0

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

# ─── Load Model & Data ──────────────────────────────────────────────────────
pipeline = joblib.load(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

df_val = pd.read_csv(VAL_PATH)
df_test = pd.read_csv(TEST_PATH)

X_val = df_val[FEATURE_COLS].values
y_val = df_val[TARGET_COL].values
X_test = df_test[FEATURE_COLS].values
y_test = df_test[TARGET_COL].values

print(f"Val set:  {len(X_val)} rows (abuse rate: {y_val.mean():.3f})")
print(f"Test set: {len(X_test)} rows (abuse rate: {y_test.mean():.3f})")


def compute_expected_cost(y_true, y_proba, threshold):
    """Compute expected cost at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    n = len(y_true)
    fn = np.sum((y_pred == 0) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return (fn * COST_FN + fp * COST_FP) / n


# ─── 1. Threshold Tuning on VALIDATION Set Only ─────────────────────────────
print(f"\n{'='*60}")
print("THRESHOLD TUNING (on validation set only)")
print(f"{'='*60}")

y_val_proba = pipeline.predict_proba(X_val)[:, 1]

threshold_range = np.arange(0.01, 0.99, 0.01)
costs = [compute_expected_cost(y_val, y_val_proba, t) for t in threshold_range]
cost_best_idx = np.argmin(costs)
cost_threshold = threshold_range[cost_best_idx]

# Also find precision-SLA threshold as reference
precisions_val, recalls_val = [], []
precision_sla_threshold = 0.5
best_recall_at_sla = 0.0
for t in threshold_range:
    y_pred_t = (y_val_proba >= t).astype(int)
    p = precision_score(y_val, y_pred_t, zero_division=0)
    r = recall_score(y_val, y_pred_t, zero_division=0)
    precisions_val.append(p)
    recalls_val.append(r)
    if p >= 0.75 and r > best_recall_at_sla:
        best_recall_at_sla = r
        precision_sla_threshold = t

print(f"Cost-optimal threshold:      {cost_threshold:.3f} (E[cost]={costs[cost_best_idx]:.4f})")
print(f"Precision-SLA threshold:     {precision_sla_threshold:.3f} (recall={best_recall_at_sla:.3f})")

# Use cost-optimal as the production threshold
best_threshold = cost_threshold
print(f"Selected production threshold: {best_threshold:.3f}")


# ─── 2. Calibration Decision ────────────────────────────────────────────────
print(f"\n{'='*60}")
print("CALIBRATION ANALYSIS")
print(f"{'='*60}")

brier_before = brier_score_loss(y_val, y_val_proba)
print(f"Brier score (before calibration): {brier_before:.4f}")

calibration_applied = False
calibration_method = None
brier_after = brier_before

# Try isotonic and sigmoid calibration
best_calibrated_pipeline = pipeline
for method in ["isotonic", "sigmoid"]:
    try:
        cal_clf = CalibratedClassifierCV(pipeline, method=method, cv=5)
        cal_clf.fit(X_val, y_val)
        cal_proba = cal_clf.predict_proba(X_val)[:, 1]
        cal_brier = brier_score_loss(y_val, cal_proba)
        print(f"  {method}: Brier = {cal_brier:.4f} ({'BETTER' if cal_brier < brier_before else 'worse'})")

        if cal_brier < brier_after:
            brier_after = cal_brier
            calibration_applied = True
            calibration_method = method
            best_calibrated_pipeline = cal_clf
    except Exception as e:
        print(f"  {method}: failed ({e})")

if calibration_applied:
    print(f"\nCalibration IMPROVED Brier score: {brier_before:.4f} -> {brier_after:.4f}")
    print(f"Shipping {calibration_method}-calibrated model as best_model.joblib")
    pipeline = best_calibrated_pipeline
    joblib.dump(pipeline, MODEL_PATH)

    # Re-tune threshold on calibrated probabilities
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    costs = [compute_expected_cost(y_val, y_val_proba, t) for t in threshold_range]
    cost_best_idx = np.argmin(costs)
    best_threshold = threshold_range[cost_best_idx]
    print(f"Re-tuned threshold on calibrated model: {best_threshold:.3f}")
else:
    print(f"\nCalibration did NOT improve Brier score. Keeping uncalibrated model.")


# ─── 3. FINAL Evaluation on Untouched Test Set ──────────────────────────────
print(f"\n{'='*60}")
print("FINAL TEST-SET EVALUATION (untouched until now)")
print(f"{'='*60}")

y_test_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_test_proba >= best_threshold).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_test_proba)
pr_auc_val = average_precision_score(y_test, y_test_proba)
brier_test = brier_score_loss(y_test, y_test_proba)

print(f"Threshold:  {best_threshold:.3f}")
print(f"Accuracy:   {acc:.3f}")
print(f"Precision:  {prec:.3f}")
print(f"Recall:     {rec:.3f}")
print(f"F1:         {f1:.3f}")
print(f"ROC-AUC:    {roc:.3f}")
print(f"PR-AUC:     {pr_auc_val:.3f}")
print(f"Brier:      {brier_test:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Genuine", "Abuse"]))


# ─── 4. Bootstrap Confidence Intervals ──────────────────────────────────────
print("Computing bootstrap 95% CIs (1000 resamples)...")
n_bootstrap = 1000
rng = np.random.default_rng(42)

boot_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
for _ in range(n_bootstrap):
    idx = rng.integers(0, len(y_test), size=len(y_test))
    y_b, p_b = y_test[idx], y_test_proba[idx]
    pred_b = (p_b >= best_threshold).astype(int)
    boot_metrics["accuracy"].append(accuracy_score(y_b, pred_b))
    boot_metrics["precision"].append(precision_score(y_b, pred_b, zero_division=0))
    boot_metrics["recall"].append(recall_score(y_b, pred_b, zero_division=0))
    boot_metrics["f1"].append(f1_score(y_b, pred_b, zero_division=0))
    try:
        boot_metrics["roc_auc"].append(roc_auc_score(y_b, p_b))
    except ValueError:
        boot_metrics["roc_auc"].append(np.nan)

ci_results = {}
for metric_name, values in boot_metrics.items():
    values = [v for v in values if not np.isnan(v)]
    ci_results[metric_name] = {
        "value": round(float(np.mean(values)), 4),
        "ci_95": [round(float(np.percentile(values, 2.5)), 4),
                  round(float(np.percentile(values, 97.5)), 4)]
    }
    print(f"  {metric_name}: {ci_results[metric_name]['value']:.4f} "
          f"[{ci_results[metric_name]['ci_95'][0]:.4f}, {ci_results[metric_name]['ci_95'][1]:.4f}]")


# ─── 5. Subgroup Fairness Analysis ──────────────────────────────────────────
print(f"\n{'='*60}")
print("SUBGROUP FAIRNESS ANALYSIS")
print(f"{'='*60}")

fairness_report = {"overall_fpr": None, "overall_fnr": None, "subgroups": {}, "min_sample_gate": 50}

# Get area info for test set rows
if os.path.exists(FULL_PATH):
    df_full = pd.read_csv(FULL_PATH)
    # Match test set rows by user_id
    test_user_ids = df_test["user_id"].values if "user_id" in df_test.columns else None

    if test_user_ids is not None and "area" in df_full.columns:
        test_areas = df_full[df_full["user_id"].isin(test_user_ids)].set_index("user_id").loc[test_user_ids, "area"].values
    elif "area" in df_test.columns:
        test_areas = df_test["area"].values
    else:
        test_areas = None

    if test_areas is not None:
        # Overall rates
        tn = np.sum((y_pred == 0) & (y_test == 0))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        tp = np.sum((y_pred == 1) & (y_test == 1))
        overall_fpr = fp / max(1, fp + tn)
        overall_fnr = fn / max(1, fn + tp)
        fairness_report["overall_fpr"] = round(float(overall_fpr), 4)
        fairness_report["overall_fnr"] = round(float(overall_fnr), 4)

        print(f"Overall FPR: {overall_fpr:.4f} | FNR: {overall_fnr:.4f}")
        print(f"Min sample gate: n >= {fairness_report['min_sample_gate']}")
        print(f"\n{'Area':<20} {'N':>5} {'FPR':>8} {'FNR':>8} {'Flag':>12}")
        print("-" * 55)

        for area in sorted(set(test_areas)):
            mask = test_areas == area
            n_area = mask.sum()

            if n_area < fairness_report["min_sample_gate"]:
                fairness_report["subgroups"][area] = {
                    "n": int(n_area), "status": "insufficient_data"
                }
                print(f"{area:<20} {n_area:>5} {'-':>8} {'-':>8} {'n<50':>12}")
                continue

            y_area = y_test[mask]
            pred_area = y_pred[mask]

            fp_a = np.sum((pred_area == 1) & (y_area == 0))
            tn_a = np.sum((pred_area == 0) & (y_area == 0))
            fn_a = np.sum((pred_area == 0) & (y_area == 1))
            tp_a = np.sum((pred_area == 1) & (y_area == 1))

            fpr_a = fp_a / max(1, fp_a + tn_a)
            fnr_a = fn_a / max(1, fn_a + tp_a)

            flag = ""
            if fpr_a > 2 * overall_fpr:
                flag = "HIGH_FPR"
            if fnr_a > 2 * overall_fnr:
                flag = flag + " HIGH_FNR" if flag else "HIGH_FNR"
            if not flag:
                flag = "OK"

            fairness_report["subgroups"][area] = {
                "n": int(n_area),
                "fpr": round(float(fpr_a), 4),
                "fnr": round(float(fnr_a), 4),
                "flag": flag
            }
            print(f"{area:<20} {n_area:>5} {fpr_a:>8.4f} {fnr_a:>8.4f} {flag:>12}")
    else:
        print("Could not join area info to test set.")
else:
    print(f"Full dataset not found at {FULL_PATH}")

fairness_path = os.path.join(RESULTS_DIR, "fairness_analysis.json")
with open(fairness_path, "w") as f:
    json.dump(fairness_report, f, indent=2)
print(f"Saved fairness report to {fairness_path}")


# ─── 6. Save Final Metrics ──────────────────────────────────────────────────
metrics = {
    "decision_threshold": round(float(best_threshold), 3),
    "threshold_method": "cost_optimized",
    "cost_fn": COST_FN,
    "cost_fp": COST_FP,
    "calibration_applied": calibration_applied,
    "calibration_method": calibration_method,
    "brier_score_before": round(float(brier_before), 4),
    "brier_score_after": round(float(brier_after), 4),
    "brier_score_test": round(float(brier_test), 4),
    "accuracy": ci_results["accuracy"],
    "precision": ci_results["precision"],
    "recall": ci_results["recall"],
    "f1": ci_results["f1"],
    "roc_auc": ci_results["roc_auc"],
    "pr_auc": round(float(pr_auc_val), 4),
}
with open(os.path.join(RESULTS_DIR, "final_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved results/final_metrics.json")

# ─── MLflow Logging ─────────────────────────────────────────────────────────
if HAS_MLFLOW:
    mlflow.set_experiment("fraud_detection")
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_param("threshold", best_threshold)
        mlflow.log_param("calibration_applied", calibration_applied)
        for m_name, m_val in ci_results.items():
            mlflow.log_metric(f"test_{m_name}", m_val["value"])
        mlflow.log_metric("test_brier", brier_test)
        mlflow.log_metric("test_pr_auc", pr_auc_val)

# ─── 7. Evaluation Visuals ──────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Genuine", "Abuse"], yticklabels=["Genuine", "Abuse"],
            linewidths=1, linecolor="black")
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title(f"Confusion Matrix (Threshold = {best_threshold:.3f})", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_EVAL_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("Saved visuals/evaluation/confusion_matrix.png")

# ROC Curve
fpr_arr, tpr_arr, _ = roc_curve(y_test, y_test_proba)
roc_auc_val = auc(fpr_arr, tpr_arr)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_arr, tpr_arr, color="#3b82f6", lw=2.5, label=f"ROC (AUC = {roc_auc_val:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Guess")
ax.fill_between(fpr_arr, tpr_arr, alpha=0.1, color="#3b82f6")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
roc_ci = ci_results["roc_auc"]["ci_95"]
ax.set_title(f"ROC Curve (AUC = {roc_auc_val:.3f} [{roc_ci[0]:.3f}, {roc_ci[1]:.3f}])",
             fontsize=14, fontweight="bold", pad=15)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_EVAL_DIR, "roc_curve.png"), dpi=150)
plt.close()
print("Saved visuals/evaluation/roc_curve.png")

# Precision-Recall Curve
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_test_proba)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rec_curve, prec_curve, color="#ef4444", lw=2.5, label=f"PR Curve (AP = {pr_auc_val:.3f})")
ax.fill_between(rec_curve, prec_curve, alpha=0.1, color="#ef4444")
ax.axhline(y=0.75, color="gray", linestyle="--", alpha=0.7, label="Precision >= 0.75 SLA")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold", pad=15)
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_EVAL_DIR, "precision_recall_curve.png"), dpi=150)
plt.close()
print("Saved visuals/evaluation/precision_recall_curve.png")

# Calibration Curve
prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=10)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(prob_pred, prob_true, "s-", color="#6366f1", lw=2, label="Model Calibration")
ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
cal_label = f" ({calibration_method})" if calibration_applied else " (uncalibrated)"
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Observed Fraction of Positives")
ax.set_title(f"Calibration Curve{cal_label} (Brier = {brier_test:.4f})",
             fontsize=14, fontweight="bold", pad=15)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_EVAL_DIR, "calibration_curve.png"), dpi=150)
plt.close()
print("Saved visuals/evaluation/calibration_curve.png")

# Threshold Analysis
val_prec_arr, val_rec_arr, val_f1_arr = [], [], []
val_cost_arr = []
for t in threshold_range:
    y_p = (y_val_proba >= t).astype(int)
    val_prec_arr.append(precision_score(y_val, y_p, zero_division=0))
    val_rec_arr.append(recall_score(y_val, y_p, zero_division=0))
    val_f1_arr.append(f1_score(y_val, y_p, zero_division=0))
    val_cost_arr.append(compute_expected_cost(y_val, y_val_proba, t))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.plot(threshold_range, val_prec_arr, label="Precision", color="#3b82f6", lw=2)
ax1.plot(threshold_range, val_rec_arr, label="Recall", color="#ef4444", lw=2)
ax1.plot(threshold_range, val_f1_arr, label="F1 Score", color="#10b981", lw=2)
ax1.axvline(x=best_threshold, color="#8b5cf6", linestyle="--", lw=2,
            label=f"Selected = {best_threshold:.3f}")
ax1.axhline(y=0.75, color="gray", linestyle=":", alpha=0.6, label="Precision SLA (0.75)")
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Score")
ax1.set_title("Threshold Trade-Off (Validation Set)", fontsize=14, fontweight="bold")
ax1.legend(loc="lower center")

ax2.plot(threshold_range, val_cost_arr, color="#f59e0b", lw=2.5,
         label=f"E[cost] (C_FN={COST_FN}, C_FP={COST_FP})")
ax2.axvline(x=best_threshold, color="#8b5cf6", linestyle="--", lw=2,
            label=f"Cost-optimal = {best_threshold:.3f}")
ax2.set_xlabel("Threshold")
ax2.set_ylabel("Expected Cost")
ax2.set_title("Cost-Based Threshold Selection", fontsize=14, fontweight="bold")
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(VISUALS_EVAL_DIR, "threshold_analysis.png"), dpi=150)
plt.close()
print("Saved visuals/evaluation/threshold_analysis.png")


# ─── 8. Explainability Visuals ──────────────────────────────────────────────

# Feature Importance
clf = pipeline.named_steps["clf"] if hasattr(pipeline, "named_steps") else pipeline
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
colors_imp = plt.cm.plasma(np.linspace(0.2, 0.9, len(imp_df)))
ax.barh(imp_df["feature"], imp_df["importance"], color=colors_imp, edgecolor="black", linewidth=0.3)
ax.set_xlabel("Global Feature Importance (Gini Gain)")
ax.set_title("Global Feature Importance", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_EXP_DIR, "feature_importance.png"), dpi=150)
plt.close()
print("Saved visuals/explainability/feature_importance.png")

# SHAP Summary Plot
if HAS_SHAP:
    print("Computing SHAP values for test sample...")
    try:
        underlying = pipeline
        if hasattr(pipeline, "estimator") and pipeline.estimator is not None:
            underlying = pipeline.estimator
        elif hasattr(pipeline, "calibrated_classifiers_") and len(pipeline.calibrated_classifiers_) > 0:
            underlying = getattr(pipeline.calibrated_classifiers_[0], "estimator", pipeline)

        if hasattr(underlying, "named_steps"):
            scaler = underlying.named_steps["scaler"]
            clf_shap = underlying.named_steps["clf"]
            X_shap = scaler.transform(X_test[:300])
        else:
            X_shap = X_test[:300]
            clf_shap = underlying

        if hasattr(clf_shap, "feature_importances_") or "Tree" in type(clf_shap).__name__ or "XGB" in type(clf_shap).__name__ or "Forest" in type(clf_shap).__name__:
            explainer = shap.TreeExplainer(clf_shap)
            shap_values = explainer.shap_values(X_shap)
        elif hasattr(clf_shap, "coef_"):
            explainer = shap.LinearExplainer(clf_shap, X_shap)
            shap_values = explainer.shap_values(X_shap)
        else:
            explainer = shap.Explainer(clf_shap.predict_proba, X_shap)
            shap_values = explainer(X_shap)
            if hasattr(shap_values, "values") and len(shap_values.values.shape) == 3:
                shap_values = shap_values.values[:, :, 1]

        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test[:300], feature_names=FEATURE_COLS, show=False)
        plt.title("SHAP Feature Attribution Summary", fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_EXP_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved visuals/explainability/shap_summary.png")
    except Exception as e:
        print(f"SHAP plot skipped: {e}")
        # Fallback bar chart of importances
        fig, ax = plt.subplots(figsize=(10, 8))
        colors_imp = plt.cm.plasma(np.linspace(0.2, 0.9, len(imp_df)))
        ax.barh(imp_df["feature"], imp_df["importance"], color=colors_imp, edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Feature Importance Magnitude")
        ax.set_title("Feature Attribution Magnitude", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_EXP_DIR, "shap_summary.png"), dpi=150)
        plt.close()
        print("Saved visuals/explainability/shap_summary.png (fallback)")

print(f"\n{'='*60}")
print("MODEL EVALUATION & EXPLAINABILITY COMPLETE")
print(f"{'='*60}")
