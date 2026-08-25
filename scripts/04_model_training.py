"""
STEP 4 - Model Training with 10-Fold Stratified Cross-Validation
================================================================
Per PRD Section 6:
  - 80/20 stratified train/test split
  - 10-fold Stratified K-Fold CV on training set
  - 7 algorithms compared: Logistic Regression, Decision Tree, Random Forest,
    Gradient Boosting, XGBoost, KNN, SVM (RBF)
  - Scoring: accuracy, precision, recall, F1, ROC-AUC, PR-AUC
  - Selection rule: rank by recall first, then F1, then ROC-AUC

Input:  data/processed/features_v2.csv
Output: models/best_model.joblib
        models/model_metadata.json
        results/cv_results.json
        data/processed/test_set.csv
        visuals/evaluation/model_comparison.png
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

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    make_scorer, precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VISUALS_EVAL_DIR = os.path.join(BASE_DIR, "visuals", "evaluation")

for d in [DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, VISUALS_EVAL_DIR]:
    os.makedirs(d, exist_ok=True)

FEATURES_PATH = os.path.join(DATA_PROCESSED_DIR, "features_v2.csv")

df = pd.read_csv(FEATURES_PATH)
print(f"Loaded {len(df)} rows from {FEATURES_PATH}")

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
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Train abuse rate: {y_train.mean():.3f} | Test abuse rate: {y_test.mean():.3f}")

test_df = df.iloc[np.where(np.isin(np.arange(len(df)),
    train_test_split(np.arange(len(df)), test_size=0.20, random_state=42, stratify=y)[1]))[0]]
test_df.to_csv(os.path.join(DATA_PROCESSED_DIR, "test_set.csv"), index=False)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(random_state=42, max_depth=10))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]),
    "XGBoost": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="logloss",
            n_jobs=-1
        ))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7, n_jobs=-1))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42))
    ]),
}

scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = {}

print(f"\n{'='*70}")
print("10-FOLD STRATIFIED CROSS-VALIDATION")
print(f"{'='*70}")

for name, pipeline in models.items():
    print(f"\nTraining: {name}...")
    try:
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring,
            return_train_score=False, n_jobs=-1
        )
        result = {
            metric: {
                "mean": float(np.mean(scores[f"test_{metric}"])),
                "std": float(np.std(scores[f"test_{metric}"])),
                "per_fold": [float(v) for v in scores[f"test_{metric}"]]
            }
            for metric in scoring.keys()
        }
        cv_results[name] = result
        print(f"  F1={result['f1']['mean']:.4f}  Recall={result['recall']['mean']:.4f}  "
              f"ROC-AUC={result['roc_auc']['mean']:.4f}  PR-AUC={result['pr_auc']['mean']:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        cv_results[name] = {"error": str(e)}

cv_path = os.path.join(RESULTS_DIR, "cv_results.json")
with open(cv_path, "w") as f:
    json.dump(cv_results, f, indent=2)
print(f"\nSaved CV results to {cv_path}")

print(f"\n{'='*70}")
print("MODEL RANKING (by recall -> F1 -> ROC-AUC)")
print(f"{'='*70}")

valid_models = {k: v for k, v in cv_results.items() if "error" not in v}
ranking = sorted(valid_models.items(), key=lambda x: (
    x[1]["recall"]["mean"],
    x[1]["f1"]["mean"],
    x[1]["roc_auc"]["mean"]
), reverse=True)

for rank, (name, scores) in enumerate(ranking, 1):
    marker = " <-- SELECTED" if rank == 1 else ""
    print(f"  {rank}. {name}: Recall={scores['recall']['mean']:.4f}  "
          f"F1={scores['f1']['mean']:.4f}  ROC-AUC={scores['roc_auc']['mean']:.4f}{marker}")

best_model_name = ranking[0][0]
print(f"\nBest model: {best_model_name}")

print(f"\nRetraining {best_model_name} on full training set...")
best_pipeline = models[best_model_name]
best_pipeline.fit(X_train, y_train)

model_path = os.path.join(MODELS_DIR, "best_model.joblib")
joblib.dump(best_pipeline, model_path)
print(f"Saved trained model to {model_path}")

meta = {
    "model_name": best_model_name,
    "feature_columns": FEATURE_COLS,
    "target_column": TARGET_COL,
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "cv_recall_mean": float(ranking[0][1]["recall"]["mean"]),
    "cv_f1_mean": float(ranking[0][1]["f1"]["mean"]),
    "cv_roc_auc_mean": float(ranking[0][1]["roc_auc"]["mean"]),
}
with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\nGenerating model comparison chart...")
metrics_to_plot = ["f1", "recall", "roc_auc", "pr_auc"]
model_names = [name for name, _ in ranking]

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(model_names))
width = 0.18
colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

for i, metric in enumerate(metrics_to_plot):
    values = [valid_models[name][metric]["mean"] for name in model_names]
    stds = [valid_models[name][metric]["std"] for name in model_names]
    bars = ax.bar(x + i * width, values, width, label=metric.upper().replace("_", "-"),
                  color=colors[i], edgecolor="black", linewidth=0.3, yerr=stds, capsize=3)

ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("10-Fold CV: Model Comparison (Ranked by Recall)", fontsize=14, fontweight="bold")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names, rotation=25, ha="right")
ax.legend(loc="lower right")
ax.set_ylim(0.8, 1.0)
ax.grid(axis="y", alpha=0.3)

ax.axvspan(-0.5, 0.5 + width * 3, alpha=0.08, color="green")
ax.annotate("SELECTED", xy=(0 + width * 1.5, 0.81), ha="center",
            fontsize=10, fontweight="bold", color="green")

plt.tight_layout()
plt.savefig(os.path.join(VISUALS_EVAL_DIR, "model_comparison.png"), dpi=150)
plt.close()
print("Saved visuals/evaluation/model_comparison.png")

print(f"\n{'='*60}")
print("MODEL TRAINING COMPLETE")
print(f"{'='*60}")
