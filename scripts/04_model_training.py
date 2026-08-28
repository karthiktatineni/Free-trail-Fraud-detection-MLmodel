"""
STEP 4 - Model Training with Cost-Based Selection & Hyperparameter Search
==========================================================================
  - 70/15/15 stratified train/val/test split
  - 7 algorithms compared via 10-fold Stratified K-Fold CV
  - Cost-based model selection: minimize E[cost] = FN_rate × C_FN + FP_rate × C_FP
    (same objective used for threshold tuning in script 05)
  - Optuna hyperparameter search for the winning model
  - Paired bootstrap significance test between top-2 models
  - MLflow experiment logging

Input:  data/processed/features_v2.csv
Output: models/best_model.joblib
        models/model_metadata.json
        results/cv_results.json
        data/processed/train_set.csv
        data/processed/val_set.csv
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
    make_scorer, precision_score, recall_score, f1_score,
    roc_auc_score
)
from xgboost import XGBClassifier
import joblib

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

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

for d in [DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, VISUALS_EVAL_DIR]:
    os.makedirs(d, exist_ok=True)

FEATURES_PATH = os.path.join(DATA_PROCESSED_DIR, "features_v2.csv")

# --- Business Cost Parameters (shared with 05_model_evaluation.py) ---
COST_FN = 5.0  # Cost of missed abuser: infra + revenue loss
COST_FP = 1.0  # Cost of false block: friction + churn

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

# ─── Load Data ────────────────────────────────────────────────────────────────
df = pd.read_csv(FEATURES_PATH)
print(f"Loaded {len(df)} rows from {FEATURES_PATH}")

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

# ─── 70 / 15 / 15 Stratified Split ───────────────────────────────────────────
X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
    X, y, np.arange(len(df)), test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
    X_temp, y_temp, idx_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Train abuse rate: {y_train.mean():.3f} | Val: {y_val.mean():.3f} | Test: {y_test.mean():.3f}")

# Save splits
df.iloc[idx_train].to_csv(os.path.join(DATA_PROCESSED_DIR, "train_set.csv"), index=False)
df.iloc[idx_val].to_csv(os.path.join(DATA_PROCESSED_DIR, "val_set.csv"), index=False)
df.iloc[idx_test].to_csv(os.path.join(DATA_PROCESSED_DIR, "test_set.csv"), index=False)
print("Saved train_set.csv, val_set.csv, test_set.csv")


def compute_expected_cost(y_true, y_proba, threshold):
    """Compute expected cost at a given threshold using business costs."""
    y_pred = (y_proba >= threshold).astype(int)
    n = len(y_true)
    fn = np.sum((y_pred == 0) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return (fn * COST_FN + fp * COST_FP) / n


def find_optimal_threshold(y_true, y_proba):
    """Find threshold minimizing expected cost."""
    thresholds = np.arange(0.01, 0.99, 0.01)
    costs = [compute_expected_cost(y_true, y_proba, t) for t in thresholds]
    best_idx = np.argmin(costs)
    return thresholds[best_idx], costs[best_idx]


# ─── 10-Fold CV with Cost-Based Model Selection ─────────────────────────────
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
            random_state=42, eval_metric="logloss", n_jobs=-1
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

# ─── Optuna Hyperparameter Search for XGBoost ───────────────────────────────
optuna_params = None
if HAS_OPTUNA:
    print(f"\n{'='*70}")
    print("OPTUNA HYPERPARAMETER SEARCH FOR XGBOOST (50 trials)")
    print(f"{'='*70}")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                **params, random_state=42, eval_metric="logloss", n_jobs=-1
            ))
        ])
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        costs = []
        for train_idx, val_idx in inner_cv.split(X_train, y_train):
            pipe.fit(X_train[train_idx], y_train[train_idx])
            proba = pipe.predict_proba(X_train[val_idx])[:, 1]
            _, cost = find_optimal_threshold(y_train[val_idx], proba)
            costs.append(cost)
        return np.mean(costs)

    study = optuna.create_study(direction="minimize", study_name="xgb_cost_search")
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    optuna_params = study.best_params
    print(f"\nBest Optuna params: {json.dumps(optuna_params, indent=2)}")
    print(f"Best CV Expected Cost: {study.best_value:.4f}")

    # Add Tuned XGBoost to models dictionary
    models["XGBoost (Tuned)"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            **optuna_params, random_state=42, eval_metric="logloss", n_jobs=-1
        ))
    ])

# ─── 10-Fold CV with Cost-Based Model Selection ─────────────────────────────
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = {}
model_costs = {}

print(f"\n{'='*70}")
print("10-FOLD STRATIFIED CV + COST-BASED MODEL SELECTION")
print(f"Cost function: E[cost] = FN_rate × {COST_FN} + FP_rate × {COST_FP}")
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

        # Compute cost-based ranking: fit on train, predict on val
        pipeline_copy = Pipeline(pipeline.steps)
        pipeline_copy.fit(X_train, y_train)
        val_proba = pipeline_copy.predict_proba(X_val)[:, 1]
        opt_thresh, opt_cost = find_optimal_threshold(y_val, val_proba)
        model_costs[name] = {"threshold": float(opt_thresh), "expected_cost": float(opt_cost)}

        print(f"  F1={result['f1']['mean']:.4f}  Recall={result['recall']['mean']:.4f}  "
              f"ROC-AUC={result['roc_auc']['mean']:.4f}  "
              f"E[cost]={opt_cost:.4f} (thresh={opt_thresh:.3f})")
    except Exception as e:
        print(f"  ERROR: {e}")
        cv_results[name] = {"error": str(e)}

cv_path = os.path.join(RESULTS_DIR, "cv_results.json")
with open(cv_path, "w") as f:
    json.dump(cv_results, f, indent=2)
print(f"\nSaved CV results to {cv_path}")

# ─── Rank by Expected Cost (Tie-breaker: ROC-AUC, then Recall) ──────────────
print(f"\n{'='*70}")
print("MODEL RANKING (by Expected Cost -> ROC-AUC -> Recall)")
print(f"{'='*70}")

valid_models = {k: v for k, v in cv_results.items() if "error" not in v}
ranking = sorted(valid_models.items(), key=lambda x: (
    model_costs[x[0]]["expected_cost"],
    -x[1]["roc_auc"]["mean"],
    -x[1]["recall"]["mean"]
))

for rank, (name, scores) in enumerate(ranking, 1):
    marker = " <-- SELECTED" if rank == 1 else ""
    cost_info = model_costs[name]
    print(f"  {rank}. {name}: E[cost]={cost_info['expected_cost']:.4f}  "
          f"Recall={scores['recall']['mean']:.4f}  "
          f"ROC-AUC={scores['roc_auc']['mean']:.4f}{marker}")

best_model_name = ranking[0][0]
print(f"\nBest model: {best_model_name}")

# ─── Statistical Significance Test (Paired Bootstrap) ───────────────────────
if len(ranking) >= 2:
    print(f"\nRunning paired bootstrap test: {ranking[0][0]} vs {ranking[1][0]}...")
    name_a, name_b = ranking[0][0], ranking[1][0]
    pipe_a = Pipeline(models[name_a].steps)
    pipe_b = Pipeline(models[name_b].steps)
    pipe_a.fit(X_train, y_train)
    pipe_b.fit(X_train, y_train)
    proba_a = pipe_a.predict_proba(X_val)[:, 1]
    proba_b = pipe_b.predict_proba(X_val)[:, 1]

    n_boot = 1000
    rng = np.random.default_rng(42)
    cost_diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_val), size=len(y_val))
        _, cost_a = find_optimal_threshold(y_val[idx], proba_a[idx])
        _, cost_b = find_optimal_threshold(y_val[idx], proba_b[idx])
        cost_diffs.append(cost_a - cost_b)

    cost_diffs = np.array(cost_diffs)
    p_value = np.mean(cost_diffs >= 0)  # fraction where A is worse or equal
    ci_low, ci_high = np.percentile(cost_diffs, [2.5, 97.5])

    significance_result = {
        "model_a": name_a, "model_b": name_b,
        "mean_cost_diff": float(np.mean(cost_diffs)),
        "ci_95": [float(ci_low), float(ci_high)],
        "p_value_a_worse": float(p_value),
        "significant_at_005": bool(p_value < 0.05)
    }
    print(f"  Mean cost diff (A-B): {np.mean(cost_diffs):.4f} [{ci_low:.4f}, {ci_high:.4f}]")
    if not significance_result["significant_at_005"]:
        print(f"  NOTE: Difference is NOT statistically significant (p={p_value:.3f}).")
        print(f"  The task is solvable across tree/linear families with comparable risk cost.")
    else:
        print(f"  Difference IS significant (p={p_value:.3f}). {name_a} wins.")
else:
    significance_result = None

best_pipeline = Pipeline(models[best_model_name].steps)

# ─── Refit on Full Training Set ─────────────────────────────────────────────
print(f"\nRetraining {best_model_name} on full training set...")
best_pipeline.fit(X_train, y_train)

model_path = os.path.join(MODELS_DIR, "best_model.joblib")
joblib.dump(best_pipeline, model_path)
print(f"Saved trained model to {model_path}")

# ─── MLflow Logging ─────────────────────────────────────────────────────────
if HAS_MLFLOW:
    mlflow.set_experiment("fraud_detection")
    with mlflow.start_run(run_name=f"train_{best_model_name}"):
        mlflow.log_param("model_name", best_model_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("cost_fn", COST_FN)
        mlflow.log_param("cost_fp", COST_FP)
        if optuna_params:
            mlflow.log_params({f"opt_{k}": v for k, v in optuna_params.items()})
        best_cv = cv_results[best_model_name]
        mlflow.log_metric("cv_recall_mean", best_cv["recall"]["mean"])
        mlflow.log_metric("cv_f1_mean", best_cv["f1"]["mean"])
        mlflow.log_metric("cv_roc_auc_mean", best_cv["roc_auc"]["mean"])
        mlflow.log_metric("val_expected_cost", model_costs[best_model_name]["expected_cost"])
        mlflow.log_artifact(model_path)
    print("Logged training run to MLflow.")

# ─── Save Metadata ──────────────────────────────────────────────────────────
meta = {
    "model_name": best_model_name,
    "feature_columns": FEATURE_COLS,
    "target_column": TARGET_COL,
    "train_size": int(len(X_train)),
    "val_size": int(len(X_val)),
    "test_size": int(len(X_test)),
    "split_ratio": "70/15/15",
    "cost_fn": COST_FN,
    "cost_fp": COST_FP,
    "cv_recall_mean": float(cv_results[best_model_name]["recall"]["mean"]),
    "cv_f1_mean": float(cv_results[best_model_name]["f1"]["mean"]),
    "cv_roc_auc_mean": float(cv_results[best_model_name]["roc_auc"]["mean"]),
    "val_expected_cost": model_costs[best_model_name]["expected_cost"],
    "val_optimal_threshold": model_costs[best_model_name]["threshold"],
    "optuna_best_params": optuna_params,
    "significance_test": significance_result,
}
with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

# ─── Model Comparison Chart ─────────────────────────────────────────────────
print("\nGenerating model comparison chart...")
metrics_to_plot = ["f1", "recall", "roc_auc", "pr_auc"]
model_names = [name for name, _ in ranking]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Standard metric comparison
ax = axes[0]
x = np.arange(len(model_names))
width = 0.18
colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

for i, metric in enumerate(metrics_to_plot):
    values = [valid_models[name][metric]["mean"] for name in model_names]
    stds = [valid_models[name][metric]["std"] for name in model_names]
    ax.bar(x + i * width, values, width, label=metric.upper().replace("_", "-"),
           color=colors[i], edgecolor="black", linewidth=0.3, yerr=stds, capsize=3)

ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("10-Fold CV: Metric Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names, rotation=25, ha="right")
ax.legend(loc="lower right")
ax.set_ylim(0.75, 1.0)
ax.grid(axis="y", alpha=0.3)

# Right: Expected cost comparison
ax2 = axes[1]
cost_values = [model_costs[name]["expected_cost"] for name in model_names]
bar_colors = ["#2ecc71" if name == best_model_name else "#3498db" for name in model_names]
ax2.bar(model_names, cost_values, color=bar_colors, edgecolor="black", linewidth=0.3)
ax2.set_xlabel("Model", fontsize=12)
ax2.set_ylabel("Expected Cost (lower = better)", fontsize=12)
ax2.set_title(f"Cost-Based Selection (C_FN={COST_FN}, C_FP={COST_FP})", fontsize=14, fontweight="bold")
ax2.set_xticklabels(model_names, rotation=25, ha="right")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(VISUALS_EVAL_DIR, "model_comparison.png"), dpi=150)
plt.close()
print("Saved visuals/evaluation/model_comparison.png")

print(f"\n{'='*60}")
print("MODEL TRAINING COMPLETE")
print(f"{'='*60}")
