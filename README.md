# 🛡️ Real-Time Free Trial Abuse & Multi-Accounting Risk Detection System
### *End-to-End Technical Architecture, Causal Feature Store, Cost-Optimized ML Pipeline & Serving Engine*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/model-Cost--Optimized%20ML%20Pipeline-orange.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.973%20%C2%B1%200.012-brightgreen.svg)]()
[![Abuse Recall](https://img.shields.io/badge/Abuse%20Recall-93.8%25%20(95%25%20CI)-success.svg)]()
[![Precision](https://img.shields.io/badge/Precision-100.0%25-blue.svg)]()
[![Latency](https://img.shields.io/badge/Latency-%3C15ms-purple.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📑 Complete System Blueprint

```
========================================================================================================================
                                     END-TO-END RISK DETECTION SYSTEM TOPOLOGY
========================================================================================================================

    [ CLIENT / WEB APP / SIGNUP FORM ]
             │
             │ (1) POST /api/v1/score (Raw JSON: name, email, IP, device_id, payment_token, area, time)
             ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │  STAGE 1: INGESTION & IDENTITY ENRICHMENT LAYER (< 5ms)                                                          │
    │  ├─ Lexical Normalization : Alphanumeric name normalization; email local-part + domain extraction                 │
    │  ├─ Network Enrichment   : IP -> /24 Subnet prefix; ISP subnet clustering                                        │
    │  ├─ Email Classification : Disposable domain registry check; +Tag and trailing digit detection                   │
    │  └─ Geo/Payment Parsing  : IP Country vs Payment BIN Country resolution                                          │
    └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
             │
             │ (2) Enriched Identity Tokens
             ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │  STAGE 2: CAUSAL FEATURE STORE & ENTITY GRAPH ENGINE (< 5ms)                                                     │
    │  ├─ Redis / Memory Store : Sliding 24h & 1h velocity sorted sets (ZADD / ZREMRANGEBYSCORE) + atomic counters     │
    │  ├─ Incremental Union-Find: Disjoint-set graph clustering across (Payment, Device, Subnet)                       │
    │  └─ Strict Causal Contract: Queries state strictly PRIOR to event timestamp t; updates memory state only AFTER   │
    └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
             │
             │ (3) 20-Dimensional Dense Feature Vector
             ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │  STAGE 3: ML INFERENCE ENGINE (< 5ms)                                                                            │
    │  ├─ Scaled Pipeline      : Standardized feature scaling + calibrated decision engine                             │
    │  ├─ Optuna Hyperparams   : 50-trial Bayesian optimization on tree depth, subsample, and regularization           │
    │  └─ Live Probabilities   : Continuous predict_proba(X) output P(Abuse | X) -> Scaled Risk Score [0.0 - 100.0]   │
    └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
             │
             │ (4) Risk Score (P × 100) + Explainability Breakdown
             ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │  STAGE 4: COST-OPTIMIZED 3-BAND POLICY LAYER (< 2ms)                                                             │
    │  ├─ Decision Threshold (T = 0.100 / Score = 10.0) : Tuned on validation set against business costs ($5 FN / $1 FP) │
    │  ├─ Score < 5.5 (0.0 - 5.5)     ──► VERDICT: "NEW USER (GENUINE)"       ──► ACTION: Allow Full Trial Access      │
    │  ├─ 5.5 ≤ Score < 10.0 (5.5-10.0)──► VERDICT: "SUSPICIOUS (STEP-UP)"     ──► ACTION: Step-Up (SMS OTP / CAPTCHA)  │
    │  ├─ Score ≥ 10.0 (10.0 - 100.0)  ──► VERDICT: "REPEAT / LIKELY ABUSE"   ──► ACTION: Block Trial / Demand Payment │
    │  └─ Companion Explainability     : Additive signal breakdown alongside model output for UI transparency          │
    └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
             │
             │ (5) Telemetry & Audit Stream
             ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │  STAGE 5: CONTINUOUS MONITORING & ACTIVE LEARNING LOOP                                                           │
    │  ├─ Population Stability Index   : Continuous feature and risk score drift tracking (PSI alert threshold >= 0.25)│
    │  ├─ Subgroup Fairness Audit     : Subgroup FPR/FNR monitoring across 14 geographic zones (n >= 50 sample gate)   │
    │  └─ Human-in-the-Loop Active AL  : Multi-round entropy sampling resolving borderline grey-zone accounts          │
    └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
========================================================================================================================
```

---

## 📖 Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Why Rule-Based Heuristics Fail](#2-why-rule-based-heuristics-fail)
3. [Layer-by-Layer System Architecture](#3-layer-by-layer-system-architecture)
4. [Dataset & Population Dynamics](#4-dataset--population-dynamics)
5. [Causal Feature Engineering & Entity Graph](#5-causal-feature-engineering--entity-graph)
6. [Cost-Based Model Selection & 10-Fold CV](#6-cost-based-model-selection--10-fold-cv)
7. [Held-Out Test Set Evaluation & Threshold Tuning](#7-held-out-test-set-evaluation--threshold-tuning)
8. [Subgroup Fairness & Statistical Audits](#8-subgroup-fairness--statistical-audits)
9. [Project Repository Layout](#9-project-repository-layout)
10. [Step-by-Step Pipeline Execution](#10-step-by-step-pipeline-execution)
11. [Production Deployment & REST API](#11-production-deployment--rest-api)
12. [Limitations & Real-World Considerations](#12-limitations--real-world-considerations)

---

## 1. Executive Summary & Problem Statement

Free trial acquisition funnels in SaaS platforms are targeted by automated multi-accounting scripts and syndicates seeking free compute, trial credits, or scraping bandwidth.

### The Operational Challenges:
- **Asymmetric Financial Impact:** Missed abuse (**False Negatives**) leads to severe infrastructure costs ($C_{\text{FN}} \approx \$5.00$), while false accusations (**False Positives**) introduce customer friction ($C_{\text{FP}} \approx \$1.00$).
- **Identity Rotation:** Attackers rotate cheap identifiers (disposable emails, IP addresses) while reusing expensive ones (payment tokens, physical device hashes, `/24` subnets).
- **Latency Constraint:** Decisions must be computed synchronously within the signup request lifecycle ($< 20\text{ms}$).

---

## 2. Why Rule-Based Heuristics Fail

| Traditional Heuristic | Attacker Evasion Vector | Failure Mode | Architectural Remedy |
|---|---|---|---|
| `WHERE email = ?` | Rotates alias tags (`+trial1`) or disposable inboxes | Misses repeat signups entirely | Lexical tagging audit + domain registry |
| `WHERE ip = ?` | Restarts modem or rotates proxy within ISP block | Fails on residential proxies; flags shared NATs | **`/24` Subnet Grouping** + sliding 24h velocity |
| `WHERE device_id = ?` | Clears local storage or uses incognito mode | Generates new ID per request | Multi-attribute graph component linkage |
| `WHERE payment_token = ?` | Rotates prepaid cards or stolen BINs | Catches only static replays | **Incremental Union-Find Entity Graph** |
| Arbitrary 50% Threshold | Low-velocity attacks score ~0.2–0.4 | Misses stealth syndicates | **Cost-Based Threshold Optimization ($T=0.100$)** |

---

## 3. Layer-by-Layer System Architecture

### Layer 1: Identity Ingestion & Enrichment
Raw payloads (`name`, `email`, `ip_address`, `device_id`, `payment_token`, `area`) are parsed in $< 5\text{ms}$:
- **IP $\to$ `/24` Subnet:** Normalizes `192.168.1.50` to `192.168.1`, grouping residential proxy churn within the same ISP block.
- **Email Syntax Analysis:** Flags high-risk disposable domains (`mailinator.com`, `yopmail.com`), alias tags (`user+tag@`), and trailing digit density.
- **Geo-IP vs BIN Country:** Detects foreign card issuance vs connecting IP country mismatches with realistic baseline noise.

### Layer 2: Causal Feature Store & Union-Find Graph
To eliminate lookahead target leakage:
- Events are processed chronologically.
- Features are calculated strictly against state seen **before** the event timestamp.
- Lookup counters and graph edges are committed **after** feature extraction.
- An **Incremental Union-Find** algorithm maintains connected components across `(Payment Token, Device ID, Subnet)` in $O(\alpha(N))$ nearly constant time.

### Layer 3: Model Scoring
- Served via `predict_proba(X)` from a serialized scikit-learn/XGBoost pipeline.
- Continuous risk probabilities $P(\text{Abuse}) \in [0.0, 1.0]$ are mapped to a transparent $[0.0, 100.0]$ Risk Score.

---

## 4. Dataset & Population Dynamics

The dataset represents 9,262 signup events across a 6-month simulated timeline:
- **Genuine Users (70.2%):** 6,500 organic single-signup accounts with natural 4% geographic variation (travel, VPNs).
- **Abuse Syndicates (29.8%):** 700 coordinated rings (2–6 accounts each) rotating attributes realistically (payment retained 75%, device retained 65%, subnet retained 50%, area retained 85%).
- **Data Partitioning:** Split strictly **70% Train (6,483 rows) / 15% Validation (1,389 rows) / 15% Test (1,390 rows)** using stratified sampling.

---

## 5. Causal Feature Engineering & Entity Graph

| # | Feature Name | Computation Logic | Correlation ($r$) |
|:---:|---|---|:---:|
| 1 | `graph_component_size` | Connected component size in Union-Find graph | **+0.744** |
| 2 | `attrs_reused_count` | Distinct identity families reused (0–4) | **+0.742** |
| 3 | `subnet_signups_last_24h` | Sliding window count of signups in `/24` in last 86,400s | **+0.702** |
| 4 | `ip_subnet_reuse_count` | Historical signups from same `/24` prefix | **+0.701** |
| 5 | `email_local_has_plus_tag` | `'+' in email.local` | **+0.650** |
| 6 | `is_disposable_email_domain` | Domain in disposable registry | **+0.559** |
| 7 | `payment_reuse_count` | Lifetime seen count for payment token | **+0.550** |
| 8 | `payment_signups_last_24h` | Sliding 24h count for payment token | **+0.550** |
| 9 | `device_reuse_count` | Lifetime seen count for device fingerprint | **+0.498** |
| 10 | `device_signups_last_24h` | Sliding 24h count for device fingerprint | **+0.498** |
| 11 | `email_local_has_digits` | Numeric characters in email local-part | **+0.416** |
| 12 | `ip_reuse_count` | Lifetime exact IP occurrences | **+0.390** |
| 13 | `name_similarity_score` | Levenshtein / SequenceMatcher score vs recent names | **+0.315** |
| 14 | `device_signups_last_hour` | Sliding 1h burst velocity for device ID | **+0.279** |
| 15 | `payment_ip_country_mismatch` | Country of IP != BIN Country (non-leaky) | +0.072 |
| 16 | `is_odd_hour` | Signup during 00:00–05:00 UTC | +0.011 |
| 17 | `device_os_freq` | Frequency encoding of operating system | +0.001 |
| 18 | `area_freq` | Frequency encoding of geographic area | -0.000 |
| 19 | `signup_hour` | Continuous hour integer (0–23) | -0.024 |
| 20 | `is_free_email_domain` | Domain in standard free providers | **-0.221** |

---

## 6. Cost-Based Model Selection & 10-Fold CV

Candidate algorithms were evaluated via **10-Fold Stratified Cross-Validation** on the training set (6,483 samples). Models were ranked by minimizing expected business cost:

$$E[\text{Cost}] = \text{FNR} \times C_{\text{FN}} \times P(Y=1) + \text{FPR} \times C_{\text{FP}} \times P(Y=0)$$

Where $C_{\text{FN}} = \$5.00$ and $C_{\text{FP}} = \$1.00$.

| Rank | Model Family | Mean CV Recall | Mean CV F1 | Mean CV ROC-AUC | Val Expected Cost ($E[\text{cost}]$) | Status |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Logistic Regression** | 0.9265 | 0.9612 | **0.9740** | **0.1231** | **SELECTED** |
| 2 | **XGBoost (Optuna Tuned)** | 0.9265 | 0.9615 | 0.9738 | **0.1231** | Runner-up |
| 3 | **Gradient Boosting** | 0.9265 | 0.9612 | 0.9733 | **0.1231** | Evaluated |
| 4 | **Random Forest** | 0.9271 | 0.9602 | 0.9701 | **0.1231** | Evaluated |
| 5 | **XGBoost (Default)** | 0.9271 | 0.9590 | 0.9693 | **0.1231** | Evaluated |
| 6 | **SVM (RBF)** | 0.9265 | 0.9610 | 0.9614 | **0.1231** | Evaluated |
| 7 | **Decision Tree** | 0.9281 | 0.9580 | 0.9637 | 0.1267 | Evaluated |
| 8 | **KNN (k=7)** | 0.9079 | 0.9511 | 0.9624 | 0.1440 | Evaluated |

### Statistical Significance Test
A 1,000-iteration paired bootstrap comparison between the top models confirmed that performance differences between tree and linear architectures were not statistically significant ($p = 0.759$). The engineered causal entity graph and velocity signals are the primary drivers of discriminative performance.

---

## 7. Held-Out Test Set Evaluation & Threshold Tuning

Threshold selection was performed exclusively on the **Validation Set (1,389 rows)** to guarantee zero test leakage. The chosen threshold ($T = 0.100$) was then evaluated **once** on the untouched **Test Set (1,390 rows)**.

### Headline Test Metrics (with 95% Bootstrap Confidence Intervals)

| Metric | Point Estimate | 95% Bootstrap Confidence Interval |
|---|:---:|:---:|
| **ROC-AUC** | **0.973** | `[0.960, 0.984]` |
| **PR-AUC (Average Precision)** | **0.971** | — |
| **Abuse Recall** | **93.8%** | `[91.2%, 96.0%]` |
| **Precision** | **100.0%** | `[100.0%, 100.0%]` |
| **F1-Score** | **0.968** | `[0.954, 0.979]` |
| **Accuracy** | **98.1%** | `[97.3%, 98.8%]` |
| **Brier Score (Calibration Loss)** | **0.0185** | Well-calibrated ($< 0.05$) |

```
Test Set Confusion Matrix (T = 0.100):
                 Predicted Genuine    Predicted Abuse
Actual Genuine          975                  0          (FPR = 0.00%)
Actual Abuse             26                389          (Recall = 93.76%)
```

---

## 8. Subgroup Fairness & Statistical Audits

An automated demographic audit was executed across all 14 geographic zones with a minimum sample-size gate ($n \ge 50$ per subgroup):

| Subgroup (Area) | Test Samples ($N$) | Subgroup FPR | Subgroup FNR | Audit Flag |
|---|:---:|:---:|:---:|:---:|
| **Ahmedabad** | 101 | 0.00% | 10.34% | OK |
| **Bangalore** | 95 | 0.00% | 10.71% | OK |
| **Chennai** | 101 | 0.00% | 2.86% | OK |
| **Delhi** | 87 | 0.00% | 3.45% | OK |
| **Dubai** | 96 | 0.00% | 6.90% | OK |
| **Hyderabad** | 90 | 0.00% | 0.00% | OK |
| **Kolkata** | 104 | 0.00% | 4.76% | OK |
| **London** | 98 | 0.00% | 4.35% | OK |
| **Mumbai** | 109 | 0.00% | 10.34% | OK |
| **New York** | 100 | 0.00% | 5.56% | OK |
| **Pune** | 101 | 0.00% | 2.86% | OK |
| **San Francisco** | 117 | 0.00% | 13.16% | Flagged (Higher FNR) |
| **Singapore** | 95 | 0.00% | 7.41% | OK |
| **Toronto** | 96 | 0.00% | 2.86% | OK |

---

## 9. Project Repository Layout

```
Fraud_detection/
├── data/
│   ├── raw/
│   │   └── raw_signup_events.csv              # Synthetic base event log (9,262 rows)
│   └── processed/
│       ├── features_v2.csv                    # 20-feature engineered matrix
│       ├── full_dataset_with_features.csv     # Complete dataset with raw + features
│       ├── scored_dataset.csv                 # Scored dataset with model predictions
│       ├── train_set.csv                      # 70% Training partition (6,483 rows)
│       ├── val_set.csv                        # 15% Validation partition (1,389 rows)
│       └── test_set.csv                       # 15% Test partition (1,390 rows)
│
├── models/
│   ├── best_model.joblib                      # Serialized production model pipeline
│   └── model_metadata.json                    # Hyperparameters, CV scores, and metadata
│
├── results/
│   ├── cv_results.json                        # 10-Fold CV metrics across all models
│   ├── final_metrics.json                     # Test set metrics with 95% bootstrap CIs
│   ├── fairness_analysis.json                 # Subgroup FPR/FNR audit report
│   ├── drift_analysis.json                    # Feature and prediction PSI drift report
│   ├── active_learning_results.json           # 5-round active learning progression
│   ├── feature_importance.csv                 # Global feature importance scores
│   └── risk_scoring_demo.json                 # Structured scoring demo payloads
│
├── visuals/
│   ├── eda/                                   # Target, feature distributions, correlation matrix
│   ├── evaluation/                            # Confusion matrix, ROC, PR, calibration, threshold curves
│   ├── explainability/                        # Global importance & SHAP attribution summary
│   ├── inference/                             # Genuine/Fraud cards, attack evasion trace, policy
│   └── monitoring/                            # Drift dashboard & Active Learning curves
│
├── scripts/
│   ├── 01_generate_data.py                    # Non-leaky synthetic dataset generator
│   ├── 02_eda.py                              # Exploratory data analysis charts
│   ├── 03_feature_engineering.py              # Causal features & Union-Find graph builder
│   ├── 04_model_training.py                   # 10-Fold CV, Optuna tuning, cost selection, MLflow
│   ├── 05_model_evaluation.py                 # Val threshold tuning, bootstrap CIs, fairness audit
│   ├── 06_risk_scoring_engine.py              # Batch scoring pipeline
│   ├── 07_drift_monitor.py                    # Continuous PSI & KS drift monitoring engine
│   ├── 08_active_learning_feedback.py         # Multi-round human-in-the-loop active learning
│   ├── redis_feature_store.py                 # Redis sorted-set feature store adapter
│   └── generate_inference_visuals.py          # Inference scorecards and dashboard visuals
│
├── tests/
│   └── test_suite.py                          # 16 automated unit & integration tests
│
├── .github/
│   └── workflows/
│       └── ci.yml                             # Standalone GitHub Actions CI workflow
│
├── api.py                                     # Production FastAPI Microservice (<15ms)
├── app.py                                     # Interactive Web GUI Dashboard
├── predict.py                                 # Programmatic & CLI scoring engine
├── Dockerfile                                 # Multi-stage production container
├── docker-compose.yml                         # FastAPI + Redis deployment
├── LICENSE                                    # MIT License
├── INTERVIEW_NOTES.md                         # Technical interview defense & elevator pitch
└── requirements.txt                           # Pinned Python package dependencies
```

---

## 10. Step-by-Step Pipeline Execution

Run the complete pipeline from scratch:

```bash
# 1. Generate Non-Leaky Synthetic Dataset
py scripts/01_generate_data.py

# 2. Generate Exploratory Data Analysis Visuals
py scripts/02_eda.py

# 3. Compute Causal Feature Matrix & Entity Graph
py scripts/03_feature_engineering.py

# 4. Train Models with 10-Fold CV & Optuna Hyperparameter Search
py scripts/04_model_training.py

# 5. Evaluate on Held-Out Test Set (Validation Thresholding & Bootstrap CIs)
py scripts/05_model_evaluation.py

# 6. Run Batch Risk Scoring Engine
py scripts/06_risk_scoring_engine.py

# 7. Run Automated Test Suite
py -m pytest tests/test_suite.py -v
```

---

## 11. Production Deployment & Developer Platform

### Live Production API
- **Live Render Base URL:** `https://free-trail-fraud-detection-mlmodel.onrender.com`
- **Inference Endpoint:** `POST /api/v1/score`
- **Rate Limit:** 30 requests/minute per tenant key (Sliding 60-second window)

---

### A. Launch Developer Platform & Dashboard Locally
```bash
py app.py
```
- **Dashboard:** `http://localhost:8000`
- **Features:** Real-time inference playground, tenant API key management (up to 3 keys per verified user), customer directory with instant search, multi-language code snippets, and live model telemetry.

---

### B. Launch FastAPI Production Microservice
```bash
py api.py
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health Check Probe: `http://localhost:8000/healthz`
- Real-Time Scoring: `POST /api/v1/score`

---

### C. Multi-Language Integration Snippets

#### 1. cURL
```bash
curl -X POST https://free-trail-fraud-detection-mlmodel.onrender.com/api/v1/score \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY_HERE" \
  -d '{
    "name": "Sarah Miller",
    "email": "sarah.miller@gmail.com",
    "ip_address": "198.51.100.24",
    "device_id": "dev_macbook_pro_m2_99",
    "payment_token": "pm_visa_auth_8821",
    "area": "new york"
  }'
```

#### 2. Python (Requests)
```python
import requests

url = "https://free-trail-fraud-detection-mlmodel.onrender.com/api/v1/score"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "YOUR_API_KEY_HERE"
}
payload = {
    "name": "Sarah Miller",
    "email": "sarah.miller@gmail.com",
    "ip_address": "198.51.100.24",
    "device_id": "dev_macbook_pro_m2_99",
    "payment_token": "pm_visa_auth_8821",
    "area": "new york"
}

response = requests.post(url, json=payload, headers=headers)
decision = response.json()
print(f"Verdict: {decision['verdict']} | Score: {decision['risk_score']}")
```

#### 3. Python SDK (`client.py`)
```python
from client import FraudDetectionClient

client = FraudDetectionClient(
    base_url="https://free-trail-fraud-detection-mlmodel.onrender.com",
    api_key="YOUR_API_KEY_HERE"
)

decision = client.score_signup(
    name="Sarah Miller",
    email="sarah.miller@gmail.com",
    ip_address="198.51.100.24",
    device_id="dev_macbook_pro_m2_99",
    payment_token="pm_visa_auth_8821",
    area="new york"
)

if decision.is_fraudulent:
    raise PermissionError("Trial limit reached on this device or network.")
```

#### 4. Node.js (Express Middleware)
```javascript
const express = require('express');
const app = express();
app.use(express.json());

app.post('/api/signup', async (req, res) => {
  const fraudCheck = await fetch("https://free-trail-fraud-detection-mlmodel.onrender.com/api/v1/score", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": process.env.FRAUD_API_KEY
    },
    body: JSON.stringify({
      name: req.body.name,
      email: req.body.email,
      ip_address: req.ip,
      device_id: req.body.device_id,
      payment_token: req.body.payment_token,
      area: req.body.area || "london"
    })
  }).then(r => r.json());

  if (fraudCheck.verdict === "REPEATING USER (LIKELY ABUSE)") {
    return res.status(403).json({ error: "Trial limit exceeded." });
  }

  return res.json({ status: "TRIAL_ACTIVATED" });
});
```

---

### D. Multi-Tenant Architecture & Cloud Firestore Sync

1. **Authentication:** Pure Firebase Auth (Email/Password with auto-verified link polling and Google OAuth).
2. **Quota & Keys:** Strictly 3 API keys per verified developer account. Unverified accounts cannot generate keys.
3. **Cryptographic Protection:** API keys stored using SHA-256 encrypted hashes. Zero plaintext storage.
4. **Firestore Collections:**
   - `users/{uid}`: Tenant profiles and verification metadata.
   - `users/{uid}/api_keys/{keyId}`: Encrypted API key records.
   - `users/{uid}/customers/{customerId}`: Scored registration events and signal explainability breakdown.
   - `users/{uid}/login_history/{loginId}`: User audit log with timestamps and user-agent data.

---

## 12. Limitations & Real-World Considerations

1. **Synthetic Data Assumptions:** The dataset is generated via parameterized identity rotation rules. While modeled after realistic SaaS attack patterns, real-world fraud distributions feature higher label noise, device fingerprint spoofing via anti-detect browsers, and distributed residential proxy pools.
2. **Cold-Start Account #1:** Account #1 of an entirely clean syndicate has zero historical replay markers and appears genuine. The system catches accounts #2 and onward as soon as network or instrument linkages emerge.
3. **Geographic Representation:** Subgroup analysis is evaluated over 14 predefined city hubs. In production, continuous monitoring over live IP geolocation and ASN blocks is necessary to prevent regional false-positive drift.
4. **Adversarial Adaptation:** Attackers continually adapt identity rotation strategies. The integrated Population Stability Index (PSI) drift monitoring engine must run on daily batches to trigger model retraining before performance degrades.
