"""
HTML-to-PDF Report Generator using Headless Chrome/Edge
======================================================
Compiles complete architecture, how-to-run quickstart (on page 1),
embedded evaluation & EDA visuals, and interview guide into a PDF.
"""

import os
import subprocess
import base64

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_OUTPUT_PATH = os.path.join(BASE_DIR, "Fraud_Detection_System_Report.pdf")
HTML_TEMP_PATH = os.path.join(BASE_DIR, "temp_report.html")

def get_base64_img(rel_path):
    full_path = os.path.join(BASE_DIR, rel_path)
    if os.path.exists(full_path):
        with open(full_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{data}"
    return ""

img_target_dist = get_base64_img("visuals/eda/target_distribution.png")
img_missing_vals = get_base64_img("visuals/eda/missing_values.png")
img_feat_dist = get_base64_img("visuals/eda/feature_distributions.png")
img_corr_mat = get_base64_img("visuals/eda/correlation_matrix.png")
img_model_comp = get_base64_img("visuals/evaluation/model_comparison.png")
img_cm = get_base64_img("visuals/evaluation/confusion_matrix.png")
img_roc = get_base64_img("visuals/evaluation/roc_curve.png")
img_pr = get_base64_img("visuals/evaluation/precision_recall_curve.png")
img_calib = get_base64_img("visuals/evaluation/calibration_curve.png")
img_thresh = get_base64_img("visuals/evaluation/threshold_analysis.png")
img_feat_imp = get_base64_img("visuals/explainability/feature_importance.png")
img_shap = get_base64_img("visuals/explainability/shap_summary.png")
img_gen_card = get_base64_img("visuals/inference/genuine_user_result.png")
img_fraud_card = get_base64_img("visuals/inference/fraud_syndicate_result.png")
img_evasion_trace = get_base64_img("visuals/inference/attack_evasion_graph_trace.png")
img_dash_summary = get_base64_img("visuals/inference/live_scoring_dashboard_summary.png")

template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Free Trial Abuse & Multi-Accounting Risk Detection System - Technical Report</title>
<style>
  @page {
    size: A4 portrait;
    margin: 12mm 12mm 12mm 12mm;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
    color: #1e293b;
    background: #ffffff;
    font-size: 9.5pt;
    line-height: 1.45;
  }
  .page {
    page-break-after: always;
    padding-bottom: 5px;
  }
  .page:last-child {
    page-break-after: avoid;
  }
  
  /* Headers */
  .doc-header {
    border-bottom: 2.5px solid #2563eb;
    padding-bottom: 8px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }
  .doc-title {
    font-size: 15pt;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.5px;
  }
  .doc-subtitle {
    font-size: 9pt;
    color: #2563eb;
    font-weight: 600;
    margin-top: 2px;
  }
  .badge-row {
    display: flex;
    gap: 6px;
    margin-top: 4px;
  }
  .badge {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    color: #334155;
    font-size: 7.5pt;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
  }
  .badge-primary { background: #eff6ff; border-color: #bfdbfe; color: #1d4ed8; }
  .badge-success { background: #ecfdf5; border-color: #a7f3d0; color: #047857; }
  .badge-warning { background: #fffbeb; border-color: #fde68a; color: #b45309; }

  h2 {
    font-size: 11pt;
    font-weight: 700;
    color: #0f172a;
    border-left: 3.5px solid #2563eb;
    padding-left: 8px;
    margin: 10px 0 6px 0;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }
  h3 {
    font-size: 9.5pt;
    font-weight: 700;
    color: #1e293b;
    margin: 8px 0 3px 0;
  }
  p { margin-bottom: 6px; }

  /* Cards & Boxes */
  .box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 8px;
  }
  .box-primary {
    background: #f0f9ff;
    border-color: #bae6fd;
    border-left: 3.5px solid #0284c7;
  }
  .box-warning {
    background: #fffbeb;
    border-color: #fef3c7;
    border-left: 3.5px solid #f59e0b;
  }
  .box-success {
    background: #f0fdf4;
    border-color: #bbf7d0;
    border-left: 3.5px solid #10b981;
  }

  /* Code blocks */
  pre, code {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 8pt;
  }
  pre {
    background: #0f172a;
    color: #e2e8f0;
    padding: 6px 10px;
    border-radius: 5px;
    margin: 4px 0 6px 0;
    overflow-x: auto;
    line-height: 1.35;
  }

  /* Tables */
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 6px 0 8px 0;
    font-size: 8pt;
  }
  th, td {
    padding: 4px 6px;
    border: 1px solid #e2e8f0;
    text-align: left;
  }
  th {
    background: #f1f5f9;
    font-weight: 700;
    color: #334155;
    text-transform: uppercase;
    font-size: 7pt;
    letter-spacing: 0.3px;
  }
  tr:nth-child(even) { background: #f8fafc; }

  /* Metric Scorecard Grid */
  .grid-4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin: 6px 0;
  }
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin: 6px 0;
  }
  .grid-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    margin: 6px 0;
  }
  .metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 6px;
    text-align: center;
  }
  .metric-val {
    font-size: 13pt;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
  }
  .metric-lbl {
    font-size: 7pt;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    margin-top: 2px;
  }
  .val-green { color: #059669; }
  .val-blue { color: #2563eb; }
  .val-purple { color: #7c3aed; }
  .val-orange { color: #d97706; }

  /* Images */
  .img-wrap {
    text-align: center;
    margin: 4px 0;
  }
  .img-wrap img {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
    border: 1px solid #e2e8f0;
  }
  .img-caption {
    font-size: 7pt;
    font-weight: 600;
    color: #64748b;
    margin-top: 2px;
    text-align: center;
  }

  ul { padding-left: 14px; margin-bottom: 6px; }
  li { margin-bottom: 2px; }
</style>
</head>
<body>

<!-- ========================================== PAGE 1 ========================================== -->
<div class="page">
  <div class="doc-header">
    <div>
      <div class="doc-title">🛡️ FraudGuard AI: Free Trial Abuse Risk Engine</div>
      <div class="doc-subtitle">Production Architecture, ML Benchmark & Operations Guide</div>
    </div>
    <div class="badge-row">
      <span class="badge badge-primary">XGBoost v2</span>
      <span class="badge badge-success">ROC-AUC: 0.976</span>
      <span class="badge badge-warning">Recall: 95.1%</span>
    </div>
  </div>

  <div class="box box-primary">
    <h3 style="margin-top: 0; color: #0369a1;">📌 Executive Metric Scorecard (Held-Out Test Set N=1,854)</h3>
    <div class="grid-4">
      <div class="metric-card">
        <div class="metric-val val-green">95.1%</div>
        <div class="metric-lbl">Abuse Recall (527 / 554)</div>
      </div>
      <div class="metric-card">
        <div class="metric-val val-blue">0.976</div>
        <div class="metric-lbl">ROC-AUC Score</div>
      </div>
      <div class="metric-card">
        <div class="metric-val val-purple">0.974</div>
        <div class="metric-lbl">PR-AUC (Avg Precision)</div>
      </div>
      <div class="metric-card">
        <div class="metric-val val-orange">&lt; 20 ms</div>
        <div class="metric-lbl">P99 Inference Latency</div>
      </div>
    </div>
  </div>

  <h2>🚀 How to Run the ML Model & Inference Systems</h2>

  <h3>1. Launch Interactive Web GUI Dashboard (&lt;200ms real-time testing)</h3>
  <pre>py app.py</pre>
  <p style="font-size: 8.5pt; color: #475569;">
    Open <strong>http://localhost:8080</strong> in your browser. Features live scoring gauge, attack presets (<em>Subnet Ring</em>, <em>Disposable Email</em>, <em>BIN Geo Mismatch</em>), drag-and-drop CSV batch scorer, and interactive SHAP visuals.
  </p>

  <h3>2. Real-Time CLI Inference on an Unseen Signup Event</h3>
  <pre>py predict.py --name "Sanjay Nair" --email "sanjay.nair+trial1@mailinator.com" --ip "39.173.180.200" --device "f21faa72fe17c06d" --payment "pm_424776171fe7" --area "ahmedabad"</pre>
  
  <h3>3. Batch Predict on an Unseen CSV Dataset</h3>
  <pre>py predict.py --csv data/raw/raw_signup_events.csv --output data/processed/scored_predictions.csv</pre>

  <h3>4. Execute the Full 6-Step End-to-End Training Pipeline</h3>
  <pre>py scripts/01_generate_data.py        # Step 1: Generates 9,269 synthetic signup events
py scripts/02_eda.py                  # Step 2: Generates EDA & Missing Value charts
py scripts/03_feature_engineering.py  # Step 3: Computes Union-Find Entity Graph & 20 Causal Features
py scripts/04_model_training.py       # Step 4: Runs 10-Fold Stratified CV across 7 algorithms
py scripts/05_model_evaluation.py     # Step 5: Held-out test evaluation, threshold tuning & SHAP
py scripts/06_risk_scoring_engine.py  # Step 6: Scores full population with 3-band verdict</pre>

  <h2>📂 Clean Repository Structure</h2>
  <div class="grid-2">
    <div>
      <ul style="font-size: 8pt;">
        <li><code>data/raw/</code>: Raw signup event stream (9,269 events)</li>
        <li><code>data/processed/</code>: 20 engineered features, scored datasets</li>
        <li><code>models/</code>: Serialized <code>best_model.joblib</code> + metadata</li>
        <li><code>results/</code>: 10-fold CV JSONs, final metrics, demo outputs</li>
      </ul>
    </div>
    <div>
      <ul style="font-size: 8pt;">
        <li><code>visuals/eda/</code>: Class, missing, feature & correlation plots</li>
        <li><code>visuals/evaluation/</code>: Confusion matrix, ROC, PR, calibration</li>
        <li><code>visuals/explainability/</code>: Global feature importance & SHAP</li>
        <li><code>app.py</code> & <code>predict.py</code>: Standalone root inference tools</li>
      </ul>
    </div>
  </div>
</div>

<!-- ========================================== PAGE 2 ========================================== -->
<div class="page">
  <div class="doc-header">
    <div class="doc-title">System Topology & Layered Architecture</div>
    <div class="doc-subtitle">End-to-End Request Lifecycle & Invariants</div>
  </div>

  <h2>1. End-to-End System Topology</h2>
  <div class="box">
    <pre style="background: #0f172a; font-size: 7pt; color: #38bdf8;">
 [CLIENT SIGNUP] ──► (1) Ingestion & Identity Enrichment (IP Subnet /24, Disposable Regex, BIN Match)
                          │
                          ▼
                     (2) Causal Feature Store & Union-Find Graph Engine (Component Size, 24h Velocity)
                          │
                          ▼
                     (3) XGBoost ML Model Pipeline (StandardScaler + Tree Split Posteriors)
                          │
                          ▼
                     (4) 3-Band Policy Engine:
                         • Score &lt; 3.3  ──► VERDICT: NEW / GENUINE           (Allow Trial)
                         • 3.3 ≤ Score &lt; 6.0 ──► VERDICT: SUSPICIOUS (Review) (Step-Up Challenge)
                         • Score ≥ 6.0  ──► VERDICT: REPEAT ABUSE            (Block / Demand Payment)
                          │
                          ▼
                     (5) Async Telemetry & Continuous Feedback Loop (Chargebacks, PSI Drift Auditing)
    </pre>
  </div>

  <h2>2. The 3-Band Policy Decision Framework</h2>
  <p>Rather than relying on brittle binary cutoffs (P >= 0.5), the system implements a 3-Band Policy calibrated against asymmetric business risk:</p>
  <table>
    <thead>
      <tr>
        <th>Risk Tier</th>
        <th>Score Range</th>
        <th>Operational Action</th>
        <th>Friction Level</th>
        <th>Target User Persona</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Band 1: Genuine</strong></td>
        <td><code>0.0 - 3.3</code></td>
        <td><strong>ALLOW</strong> Instant Access</td>
        <td>Zero Friction</td>
        <td>Organic, single-account legitimate users</td>
      </tr>
      <tr>
        <td><strong>Band 2: Grey Zone</strong></td>
        <td><code>3.3 - 6.0</code></td>
        <td><strong>STEP-UP CHALLENGE</strong></td>
        <td>Low (SMS / CAPTCHA)</td>
        <td>Foreign IP travel, corporate NAT, borderline reuse</td>
      </tr>
      <tr>
        <td><strong>Band 3: Repeat Abuse</strong></td>
        <td><code>6.0 - 100.0</code></td>
        <td><strong>BLOCK / REQUIRE PAYMENT</strong></td>
        <td>Hard Wall</td>
        <td>Multi-accounting bot farms, recycled cards/subnets</td>
      </tr>
    </tbody>
  </table>

  <h2>3. Why Simple Rules Fail vs Graph Linkage</h2>
  <div class="box box-warning">
    <p><strong>The Adversarial Rotation Problem:</strong> Attackers rotate cheap identity artifacts (disposable emails, IP addresses) while keeping expensive artifacts (payment cards, residential subnets) fixed. Naive models over-index on disposable domains (90% weight), becoming blind when attackers switch to custom domains.</p>
    <p><strong>Our Solution:</strong> By constructing an <strong>Incremental Union-Find Connected Component Graph</strong> over <code>(Payment Token, Device ID, IP Subnet)</code>, the entity cluster size is preserved regardless of email/IP rotation (r = 0.746).</p>
  </div>
</div>

<!-- ========================================== PAGE 3 ========================================== -->
<div class="page">
  <div class="doc-header">
    <div class="doc-title">Exploratory Data Analysis & Population Dynamics</div>
    <div class="doc-subtitle">Synthetic Dataset Audit (N = 9,269 Events)</div>
  </div>

  <h2>1. Dataset Population Characteristics</h2>
  <p>The system was trained on a benchmark dataset adhering strictly to PRD Section 4:</p>
  <ul>
    <li><strong>Genuine Single-Signups (70.1% / 6,500 events):</strong> Independent credentials, standard email domains, organic daytime signups.</li>
    <li><strong>Abuse Syndicates (29.9% / 2,769 events across 700 rings):</strong> 2 to 6 linked accounts per syndicate (Mean: 3.96 accounts/ring) with realistic rotation probabilities (Card kept 75%, Device kept 65%, Subnet kept 100%, Email rotated 100%).</li>
  </ul>

  <div class="grid-2">
    <div class="img-wrap">
      <img src="__IMG_TARGET_DIST__" alt="Target Distribution">
      <div class="img-caption">Figure 1: Class Balance (70.1% Genuine / 29.9% Abuse)</div>
    </div>
    <div class="img-wrap">
      <img src="__IMG_MISSING_VALS__" alt="Missing Values Audit">
      <div class="img-caption">Figure 2: Data Integrity & Completeness Audit (100% Clean)</div>
    </div>
  </div>

  <h2>2. Signal Densities & Raw Correlation Structure</h2>
  <div class="grid-2">
    <div class="img-wrap">
      <img src="__IMG_FEAT_DIST__" alt="Feature Distributions">
      <div class="img-caption">Figure 3: Feature Densities by Class (Genuine vs Repeat Abuse)</div>
    </div>
    <div class="img-wrap">
      <img src="__IMG_CORR_MAT__" alt="Correlation Matrix">
      <div class="img-caption">Figure 4: Correlation Matrix of Raw Heuristic Signals</div>
    </div>
  </div>
</div>

<!-- ========================================== PAGE 4 ========================================== -->
<div class="page">
  <div class="doc-header">
    <div class="doc-title">Causal Feature Engineering & 20-Feature Schema</div>
    <div class="doc-subtitle">Strict Temporal Causality & Entity Resolution</div>
  </div>

  <h2>1. 20-Dimensional Dense Feature Schema</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Feature Name</th>
        <th>Computation Formula</th>
        <th>Signal Family</th>
        <th>Correlation (r)</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>1</td><td><code>graph_component_size</code></td><td>Union-Find component size across (Card, Dev, Subnet)</td><td>Entity Graph</td><td><strong>+0.746</strong></td></tr>
      <tr><td>2</td><td><code>attrs_reused_count</code></td><td>Count of reused identity families (0 to 4)</td><td>Meta-Reuse</td><td><strong>+0.741</strong></td></tr>
      <tr><td>3</td><td><code>subnet_signups_last_24h</code></td><td>Sliding queue count in /24 subnet in past 86,400s</td><td>24h Velocity</td><td><strong>+0.703</strong></td></tr>
      <tr><td>4</td><td><code>ip_subnet_reuse_count</code></td><td>Historical lifetime count from same /24 prefix</td><td>Lifetime Network</td><td><strong>+0.702</strong></td></tr>
      <tr><td>5</td><td><code>email_local_has_plus_tag</code></td><td>Flag for '+' in local email part</td><td>Lexical Alias</td><td><strong>+0.649</strong></td></tr>
      <tr><td>6</td><td><code>is_disposable_email_domain</code></td><td>Domain lookup against disposable blacklist</td><td>Reputation</td><td><strong>+0.557</strong></td></tr>
      <tr><td>7</td><td><code>payment_reuse_count</code></td><td>Historical lifetime reuse of payment token</td><td>Payment Replay</td><td><strong>+0.550</strong></td></tr>
      <tr><td>8</td><td><code>payment_signups_last_24h</code></td><td>Sliding queue card reuse in past 24h</td><td>Payment Velocity</td><td><strong>+0.550</strong></td></tr>
      <tr><td>9</td><td><code>device_reuse_count</code></td><td>Historical lifetime reuse of device hash</td><td>Device Replay</td><td><strong>+0.501</strong></td></tr>
      <tr><td>10</td><td><code>device_signups_last_24h</code></td><td>Sliding queue device reuse in past 24h</td><td>Device Velocity</td><td><strong>+0.501</strong></td></tr>
      <tr><td>11</td><td><code>email_local_has_digits</code></td><td>Flag for numeric characters in email username</td><td>Lexical Pattern</td><td><strong>+0.403</strong></td></tr>
      <tr><td>12</td><td><code>ip_reuse_count</code></td><td>Historical exact IP address reuse count</td><td>Exact IP Replay</td><td><strong>+0.389</strong></td></tr>
      <tr><td>13</td><td><code>payment_ip_country_mismatch</code></td><td>Flag for IP Country != Payment BIN Country</td><td>Geo / BIN Fraud</td><td><strong>+0.343</strong></td></tr>
      <tr><td>14</td><td><code>name_similarity_score</code></td><td>Max SequenceMatcher ratio over recent 300 names</td><td>Fuzzy Lexical</td><td><strong>+0.309</strong></td></tr>
      <tr><td>15</td><td><code>device_signups_last_hour</code></td><td>Fixed 1h bucket count for device ID</td><td>1h Velocity</td><td><strong>+0.285</strong></td></tr>
      <tr><td>16</td><td><code>area_freq</code></td><td>Target-independent frequency encoding P(City)</td><td>Demographics</td><td>+0.022</td></tr>
      <tr><td>17</td><td><code>is_odd_hour</code></td><td>Flag for signup hour in [00:00 - 05:59]</td><td>Temporal Risk</td><td>+0.021</td></tr>
      <tr><td>18</td><td><code>device_os_freq</code></td><td>Frequency encoding P(OS)</td><td>Hardware Spec</td><td>+0.014</td></tr>
      <tr><td>19</td><td><code>signup_hour</code></td><td>Continuous signup hour integer (0 - 23)</td><td>Time of Day</td><td>-0.032</td></tr>
      <tr><td>20</td><td><code>is_free_email_domain</code></td><td>Domain lookup against Gmail/Yahoo/Outlook</td><td>Baseline Genuine</td><td><strong>-0.226</strong></td></tr>
    </tbody>
  </table>

  <h2>2. The Causal Temporal Guarantee</h2>
  <div class="box box-success">
    <p><strong>Zero Lookahead Leakage Contract:</strong> All counters, sliding windows, and union-find disjoint sets are queried strictly prior to event timestamp t_i. Memory state is updated <em>only after</em> computing event i's risk score, guaranteeing that offline validation results match production serving.</p>
  </div>
</div>

<!-- ========================================== PAGE 5 ========================================== -->
<div class="page">
  <div class="doc-header">
    <div class="doc-title">Model Benchmark & 10-Fold Stratified Cross-Validation</div>
    <div class="doc-subtitle">Comparative Algorithm Selection on Train Set (N=7,415)</div>
  </div>

  <h2>1. 10-Fold Stratified Cross-Validation Benchmark</h2>
  <div class="img-wrap">
    <img src="__IMG_MODEL_COMP__" alt="10-Fold CV Model Comparison" style="max-height: 200px;">
    <div class="img-caption">Figure 5: 10-Fold CV Benchmark across 7 Algorithms (Ranked by Recall)</div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Rank</th>
        <th>Model Architecture</th>
        <th>Abuse Recall (Mean ± Std)</th>
        <th>F1-Score</th>
        <th>ROC-AUC</th>
        <th>PR-AUC</th>
        <th>Operational Decision</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background: #eff6ff; font-weight: 700;">
        <td>🥇</td>
        <td>XGBoost (StandardScaler + Tree)</td>
        <td>0.9305 ± 0.012</td>
        <td>0.9621</td>
        <td>0.9725</td>
        <td>0.9705</td>
        <td>SELECTED: Highest Abuse Recall</td>
      </tr>
      <tr>
        <td>🥈</td>
        <td>Decision Tree (Max Depth 10)</td>
        <td>0.9296 ± 0.015</td>
        <td>0.9570</td>
        <td>0.9651</td>
        <td>0.9565</td>
        <td>High split variance</td>
      </tr>
      <tr>
        <td>🥉</td>
        <td>Random Forest (100 Trees)</td>
        <td>0.9287 ± 0.011</td>
        <td>0.9609</td>
        <td>0.9729</td>
        <td>0.9686</td>
        <td>Strong, slightly lower recall</td>
      </tr>
      <tr>
        <td>4</td>
        <td>Gradient Boosting (100 Trees)</td>
        <td>0.9273 ± 0.013</td>
        <td>0.9618</td>
        <td>0.9777</td>
        <td>0.9746</td>
        <td>Higher inference latency</td>
      </tr>
      <tr>
        <td>5</td>
        <td>Logistic Regression (L2)</td>
        <td>0.9273 ± 0.012</td>
        <td>0.9618</td>
        <td>0.9791</td>
        <td>0.9756</td>
        <td>Misses high-order interactions</td>
      </tr>
      <tr>
        <td>6</td>
        <td>SVM (RBF Kernel)</td>
        <td>0.9273 ± 0.012</td>
        <td>0.9618</td>
        <td>0.9660</td>
        <td>0.9676</td>
        <td>O(N^2) latency exceeds 200ms SLA</td>
      </tr>
      <tr>
        <td>7</td>
        <td>KNN (k = 7)</td>
        <td>0.9097 ± 0.018</td>
        <td>0.9524</td>
        <td>0.9661</td>
        <td>0.9552</td>
        <td>Curse of dimensionality degradation</td>
      </tr>
    </tbody>
  </table>

  <h2>2. Selection Rationale: Recall-First Prioritization</h2>
  <p>In free trial defense, missed abuse costs direct server and API expenditure, whereas borderline false positives are gracefully routed to step-up verification. XGBoost demonstrated the highest recall across all folds.</p>
</div>

<!-- ========================================== PAGE 6 ========================================== -->
<div class="page">
  <div class="doc-header">
    <div class="doc-title">Held-Out Test Set Evaluation & Threshold Tuning</div>
    <div class="doc-subtitle">Validation on N = 1,854 Events (554 Abuse Cases)</div>
  </div>

  <h2>1. Confusion Matrix & Threshold Analysis</h2>
  <div class="grid-2">
    <div class="img-wrap">
      <img src="__IMG_CM__" alt="Confusion Matrix">
      <div class="img-caption">Figure 6: Confusion Matrix at Tuned Threshold T = 0.060 (527 / 554 Caught)</div>
    </div>
    <div class="img-wrap">
      <img src="__IMG_THRESH__" alt="Threshold Analysis">
      <div class="img-caption">Figure 7: Precision-Recall-F1 Trade-Off Scan across Thresholds</div>
    </div>
  </div>

  <h2>2. ROC, Precision-Recall & Calibration Curves</h2>
  <div class="grid-3">
    <div class="img-wrap">
      <img src="__IMG_ROC__" alt="ROC Curve">
      <div class="img-caption">Figure 8: ROC Curve (AUC = 0.976)</div>
    </div>
    <div class="img-wrap">
      <img src="__IMG_PR__" alt="PR Curve">
      <div class="img-caption">Figure 9: PR Curve (AP = 0.974)</div>
    </div>
    <div class="img-wrap">
      <img src="__IMG_CALIB__" alt="Calibration Curve">
      <div class="img-caption">Figure 10: Calibration Curve</div>
    </div>
  </div>

  <h2>3. Test Set Performance Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Metric</th>
        <th>PRD Baseline (v1)</th>
        <th>Enhanced Pipeline (v2)</th>
        <th>Operational Benefit</th>
      </tr>
    </thead>
    <tbody>
      <tr><td><strong>Abuse Recall</strong></td><td>90.2%</td><td><strong>95.1%</strong></td><td>Catches 527 of 554 repeat syndicates (+4.9% gain)</td></tr>
      <tr><td><strong>Abuse Precision</strong></td><td>75.0%</td><td><strong>78.8%</strong></td><td>Controlled review queue volume</td></tr>
      <tr><td><strong>ROC-AUC Score</strong></td><td>0.962</td><td><strong>0.976</strong></td><td>Near-perfect ranking discrimination</td></tr>
      <tr><td><strong>Average Precision (PR-AUC)</strong></td><td>0.949</td><td><strong>0.974</strong></td><td>Superior precision across all operational thresholds</td></tr>
      <tr><td><strong>Overall Accuracy</strong></td><td>89.1%</td><td><strong>90.9%</strong></td><td>High overall reliability</td></tr>
    </tbody>
  </table>
</div>

<!-- ========================================== PAGE 7 ========================================== -->
<div class="page">
  <div class="doc-header">
    <div class="doc-title">Explainability, Interpretability & SHAP Analysis</div>
    <div class="doc-subtitle">Global Attribution & Real-Time Local Explanations</div>
  </div>

  <h2>1. Global Feature Importance & SHAP Summary Plot</h2>
  <div class="grid-2">
    <div class="img-wrap">
      <img src="__IMG_FEAT_IMP__" alt="Feature Importance">
      <div class="img-caption">Figure 11: Global Feature Importance (Gini Gain)</div>
    </div>
    <div class="img-wrap">
      <img src="__IMG_SHAP__" alt="SHAP Summary">
      <div class="img-caption">Figure 12: SHAP Feature Attribution (Directional Impact)</div>
    </div>
  </div>

  <h2>2. Local Per-Event Signal Decomposition</h2>
  <p>For every scored event, the engine generates an interpretable signal contribution breakdown:</p>
  <pre style="font-size: 7.5pt;">
{
  "user_id": "demo_fraud_syndicate_03",
  "risk_score": 99.9,
  "verdict": "REPEAT / LIKELY ABUSE",
  "recommended_action": "BLOCK / REQUIRE PAYMENT",
  "model_confidence_pct": 99.9,
  "decision_threshold": 6.0,
  "signal_breakdown": {
    "email_local_has_plus_tag": 18.68,
    "is_disposable_email_domain": 6.97,
    "signup_hour": 0.33,
    "email_local_has_digits": 0.19,
    "name_similarity_score": 0.14
  }
}</pre>
</div>

<!-- ========================================== PAGE 8 ========================================== -->
<div class="page">
  <div class="doc-header">
    <div class="doc-title">Live Scenarios & Attack Evasion Proof</div>
    <div class="doc-subtitle">Side-by-Side Scoring & Dynamic Syndicate Trace</div>
  </div>

  <h2>1. Scenario Comparison: Genuine vs Fraudulent Scorecards</h2>
  <div class="grid-2">
    <div class="img-wrap">
      <img src="__IMG_GEN_CARD__" alt="Genuine User Scorecard">
      <div class="img-caption">Figure 13: Live Model Output for Genuine User (0.5 / 100 -> ALLOW)</div>
    </div>
    <div class="img-wrap">
      <img src="__IMG_FRAUD_CARD__" alt="Fraud Syndicate Scorecard">
      <div class="img-caption">Figure 14: Live Model Output for Fraud Syndicate (99.9 / 100 -> BLOCK)</div>
    </div>
  </div>

  <h2>2. Dynamic Attack Evasion Graph Trace & Policy Overview</h2>
  <div class="grid-2">
    <div class="img-wrap">
      <img src="__IMG_EVASION_TRACE__" alt="Attack Evasion Graph Trace">
      <div class="img-caption">Figure 15: Attack Evasion Trace (Graph Cluster Growth Catches Rotated Identity)</div>
    </div>
    <div class="img-wrap">
      <img src="__IMG_DASH_SUMMARY__" alt="Policy Dashboard Summary">
      <div class="img-caption">Figure 16: 3-Band Policy Operational Framework</div>
    </div>
  </div>
</div>

<!-- ========================================== PAGE 9 ========================================== -->
<div class="page">
  <div class="doc-header">
    <div class="doc-title">Interview Defense Master Playbook</div>
    <div class="doc-subtitle">High-Frequency Technical Questions & Solutions</div>
  </div>

  <h2>1. The 2-Minute Elevator Pitch</h2>
  <div class="box box-primary">
    <p><em>"In this project, I engineered a real-time risk detection system to prevent free-trial farming and multi-accounting syndicates in SaaS applications. Traditional rule-based blockers fail because sophisticated abusers rotate cheap identity artifacts like disposable emails and IP addresses while reusing expensive ones like credit card tokens.</em></p>
    <p><em>To counter this, I developed a strictly causal feature engineering pipeline featuring an <strong>Incremental Union-Find Entity Graph</strong> connecting payment tokens, device hashes, and /24 subnets. This graph linkage proved to be our most predictive signal (r=0.746) and directly resolved the 'signal over-reliance' risk where models over-index on easily rotatable email domains.</em></p>
    <p><em>I benchmarked 7 algorithms across 10-fold Stratified Cross-Validation, selected <strong>XGBoost</strong>, and tuned the decision threshold to 0.060 under a Precision &ge; 75% SLA, achieving <strong>95.1% abuse recall</strong>, an <strong>ROC-AUC of 0.976</strong>, and a <strong>PR-AUC of 0.974</strong>. The entire scoring engine runs in &lt;20ms with full SHAP explainability and a 3-band action policy."</em></p>
  </div>

  <h2>2. Deep-Dive Interview Defense Q&A</h2>
  
  <h3>Q1: "Why did you prioritize Recall over Accuracy or Precision?"</h3>
  <p><strong>Answer:</strong> In fraud prevention, the business cost matrix is asymmetric. A False Negative (missed abuser) results in permanent compute and quota loss. A False Positive in our architecture does NOT trigger an immediate hard ban; our downstream 3-band policy routes borderline scores (3.3 &le; Score &lt; 6.0) to step-up verification challenges (SMS OTP, CAPTCHA, or $1 card micro-auth). Genuine users pass easily, while automated attack scripts fail.</p>

  <h3>Q2: "How did you guarantee zero data leakage in feature extraction?"</h3>
  <p><strong>Answer:</strong> All counters, sliding velocity queues, and union-find graph components were computed strictly <strong>causally</strong>: events were sorted chronologically, queried against historical state seen <em>prior to</em> event timestamp t_i, and memory state was updated <em>only after</em> calculating event i's features.</p>

  <h3>Q3: "How would you scale this architecture to 50,000 requests per second in production?"</h3>
  <p><strong>Answer:</strong> (1) An in-memory <strong>Redis Cluster</strong> with sliding-window sorted sets (<code>ZADD</code>/<code>ZREMRANGEBYSCORE</code>) for sub-millisecond velocity lookups. (2) A distributed Union-Find Graph service. (3) Containerized <strong>Triton Inference Server</strong> or FastAPI pods on Kubernetes with auto-scaling to keep inference latency under 10ms. (4) Evidently AI tracking Population Stability Index (PSI) to trigger automated retraining on concept drift.</p>
</div>

</body>
</html>
"""

# Replace image placeholders
html_final = template.replace("__IMG_TARGET_DIST__", img_target_dist)
html_final = html_final.replace("__IMG_MISSING_VALS__", img_missing_vals)
html_final = html_final.replace("__IMG_FEAT_DIST__", img_feat_dist)
html_final = html_final.replace("__IMG_CORR_MAT__", img_corr_mat)
html_final = html_final.replace("__IMG_MODEL_COMP__", img_model_comp)
html_final = html_final.replace("__IMG_CM__", img_cm)
html_final = html_final.replace("__IMG_ROC__", img_roc)
html_final = html_final.replace("__IMG_PR__", img_pr)
html_final = html_final.replace("__IMG_CALIB__", img_calib)
html_final = html_final.replace("__IMG_THRESH__", img_thresh)
html_final = html_final.replace("__IMG_FEAT_IMP__", img_feat_imp)
html_final = html_final.replace("__IMG_SHAP__", img_shap)
html_final = html_final.replace("__IMG_GEN_CARD__", img_gen_card)
html_final = html_final.replace("__IMG_FRAUD_CARD__", img_fraud_card)
html_final = html_final.replace("__IMG_EVASION_TRACE__", img_evasion_trace)
html_final = html_final.replace("__IMG_DASH_SUMMARY__", img_dash_summary)

with open(HTML_TEMP_PATH, "w", encoding="utf-8") as f:
    f.write(html_final)

print(f"Generated temporary HTML report at {HTML_TEMP_PATH}")

browser_paths = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

browser_exe = None
for p in browser_paths:
    if os.path.exists(p):
        browser_exe = p
        break

if browser_exe:
    print(f"Compiling PDF via headless browser: {browser_exe}...")
    cmd = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_OUTPUT_PATH}",
        HTML_TEMP_PATH
    ]
    subprocess.run(cmd, check=True)
    print(f"\n==============================================================")
    print(f"PDF COMPILED SUCCESSFULLY!")
    print(f"Saved to: {PDF_OUTPUT_PATH}")
    print(f"Size: {os.path.getsize(PDF_OUTPUT_PATH) / 1024:.1f} KB")
    print(f"==============================================================")
else:
    print("No headless browser found. HTML report is available at:", HTML_TEMP_PATH)

if os.path.exists(HTML_TEMP_PATH):
    os.remove(HTML_TEMP_PATH)
