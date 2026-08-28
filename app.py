"""
INTERACTIVE WEB GUI & INFERENCE SERVER
=======================================
A standalone, zero-dependency interactive dashboard for scoring unseen events,
running attack simulations, batch scoring CSVs, and exploring evaluation visuals.

Run:
  py scripts/app.py
  Then open: http://localhost:8080 in your browser.
"""

import os
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import pandas as pd
import numpy as np

from predict import FraudRiskEngine, FEATURE_COLS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

engine = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Free Trial Abuse & Multi-Accounting Risk Detection System</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0b0f19;
      --card-bg: rgba(18, 24, 38, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-blue: #3b82f6;
      --accent-indigo: #6366f1;
      --accent-purple: #8b5cf6;
      --risk-low: #10b981;
      --risk-med: #f59e0b;
      --risk-high: #ef4444;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: radial-gradient(circle at 15% 15%, #131d36 0%, var(--bg-dark) 55%), #070a12;
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: rgba(11, 15, 25, 0.85);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--card-border);
      padding: 16px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-icon {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 18px;
    }
    .brand-title {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.5px;
    }
    .brand-badge {
      background: rgba(99, 102, 241, 0.2);
      border: 1px solid rgba(99, 102, 241, 0.4);
      color: #a5b4fc;
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 20px;
      font-weight: 600;
    }
    nav {
      display: flex;
      gap: 8px;
    }
    .nav-btn {
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      padding: 8px 16px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 500;
      transition: all 0.2s;
    }
    .nav-btn:hover, .nav-btn.active {
      color: #fff;
      background: rgba(255, 255, 255, 0.05);
      border-color: var(--card-border);
    }
    .nav-btn.active {
      background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2));
      border-color: rgba(99, 102, 241, 0.4);
      color: #93c5fd;
    }
    main {
      flex: 1;
      max-width: 1380px;
      width: 100%;
      margin: 0 auto;
      padding: 28px 24px;
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    
    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }
    @media (max-width: 900px) {
      .grid-2 { grid-template-columns: 1fr; }
    }
    
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 24px;
      backdrop-filter: blur(12px);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .card-title {
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    
    .presets-bar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 20px;
    }
    .preset-btn {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #d1d5db;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      cursor: pointer;
      font-weight: 500;
      transition: all 0.2s;
    }
    .preset-btn:hover {
      background: rgba(59, 130, 246, 0.2);
      border-color: var(--accent-blue);
      color: #fff;
    }
    
    .form-group {
      margin-bottom: 14px;
    }
    .form-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    label {
      display: block;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    input, select, textarea {
      width: 100%;
      background: rgba(10, 14, 23, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      padding: 10px 14px;
      color: #fff;
      font-family: inherit;
      font-size: 13px;
      outline: none;
      transition: border-color 0.2s;
    }
    input:focus, select:focus, textarea:focus {
      border-color: var(--accent-blue);
      box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    
    .btn-primary {
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-indigo));
      border: none;
      color: #fff;
      padding: 12px 24px;
      border-radius: 10px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      width: 100%;
      box-shadow: 0 4px 15px rgba(59, 130, 246, 0.35);
      transition: all 0.2s;
      margin-top: 8px;
    }
    .btn-primary:hover {
      opacity: 0.95;
      transform: translateY(-1px);
    }
    
    .gauge-box {
      text-align: center;
      padding: 24px;
      background: rgba(10, 14, 23, 0.6);
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      margin-bottom: 20px;
    }
    .gauge-score {
      font-size: 56px;
      font-weight: 800;
      letter-spacing: -1px;
      line-height: 1;
      margin-bottom: 8px;
      font-family: 'JetBrains Mono', monospace;
    }
    .verdict-badge {
      display: inline-block;
      padding: 6px 16px;
      border-radius: 20px;
      font-weight: 700;
      font-size: 13px;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }
    .verdict-low { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .verdict-med { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .verdict-high { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-bottom: 20px;
    }
    .meta-card {
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid rgba(255, 255, 255, 0.05);
      padding: 12px;
      border-radius: 8px;
      text-align: center;
    }
    .meta-title { font-size: 11px; color: var(--text-muted); margin-bottom: 4px; text-transform: uppercase; }
    .meta-value { font-size: 14px; font-weight: 600; color: #fff; }
    
    .signals-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    .signals-table th {
      text-align: left;
      color: var(--text-muted);
      padding: 8px 6px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 10px;
    }
    .signals-table td {
      padding: 8px 6px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .signal-bar-wrap {
      background: rgba(255, 255, 255, 0.05);
      height: 6px;
      border-radius: 3px;
      overflow: hidden;
      margin-top: 4px;
    }
    .signal-bar {
      height: 100%;
      border-radius: 3px;
      background: linear-gradient(90deg, var(--accent-blue), var(--risk-high));
    }
    
    .gallery-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 20px;
    }
    .gallery-item {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      overflow: hidden;
    }
    .gallery-item img {
      width: 100%;
      height: auto;
      display: block;
    }
    .gallery-label {
      padding: 12px 16px;
      font-size: 13px;
      font-weight: 600;
      border-top: 1px solid var(--card-border);
      background: rgba(0, 0, 0, 0.2);
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">&#x26E8;</div>
      <div>
        <div class="brand-title">Fraud Detection ML Model</div>
      </div>
      <span class="brand-badge">ML Risk Engine (ROC-AUC 0.973 | Recall 93.8%)</span>
    </div>
    <nav>
      <button class="nav-btn active" onclick="switchTab('scorer')">Live Scorer</button>
      <button class="nav-btn" onclick="switchTab('batch')">Batch CSV</button>
      <button class="nav-btn" onclick="switchTab('eval')">Model Visuals</button>
      <button class="nav-btn" onclick="switchTab('arch')">Architecture</button>
    </nav>
  </header>

  <main>
    <!-- TAB 1: LIVE SCORER -->
    <section id="tab-scorer" class="tab-content active">
      <div class="grid-2">
        <!-- Input Form -->
        <div class="card">
          <div class="card-title">
            <span>New Signup Event</span>
            <span style="font-size: 11px; font-weight: 400; color: var(--text-muted);">Real-time Scoring</span>
          </div>

          <div class="presets-bar">
            <span style="font-size: 11px; color: var(--text-muted); align-self: center; font-weight: 600;">Attack Scenarios:</span>
            <button class="preset-btn" style="border-color: #10b981; color: #6ee7b7;" onclick="loadPreset('first_time')">1. Clean First-Time Signup (Low Risk)</button>
            <button class="preset-btn" style="border-color: #ef4444; color: #fca5a5;" onclick="loadPreset('repeat_attack')">2. Repeat Attack (Same Card/Device -> High Risk)</button>
            <button class="preset-btn" onclick="loadPreset('zero_shot_burner')">3. Zero-Shot Burner Email</button>
            <button class="preset-btn" onclick="loadPreset('geo_mismatch')">4. BIN Geo Mismatch</button>
            <button class="preset-btn" style="background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.4); margin-left: auto;" onclick="clearHistory()">Reset Session</button>
          </div>

          <form id="score-form" onsubmit="handleScore(event)">
            <div class="form-row">
              <div class="form-group">
                <label>User Full Name</label>
                <input type="text" id="inp-name" value="Karthik Tatineni" required>
              </div>
              <div class="form-group">
                <label>Signup Email</label>
                <input type="email" id="inp-email" value="karthik@gmail.com" required>
              </div>
            </div>

            <div class="form-row">
              <div class="form-group">
                <label>IP Address</label>
                <input type="text" id="inp-ip" value="103.20.10.5" required>
              </div>
              <div class="form-group">
                <label>Device ID / Fingerprint</label>
                <input type="text" id="inp-device" value="dev_phone_karthik_101" required>
              </div>
            </div>

            <div class="form-row">
              <div class="form-group">
                <label>Payment Token</label>
                <input type="text" id="inp-payment" value="pm_visa_karthik_101" required>
              </div>
              <div class="form-group">
                <label>Signup Area / City</label>
                <select id="inp-area">
                  <option value="mumbai" selected>Mumbai (IN)</option>
                  <option value="delhi">Delhi (IN)</option>
                  <option value="bangalore">Bangalore (IN)</option>
                  <option value="ahmedabad">Ahmedabad (IN)</option>
                  <option value="new_york">New York (US)</option>
                  <option value="london">London (GB)</option>
                  <option value="singapore">Singapore (SG)</option>
                  <option value="dubai">Dubai (AE)</option>
                </select>
              </div>
            </div>

            <div class="form-row">
              <div class="form-group">
                <label>Device OS</label>
                <select id="inp-os">
                  <option value="android" selected>Android</option>
                  <option value="ios">iOS</option>
                  <option value="windows">Windows</option>
                  <option value="macos">macOS</option>
                  <option value="linux">Linux</option>
                </select>
              </div>
              <div class="form-group">
                <label>Payment Country (BIN)</label>
                <select id="inp-payment-country">
                  <option value="IN" selected>India (IN)</option>
                  <option value="US">United States (US)</option>
                  <option value="GB">United Kingdom (GB)</option>
                  <option value="SG">Singapore (SG)</option>
                </select>
              </div>
            </div>

            <button type="submit" class="btn-primary" id="btn-score">Score Signup Event & Log to Memory</button>
          </form>
        </div>

        <!-- Output Score Panel -->
        <div class="card">
          <div class="card-title">
            <span>Risk Assessment Result</span>
            <span id="res-time" style="font-size: 11px; font-weight: 400; color: var(--text-muted); font-family: monospace;"></span>
          </div>

          <div class="gauge-box">
            <div class="gauge-score" id="res-score" style="color: var(--risk-low);">--</div>
            <div id="res-verdict" class="verdict-badge verdict-low">READY TO SCORE</div>
          </div>

          <div class="meta-grid">
            <div class="meta-card">
              <div class="meta-title">Recommended Action</div>
              <div class="meta-value" id="res-action">--</div>
            </div>
            <div class="meta-card">
              <div class="meta-title">Confidence</div>
              <div class="meta-value" id="res-conf">--</div>
            </div>
            <div class="meta-card">
              <div class="meta-title">Graph Cluster</div>
              <div class="meta-value" id="res-graph">--</div>
            </div>
          </div>

          <div class="card-title" style="margin-top: 16px; margin-bottom: 8px;">
            <span style="font-size: 13px;">Additive Signal Point Breakdown</span>
          </div>
          <table class="signals-table">
            <thead>
              <tr>
                <th>Signal Name</th>
                <th>Raw Value</th>
                <th style="width: 45%;">Points Added</th>
              </tr>
            </thead>
            <tbody id="res-signals">
              <tr><td colspan="3" style="text-align: center; color: var(--text-muted); padding: 20px;">Submit a signup event to view signal breakdown.</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Live Session Activity History Table -->
      <div class="card" style="margin-top: 24px;">
        <div class="card-title">
          <span>Live Session Activity & Entity Memory Log</span>
          <span style="font-size: 11px; font-weight: 400; color: var(--text-muted);">Real-time Causal Audit Trail</span>
        </div>
        <p style="font-size: 12.5px; color: var(--text-muted); margin-bottom: 12px;">
          Every signup tested below is causally committed into the live feature store. Subsequent attempts reusing the same card or device will immediately be flagged with High Risk.
        </p>
        <table class="signals-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Time</th>
              <th>User Name & Email</th>
              <th>Payment Token</th>
              <th>Device ID</th>
              <th>IP Address</th>
              <th>Risk Score</th>
              <th>Verdict</th>
              <th>Key Detection Signals</th>
            </tr>
          </thead>
          <tbody id="history-tbody">
            <tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 16px;">No signups submitted yet in this session.</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- TAB 2: BATCH CSV -->
    <section id="tab-batch" class="tab-content">
      <div class="card">
        <div class="card-title">Batch Unseen Signups Scoring</div>
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 16px;">
          Paste raw signup event records in CSV format to perform real-time causal scoring across all events.
        </p>
        <textarea id="batch-csv" rows="6" placeholder="name,email,ip_address,device_id,payment_token,area,device_os
Alice Green,alice@gmail.com,192.168.1.5,dev_a1,pm_a1,mumbai,android
Bob White,bob+trial@mailinator.com,39.173.180.50,dev_b1,pm_424776171fe7,ahmedabad,ios"></textarea>
        <button class="btn-primary" style="margin-top: 12px;" onclick="handleBatchScore()">Score Batch Signups</button>

        <div id="batch-results-wrap" style="margin-top: 24px; display: none;">
          <div class="card-title" style="font-size: 14px;">Batch Predictions</div>
          <table class="signals-table">
            <thead>
              <tr>
                <th>User / Name</th>
                <th>Risk Score</th>
                <th>Verdict</th>
                <th>Action</th>
                <th>Top Signal</th>
              </tr>
            </thead>
            <tbody id="batch-tbody"></tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 3: VISUALS GALLERY -->
    <section id="tab-eval" class="tab-content">
      <div class="gallery-grid">
        <div class="gallery-item">
          <img src="/visuals/evaluation/model_comparison.png" alt="Model Comparison">
          <div class="gallery-label">10-Fold CV: 7 Model Comparison (Ranked by Recall)</div>
        </div>
        <div class="gallery-item">
          <img src="/visuals/evaluation/confusion_matrix.png" alt="Confusion Matrix">
          <div class="gallery-label">Confusion Matrix (Threshold 0.055, Recall 95.1%)</div>
        </div>
        <div class="gallery-item">
          <img src="/visuals/evaluation/roc_curve.png" alt="ROC Curve">
          <div class="gallery-label">ROC Curve (AUC = 0.976)</div>
        </div>
        <div class="gallery-item">
          <img src="/visuals/evaluation/precision_recall_curve.png" alt="PR Curve">
          <div class="gallery-label">Precision-Recall Curve (PR-AUC = 0.974)</div>
        </div>
        <div class="gallery-item">
          <img src="/visuals/evaluation/calibration_curve.png" alt="Calibration Curve">
          <div class="gallery-label">Probability Calibration Curve (Reliability Diagram)</div>
        </div>
        <div class="gallery-item">
          <img src="/visuals/evaluation/threshold_analysis.png" alt="Threshold Analysis">
          <div class="gallery-label">Precision-Recall-F1 Threshold Trade-off Analysis</div>
        </div>
        <div class="gallery-item">
          <img src="/visuals/explainability/feature_importance.png" alt="Feature Importance">
          <div class="gallery-label">Global Feature Importance (Gini Gain)</div>
        </div>
        <div class="gallery-item">
          <img src="/visuals/explainability/shap_summary.png" alt="SHAP Summary">
          <div class="gallery-label">SHAP Feature Attribution Summary Plot</div>
        </div>
        <div class="gallery-item">
          <img src="/visuals/eda/target_distribution.png" alt="Target Distribution">
          <div class="gallery-label">EDA: Class Distribution (Imbalanced 70/30)</div>
        </div>
        <div class="gallery-item">
          <img src="/visuals/eda/missing_values.png" alt="Missing Values">
          <div class="gallery-label">EDA: Missing Values & Data Integrity Audit</div>
        </div>
        <div class="gallery-item">
          <img src="/visuals/eda/feature_distributions.png" alt="Feature Distributions">
          <div class="gallery-label">EDA: Feature Distributions by Class</div>
        </div>
        <div class="gallery-item">
          <img src="/visuals/eda/correlation_matrix.png" alt="Correlation Matrix">
          <div class="gallery-label">EDA: Raw Signal Correlation Matrix</div>
        </div>
      </div>
    </section>

    <!-- TAB 4: ARCHITECTURE -->
    <section id="tab-arch" class="tab-content">
      <div class="card">
        <div class="card-title">Production System Architecture</div>
        <p style="font-size: 14px; line-height: 1.6; color: #d1d5db; margin-bottom: 20px;">
          This system defends against multi-accounting and free trial abuse by linking weak identity artifacts (IP subnets, device hashes, payment tokens, fuzzy names) using <strong>Incremental Union-Find Connected Components</strong>, rolling velocity counters, and an <strong>XGBoost classifier</strong> tuned for high recall.
        </p>
        <div style="background: rgba(0,0,0,0.4); padding: 20px; border-radius: 8px; font-family: monospace; font-size: 12px; line-height: 1.5; color: #93c5fd; overflow-x: auto;">
  [SIGNUP EVENT] (name, email, ip, device_id, payment_token, area, time)
         |
  [IDENTITY RESOLUTION & GRAPH LAYER]
   - Subnet /24 extraction
   - Disposable email & plus-tag regex
   - Causal Incremental Union-Find (Component Size & Density)
   - 24h & 1h Rolling Velocity Windows
         |
  [FEATURE STORE (Redis / In-Memory)]
   - Running counters strictly prior to event timestamp
         |
  [ML RISK MODEL (XGBoost Pipeline)]
   - Probability P(abuse) in [0, 1]
         |
  [3-BAND DECISION & EXPLAINABILITY LAYER]
   - 0 - 25        -&gt; ALLOW (New / Genuine User)
   - 25 - 50       -&gt; STEP-UP / MANUAL REVIEW (Suspicious Grey Zone)
   - &gt;= 50         -&gt; BLOCK TRIAL / DEMAND PAYMENT (Repeat Abuse)
        </div>
      </div>
    </section>
  </main>

  <script>
    function switchTab(tabId) {
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
      document.getElementById('tab-' + tabId).classList.add('active');
      event.target.classList.add('active');
    }

    let sharedCard = "pm_karthik_card_" + Math.floor(1000 + Math.random() * 9000);
    let sharedDevice = "dev_karthik_phone_" + Math.floor(1000 + Math.random() * 9000);
    let sessionHistory = [];

    const PRESETS = {
      first_time: {
        name: "Karthik Tatineni",
        email: "karthik@gmail.com",
        ip: "103.20.10.5",
        device: sharedDevice,
        payment: sharedCard,
        area: "mumbai",
        os: "android",
        country: "IN"
      },
      repeat_attack: {
        name: "Karthik T",
        email: "karthik.alt+trial2@yahoo.com",
        ip: "103.20.10.88",
        device: sharedDevice,
        payment: sharedCard,
        area: "mumbai",
        os: "android",
        country: "IN"
      },
      zero_shot_burner: {
        name: "Quick Farmer",
        email: "farmer99@mailinator.com",
        ip: "198.51.100.12",
        device: "dev_fresh_" + Math.random().toString(36).substring(7),
        payment: "pm_fresh_" + Math.random().toString(36).substring(7),
        area: "dubai",
        os: "windows",
        country: "AE"
      },
      geo_mismatch: {
        name: "Card Swapper",
        email: "user456@gmail.com",
        ip: "103.21.244.10",
        device: "dev_fresh_" + Math.random().toString(36).substring(7),
        payment: "pm_us_stolen_card",
        area: "mumbai",
        os: "ios",
        country: "US"
      }
    };

    function loadPreset(key) {
      const p = PRESETS[key];
      if (!p) return;
      document.getElementById('inp-name').value = p.name;
      document.getElementById('inp-email').value = p.email;
      document.getElementById('inp-ip').value = p.ip;
      document.getElementById('inp-device').value = p.device;
      document.getElementById('inp-payment').value = p.payment;
      document.getElementById('inp-area').value = p.area;
      document.getElementById('inp-os').value = p.os;
      document.getElementById('inp-payment-country').value = p.country;
      document.getElementById('score-form').dispatchEvent(new Event('submit'));
    }

    function clearHistory() {
      sessionHistory = [];
      sharedCard = "pm_karthik_card_" + Math.floor(1000 + Math.random() * 9000);
      sharedDevice = "dev_karthik_phone_" + Math.floor(1000 + Math.random() * 9000);
      PRESETS.first_time.payment = sharedCard;
      PRESETS.first_time.device = sharedDevice;
      PRESETS.repeat_attack.payment = sharedCard;
      PRESETS.repeat_attack.device = sharedDevice;
      renderHistoryTable();
      alert("Session memory reset! You can now test a clean first-time signup.");
    }

    function renderHistoryTable() {
      const tbody = document.getElementById('history-tbody');
      if (sessionHistory.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 16px;">No signups submitted yet in this session.</td></tr>';
        return;
      }
      tbody.innerHTML = '';
      sessionHistory.forEach((item, idx) => {
        const tr = document.createElement('tr');
        const badgeClass = item.severity === 'low' ? 'verdict-low' : item.severity === 'medium' ? 'verdict-med' : 'verdict-high';
        const topSignals = Object.entries(item.signals).filter(([k, v]) => v > 0).map(([k, v]) => `${k} (+${v.toFixed(0)}p)`).join(', ');
        tr.innerHTML = `
          <td style="font-weight: 700; font-family: monospace;">#${sessionHistory.length - idx}</td>
          <td style="font-size: 11px; color: var(--text-muted); font-family: monospace;">${item.time}</td>
          <td><strong>${item.name}</strong><br><span style="font-size: 11px; color: var(--text-muted);">${item.email}</span></td>
          <td style="font-family: monospace; font-size: 12px; color: #93c5fd;">${item.payment}</td>
          <td style="font-family: monospace; font-size: 12px; color: #d8b4fe;">${item.device}</td>
          <td style="font-family: monospace; font-size: 12px;">${item.ip}</td>
          <td style="font-weight: 800; font-family: monospace; font-size: 13px; color: ${item.severity === 'low' ? '#6ee7b7' : item.severity === 'medium' ? '#fde68a' : '#fca5a5'};">${item.risk_score.toFixed(1)}</td>
          <td><span class="verdict-badge ${badgeClass}" style="font-size: 10px; padding: 2px 7px;">${item.verdict}</span></td>
          <td style="font-size: 11px; font-family: monospace; color: #cbd5e1;">${topSignals || 'None (Clean)'}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    async function handleScore(e) {
      e.preventDefault();
      const payload = {
        name: document.getElementById('inp-name').value,
        email: document.getElementById('inp-email').value,
        ip_address: document.getElementById('inp-ip').value,
        device_id: document.getElementById('inp-device').value,
        payment_token: document.getElementById('inp-payment').value,
        area: document.getElementById('inp-area').value,
        device_os: document.getElementById('inp-os').value,
        payment_country: document.getElementById('inp-payment-country').value,
        signup_time: new Date().toISOString()
      };

      const startT = performance.now();
      try {
        const res = await fetch('/api/score', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        const duration = (performance.now() - startT).toFixed(1);

        document.getElementById('res-time').innerText = duration + ' ms latency';
        document.getElementById('res-score').innerText = data.risk_score.toFixed(1) + ' / 100';
        
        const badge = document.getElementById('res-verdict');
        badge.innerText = data.verdict;
        badge.className = 'verdict-badge ' + (data.severity === 'low' ? 'verdict-low' : data.severity === 'medium' ? 'verdict-med' : 'verdict-high');

        const scoreColor = data.severity === 'low' ? 'var(--risk-low)' : data.severity === 'medium' ? 'var(--risk-med)' : 'var(--risk-high)';
        document.getElementById('res-score').style.color = scoreColor;

        document.getElementById('res-action').innerText = data.recommended_action;
        document.getElementById('res-conf').innerText = data.model_confidence_pct + '%';
        document.getElementById('res-graph').innerText = (data.raw_features.graph_component_size || 1) + ' linked nodes';

        const tbody = document.getElementById('res-signals');
        tbody.innerHTML = '';
        const signals = Object.entries(data.signal_breakdown);

        signals.forEach(([sig, val]) => {
          const raw = data.raw_features[sig];
          const pct = Math.min((val / 30.0) * 100, 100);
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td style="font-weight: 600; font-family: monospace; color: #e2e8f0;">${sig}</td>
            <td style="color: var(--text-muted); font-family: monospace;">${raw !== undefined ? raw : '--'}</td>
            <td>
              <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 2px;">
                <span style="font-weight: bold; color: ${val > 0 ? '#fca5a5' : '#6ee7b7'};">${val > 0 ? '+' : ''}${val.toFixed(1)} pts</span>
              </div>
              <div class="signal-bar-wrap">
                <div class="signal-bar" style="width: ${val > 0 ? pct : 0}%; background: ${val >= 15 ? 'linear-gradient(90deg, #f59e0b, #ef4444)' : '#3b82f6'};"></div>
              </div>
            </td>
          `;
          tbody.appendChild(tr);
        });

        // Add to Session History
        const now = new Date();
        sessionHistory.unshift({
          time: now.toTimeString().split(' ')[0],
          name: payload.name,
          email: payload.email,
          payment: payload.payment_token,
          device: payload.device_id,
          ip: payload.ip_address,
          risk_score: data.risk_score,
          verdict: data.verdict,
          severity: data.severity,
          signals: data.signal_breakdown
        });
        renderHistoryTable();

      } catch (err) {
        alert('Scoring error: ' + err);
      }
    }

    async function handleBatchScore() {
      const text = document.getElementById('batch-csv').value.trim();
      if (!text) return alert('Please paste CSV rows first.');

      const lines = text.split('\\n').filter(l => l.trim().length > 0);
      if (lines.length < 2) return alert('CSV must have a header and at least 1 data row.');

      const headers = lines[0].split(',').map(h => h.trim());
      const rows = lines.slice(1).map(line => {
        const vals = line.split(',').map(v => v.trim());
        const obj = {};
        headers.forEach((h, i) => obj[h] = vals[i]);
        return obj;
      });

      const res = await fetch('/api/score-batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(rows)
      });
      const data = await res.json();

      const tbody = document.getElementById('batch-tbody');
      tbody.innerHTML = '';
      data.forEach(item => {
        const tr = document.createElement('tr');
        const badgeClass = item.severity === 'low' ? 'verdict-low' : item.severity === 'medium' ? 'verdict-med' : 'verdict-high';
        tr.innerHTML = `
          <td><strong>${item.name || item.user_id}</strong><br><span style="font-size:11px;color:var(--text-muted);">${item.email || ''}</span></td>
          <td style="font-weight:700; font-family:monospace;">${item.risk_score.toFixed(1)}</td>
          <td><span class="verdict-badge ${badgeClass}" style="font-size:10px;padding:3px 8px;">${item.verdict}</span></td>
          <td style="font-size:12px;">${item.recommended_action}</td>
          <td style="font-size:11px;font-family:monospace;color:var(--text-muted);">${item.top_signal || '--'}</td>
        `;
        tbody.appendChild(tr);
      });
      document.getElementById('batch-results-wrap').style.display = 'block';
    }

    window.onload = () => { loadPreset('first_time'); };
  </script>
</body>
</html>
"""


class FraudAppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        if path.startswith("/visuals/"):
            rel_path = path[len("/visuals/"):]
            file_path = os.path.join(VISUALS_DIR, rel_path.replace("/", os.sep))
            if os.path.exists(file_path) and os.path.isfile(file_path):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not Found")

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        if path == "/api/score":
            try:
                data = json.loads(body.decode("utf-8"))
                result = engine.score_event(data, update_state=True)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if path == "/api/score-batch":
            try:
                rows = json.loads(body.decode("utf-8"))
                results = []
                for row in rows:
                    res = engine.score_event(row, update_state=True)
                    results.append({
                        "user_id": res["user_id"],
                        "name": row.get("name", res["user_id"]),
                        "email": row.get("email", ""),
                        "risk_score": res["risk_score"],
                        "verdict": res["verdict"],
                        "recommended_action": res["recommended_action"],
                        "severity": res["severity"],
                        "confidence": res["model_confidence_pct"],
                        "top_signal": list(res["signal_breakdown"].keys())[0] if res["signal_breakdown"] else "",
                    })
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(results).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def start_server(port=None):
    global engine
    port = port or int(os.environ.get("PORT", 8080))
    print("Initializing Fraud Risk Engine for Web GUI...")
    engine = FraudRiskEngine(warm_start=True)
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, FraudAppHandler)
    print(f"\n==============================================================")
    print(f"FRAUD DETECTION GUI READY")
    print(f"Open in your browser: http://0.0.0.0:{port}")
    print(f"==============================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    start_server()
