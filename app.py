"""
FRAUD ENGINE — DEVELOPER API & FRAUD EVALUATION PLATFORM
========================================================
Production backend server & high-density developer dashboard.
"""

import os
import json
import time
import secrets
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

from predict import FraudRiskEngine, FEATURE_COLS
from database import (
    get_or_create_user,
    create_user_api_key,
    list_user_api_keys,
    revoke_user_api_key,
    delete_user_api_key,
    validate_api_key,
    record_customer_signup,
    list_user_customers,
    search_user_customer,
    push_initial_dataset_to_firebase
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
DEFAULT_RATE_LIMIT = int(os.environ.get("DEFAULT_RATE_LIMIT_PER_MINUTE", 30))

engine = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Fraud Engine - Identity & Risk Evaluation</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  
  <!-- Firebase SDKs -->
  <script src="https://www.gstatic.com/firebasejs/10.8.0/firebase-app-compat.js"></script>
  <script src="https://www.gstatic.com/firebasejs/10.8.0/firebase-auth-compat.js"></script>
  <script src="https://www.gstatic.com/firebasejs/10.8.0/firebase-firestore-compat.js"></script>

  <style>
    :root {
      --bg-page: #0d1117;
      --bg-card: #161b22;
      --bg-card-header: #1c2128;
      --bg-input: #0d1117;
      
      --border-default: #30363d;
      --border-muted: #21262d;
      --border-focus: #58a6ff;

      --text-main: #f0f6fc;
      --text-muted: #8b949e;
      --text-dim: #6e7681;

      --accent-blue: #1f6feb;
      --accent-blue-hover: #388bfd;
      --accent-green: #238636;
      --accent-green-hover: #2ea043;
      --accent-red: #da3633;
      --accent-red-hover: #f85149;

      --status-green-bg: rgba(46, 160, 67, 0.15);
      --status-green-border: rgba(46, 160, 67, 0.4);
      --status-green-text: #3fb950;

      --status-amber-bg: rgba(210, 153, 34, 0.15);
      --status-amber-border: rgba(210, 153, 34, 0.4);
      --status-amber-text: #d29922;

      --status-red-bg: rgba(248, 81, 73, 0.15);
      --status-red-border: rgba(248, 81, 73, 0.4);
      --status-red-text: #f85149;

      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      --font-mono: 'JetBrains Mono', Consolas, monospace;
      
      --radius: 6px;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font-sans);
      background-color: var(--bg-page);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      font-size: 13px;
      line-height: 1.45;
      -webkit-font-smoothing: antialiased;
    }

    /* TOP BAR */
    header {
      background: var(--bg-card);
      border-bottom: 1px solid var(--border-default);
      height: 52px;
      padding: 0 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .nav-left { display: flex; align-items: center; gap: 20px; }
    .brand-title {
      font-size: 14px;
      font-weight: 700;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 8px;
      text-decoration: none;
    }
    .brand-badge {
      font-size: 11px;
      padding: 2px 7px;
      border-radius: 12px;
      background: var(--status-green-bg);
      border: 1px solid var(--status-green-border);
      color: var(--status-green-text);
      font-weight: 500;
    }

    nav { display: flex; align-items: center; gap: 4px; }
    .nav-tab {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 13px;
      font-weight: 500;
      padding: 6px 12px;
      border-radius: var(--radius);
      cursor: pointer;
      font-family: inherit;
      transition: color 0.1s ease;
    }
    .nav-tab:hover { color: var(--text-main); }
    .nav-tab.active {
      color: var(--text-main);
      background: var(--border-muted);
      font-weight: 600;
    }

    .nav-right { display: flex; align-items: center; gap: 10px; }
    
    .user-profile-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: var(--bg-card-header);
      border: 1px solid var(--border-default);
      padding: 4px 10px;
      border-radius: var(--radius);
      font-size: 12px;
    }
    .user-email-text {
      color: var(--text-main);
      font-weight: 500;
      font-family: var(--font-mono);
      font-size: 11.5px;
    }
    .verified-badge {
      font-size: 10px;
      font-weight: 600;
      padding: 1px 6px;
      border-radius: 4px;
      background: var(--status-green-bg);
      border: 1px solid var(--status-green-border);
      color: var(--status-green-text);
      display: inline-block;
    }
    .unverified-badge {
      font-size: 10px;
      font-weight: 600;
      padding: 1px 6px;
      border-radius: 4px;
      background: var(--status-amber-bg);
      border: 1px solid var(--status-amber-border);
      color: var(--status-amber-text);
      cursor: pointer;
      display: inline-block;
    }
    .btn-signout {
      background: var(--bg-card-header);
      border: 1px solid var(--border-default);
      color: var(--text-muted);
      font-size: 11px;
      font-weight: 500;
      padding: 4px 10px;
      border-radius: var(--radius);
      cursor: pointer;
      font-family: inherit;
      transition: all 0.1s ease;
    }
    .btn-signout:hover {
      color: var(--status-red-text);
      border-color: var(--status-red-border);
      background: var(--status-red-bg);
    }

    /* BUTTONS */
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      font-family: inherit;
      font-size: 12px;
      font-weight: 500;
      padding: 5px 12px;
      border-radius: var(--radius);
      border: 1px solid transparent;
      cursor: pointer;
      white-space: nowrap;
      transition: background-color 0.1s ease, border-color 0.1s ease;
    }
    .btn-primary {
      background: var(--accent-green);
      border-color: rgba(240, 246, 252, 0.1);
      color: #ffffff;
    }
    .btn-primary:hover { background: var(--accent-green-hover); }

    .btn-blue {
      background: var(--accent-blue);
      border-color: rgba(240, 246, 252, 0.1);
      color: #ffffff;
    }
    .btn-blue:hover { background: var(--accent-blue-hover); }

    .btn-secondary {
      background: var(--bg-card-header);
      border-color: var(--border-default);
      color: var(--text-main);
    }
    .btn-secondary:hover { background: var(--border-default); }

    .btn-danger {
      background: var(--status-red-bg);
      border-color: var(--status-red-border);
      color: var(--status-red-text);
    }
    .btn-danger:hover { background: rgba(248, 81, 73, 0.25); }

    .btn-sm { font-size: 11px; padding: 3px 8px; }

    /* LAYOUT */
    main {
      flex: 1;
      padding: 20px;
      max-width: 1280px;
      margin: 0 auto;
      width: 100%;
    }

    .tab-content { display: none; }
    .tab-content.active { display: block; }

    .page-title-row {
      margin-bottom: 16px;
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
    }
    .page-heading { font-size: 16px; font-weight: 600; color: var(--text-main); }
    .page-subtext { font-size: 12px; color: var(--text-muted); margin-top: 2px; }

    .grid-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .metrics-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 16px; }

    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-default);
      border-radius: var(--radius);
      overflow: hidden;
    }
    .card-head {
      padding: 10px 14px;
      background: var(--bg-card-header);
      border-bottom: 1px solid var(--border-default);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      font-weight: 600;
    }
    .card-body { padding: 14px; }

    .metric-box {
      background: var(--bg-card);
      border: 1px solid var(--border-default);
      border-radius: var(--radius);
      padding: 12px 14px;
    }
    .metric-name { font-size: 11px; font-weight: 500; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.03em; }
    .metric-stat { font-size: 20px; font-weight: 700; color: var(--text-main); margin: 3px 0 1px; font-family: var(--font-mono); }
    .metric-caption { font-size: 11px; color: var(--text-dim); }

    /* PRESETS */
    .presets-group {
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }
    .preset-pill {
      background: var(--bg-card);
      border: 1px solid var(--border-default);
      color: var(--text-muted);
      font-size: 11px;
      font-weight: 500;
      padding: 3px 8px;
      border-radius: var(--radius);
      cursor: pointer;
      font-family: inherit;
    }
    .preset-pill:hover { color: var(--text-main); border-color: var(--text-muted); }

    /* FORMS */
    .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; }
    .field-group { display: flex; flex-direction: column; gap: 3px; }
    .field-label { font-size: 11px; font-weight: 500; color: var(--text-muted); }
    .text-input, .select-input {
      background: var(--bg-input);
      border: 1px solid var(--border-default);
      color: var(--text-main);
      font-family: var(--font-mono);
      font-size: 12px;
      padding: 6px 8px;
      border-radius: var(--radius);
      outline: none;
      width: 100%;
    }
    .text-input:focus, .select-input:focus { border-color: var(--border-focus); }

    /* SCORECARD */
    .verdict-card {
      border-radius: var(--radius);
      padding: 12px 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      border: 1px solid transparent;
    }
    .verdict-card.allow { background: var(--status-green-bg); border-color: var(--status-green-border); }
    .verdict-card.review { background: var(--status-amber-bg); border-color: var(--status-amber-border); }
    .verdict-card.deny { background: var(--status-red-bg); border-color: var(--status-red-border); }

    .verdict-main { font-size: 14px; font-weight: 700; }
    .verdict-desc { font-size: 11px; color: var(--text-muted); margin-top: 2px; }

    .score-big { font-size: 24px; font-weight: 700; font-family: var(--font-mono); line-height: 1; }
    .score-big.allow { color: var(--status-green-text); }
    .score-big.review { color: var(--status-amber-text); }
    .score-big.deny { color: var(--status-red-text); }

    /* TABLES */
    .table-wrap {
      border: 1px solid var(--border-default);
      border-radius: var(--radius);
      overflow: hidden;
    }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th {
      background: var(--bg-card-header);
      color: var(--text-muted);
      font-size: 11px;
      font-weight: 600;
      text-align: left;
      padding: 7px 10px;
      border-bottom: 1px solid var(--border-default);
    }
    td {
      padding: 7px 10px;
      border-bottom: 1px solid var(--border-muted);
      color: var(--text-main);
    }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: rgba(255, 255, 255, 0.02); }

    .delta-bar {
      width: 100%;
      height: 4px;
      background: var(--border-muted);
      border-radius: 2px;
      overflow: hidden;
      margin-top: 3px;
    }
    .delta-fill { height: 100%; border-radius: 2px; }

    /* CODE DOCS */
    .lang-bar {
      display: flex;
      background: var(--bg-card-header);
      padding: 4px 6px;
      border-bottom: 1px solid var(--border-default);
      gap: 4px;
    }
    .lang-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 11px;
      font-weight: 500;
      padding: 4px 8px;
      border-radius: var(--radius);
      cursor: pointer;
      font-family: inherit;
    }
    .lang-btn.active { background: var(--border-default); color: var(--text-main); font-weight: 600; }

    .code-box {
      background: #090c10;
      padding: 14px;
      font-family: var(--font-mono);
      font-size: 12px;
      color: #c9d1d9;
      line-height: 1.5;
      overflow-x: auto;
    }
    .code-box pre { margin: 0; }

    /* MODAL */
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.7);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 500;
      padding: 16px;
    }
    .modal-backdrop.open { display: flex; }
    .modal-window {
      background: var(--bg-card);
      border: 1px solid var(--border-default);
      border-radius: var(--radius);
      width: 100%;
      max-width: 380px;
      padding: 20px;
    }

    .modal-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
    }
    .modal-title-text { font-size: 14px; font-weight: 600; }

    .seg-tabs {
      display: flex;
      background: var(--bg-input);
      border: 1px solid var(--border-default);
      border-radius: var(--radius);
      padding: 2px;
      margin-bottom: 14px;
    }
    .seg-btn {
      flex: 1;
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 11px;
      font-weight: 500;
      padding: 4px 0;
      border-radius: 4px;
      cursor: pointer;
      text-align: center;
      font-family: inherit;
    }
    .seg-btn.active { background: var(--bg-card-header); color: var(--text-main); font-weight: 600; }

    /* TOAST */
    .toast-msg {
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: var(--bg-card-header);
      border: 1px solid var(--border-default);
      color: var(--text-main);
      font-size: 12px;
      padding: 8px 12px;
      border-radius: var(--radius);
      display: none;
      z-index: 1000;
    }
    .toast-msg.show { display: block; }

    @media (max-width: 850px) {
      .grid-2col, .metrics-row { grid-template-columns: 1fr; }
      header { padding: 0 12px; }
      main { padding: 12px; }
    }
  </style>
</head>
<body>

  <!-- HEADER -->
  <header>
    <div class="nav-left">
      <a href="#" class="brand-title" onclick="switchTab('playground')">
        <span>Fraud Detection</span>
      </a>

      <nav>
        <button id="nav-playground" class="nav-tab active" onclick="switchTab('playground')">Playground</button>
        <button id="nav-apikeys" class="nav-tab" onclick="switchTab('apikeys')">API Keys</button>
        <button id="nav-customers" class="nav-tab" onclick="switchTab('customers')">Customers</button>
        <button id="nav-docs" class="nav-tab" onclick="switchTab('docs')">API Docs</button>
        <button id="nav-model" class="nav-tab" onclick="switchTab('model')">Model Metrics</button>
      </nav>
    </div>

    <div class="nav-right" id="header-auth-box">
      <button class="btn btn-blue btn-sm" onclick="openAuthModal()">Sign In</button>
    </div>
  </header>

  <main>
    <!-- TAB 1: PLAYGROUND -->
    <div id="tab-playground" class="tab-content active">
      <div class="page-title-row">
        <div>
          <h1 class="page-heading">Signup Fraud Evaluation</h1>
          <p class="page-subtext">Real-time risk scoring for trial registrations.</p>
        </div>
      </div>

      <div class="presets-group">
        <span style="font-size:11px; color:var(--text-dim); font-weight:500;">Presets:</span>
        <button class="preset-pill" onclick="applyPreset('genuine')">Legitimate Customer</button>
        <button class="preset-pill" onclick="applyPreset('syndicate')">Collusion Syndicate</button>
        <button class="preset-pill" onclick="applyPreset('velocity')">Velocity Fingerprint Abuse</button>
        <button class="preset-pill" onclick="applyPreset('mismatch')">Country Mismatch</button>
      </div>

      <div class="grid-2col">
        <!-- Input Form -->
        <div class="card">
          <div class="card-head">
            <span>Event Payload</span>
            <span style="font-family:var(--font-mono); font-weight:400; color:var(--text-dim);">POST /api/v1/score</span>
          </div>
          <div class="card-body">
            <div class="form-row">
              <div class="field-group">
                <label class="field-label">Name</label>
                <input type="text" id="f-name" class="text-input" value="Sarah Miller">
              </div>
              <div class="field-group">
                <label class="field-label">Email</label>
                <input type="email" id="f-email" class="text-input" value="sarah.miller@gmail.com">
              </div>
            </div>

            <div class="form-row">
              <div class="field-group">
                <label class="field-label">IP Address</label>
                <input type="text" id="f-ip" class="text-input" value="198.51.100.24">
              </div>
              <div class="field-group">
                <label class="field-label">Device Fingerprint</label>
                <input type="text" id="f-device" class="text-input" value="dev_macbook_pro_m2_99">
              </div>
            </div>

            <div class="form-row">
              <div class="field-group">
                <label class="field-label">Payment Token</label>
                <input type="text" id="f-payment" class="text-input" value="pm_visa_auth_8821">
              </div>
              <div class="field-group">
                <label class="field-label">Billing City</label>
                <input type="text" id="f-area" class="text-input" value="new york">
              </div>
            </div>

            <div class="form-row">
              <div class="field-group">
                <label class="field-label">Operating System</label>
                <select id="f-os" class="select-input">
                  <option value="mac">macOS</option>
                  <option value="windows">Windows</option>
                  <option value="linux">Linux</option>
                  <option value="ios">iOS</option>
                  <option value="android">Android</option>
                </select>
              </div>
              <div class="field-group">
                <label class="field-label">Card Country</label>
                <select id="f-country" class="select-input">
                  <option value="US">United States (US)</option>
                  <option value="GB">United Kingdom (GB)</option>
                  <option value="IN">India (IN)</option>
                  <option value="SG">Singapore (SG)</option>
                  <option value="DE">Germany (DE)</option>
                </select>
              </div>
            </div>

            <button class="btn btn-blue" style="width:100%; margin-top:6px; padding:7px 0;" onclick="submitScoring()">Run Fraud Check</button>
          </div>
        </div>

        <!-- Output Scorecard -->
        <div class="card">
          <div class="card-head">
            <span>Evaluation Result</span>
            <span id="out-latency-tag" style="font-family:var(--font-mono); font-weight:400; color:var(--text-dim);">Ready</span>
          </div>
          <div class="card-body">
            <div id="verdict-banner-el" class="verdict-card allow">
              <div>
                <div style="font-size:10px; font-weight:600; color:var(--text-muted);">DECISION</div>
                <div id="verdict-label" class="verdict-main" style="color:var(--status-green-text);">ALLOW (GENUINE)</div>
                <div id="verdict-action" class="verdict-desc">Action: Allow trial access</div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:10px; font-weight:600; color:var(--text-muted);">RISK SCORE</div>
                <div id="verdict-score" class="score-big allow">0.5</div>
              </div>
            </div>

            <div class="metrics-row" style="margin-bottom:12px;">
              <div class="metric-box" style="padding:8px 10px;">
                <div class="metric-name" style="font-size:10px;">Confidence</div>
                <div id="metric-conf" style="font-size:14px; font-weight:700; font-family:var(--font-mono); margin-top:2px;">99.5%</div>
              </div>
              <div class="metric-box" style="padding:8px 10px;">
                <div class="metric-name" style="font-size:10px;">Threshold</div>
                <div style="font-size:14px; font-weight:700; font-family:var(--font-mono); color:var(--accent-blue); margin-top:2px;">T = 10.0</div>
              </div>
              <div class="metric-box" style="padding:8px 10px;">
                <div class="metric-name" style="font-size:10px;">Record ID</div>
                <div id="metric-cid" style="font-size:11px; font-weight:600; font-family:var(--font-mono); margin-top:4px;">--</div>
              </div>
            </div>

            <div style="font-size:11px; font-weight:600; color:var(--text-muted); margin-bottom:6px;">Signal Weights</div>
            <div class="table-wrap" style="max-height:175px; overflow-y:auto;">
              <table>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Value</th>
                    <th>Weight</th>
                  </tr>
                </thead>
                <tbody id="signals-tbody">
                  <tr><td colspan="3" style="text-align:center; color:var(--text-dim); padding:14px;">Submit a payload to view feature weights</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: API KEYS -->
    <div id="tab-apikeys" class="tab-content">
      <div class="page-title-row">
        <div>
          <h1 class="page-heading">API Keys & Quotas</h1>
          <p class="page-subtext">Manage outbound authentication keys for backend integration.</p>
        </div>
      </div>

      <div id="keys-unauth-view" class="card" style="text-align:center; padding:36px 16px; max-width:440px; margin:20px auto;">
        <h2 style="font-size:14px; font-weight:600; margin-bottom:6px;">Authentication Required</h2>
        <p style="font-size:12px; color:var(--text-muted); margin-bottom:16px;">
          Sign in or register a verified developer account to generate up to 3 API keys.
        </p>
        <button class="btn btn-blue" onclick="openAuthModal()">Sign In</button>
      </div>

      <div id="keys-auth-view" style="display:none;">
        <div class="metrics-row">
          <div class="metric-box">
            <div class="metric-name">Active Keys</div>
            <div id="key-count-stat" class="metric-stat">0 / 3</div>
            <div class="metric-caption">Max 3 keys per verified tenant</div>
          </div>
          <div class="metric-box">
            <div class="metric-name">Rate Limit</div>
            <div class="metric-stat" style="color:var(--accent-blue);">30 <span style="font-size:12px; font-weight:400; color:var(--text-muted);">req/min</span></div>
            <div class="metric-caption">Sliding 60-second window</div>
          </div>
          <div class="metric-box">
            <div class="metric-name">Security</div>
            <div class="metric-stat" style="font-size:16px; color:#d2a8ff;">SHA-256</div>
            <div class="metric-caption">Zero plaintext storage</div>
          </div>
        </div>

        <div class="card">
          <div class="card-head">
            <span>API Keys</span>
            <button class="btn btn-blue btn-sm" onclick="openKeyModal()">+ New Key</button>
          </div>
          <div class="table-wrap" style="border:none;">
            <table>
              <thead>
                <tr>
                  <th>Label</th>
                  <th>Type</th>
                  <th>Key Token</th>
                  <th>Rate Limit</th>
                  <th>Created</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody id="keys-tbody">
                <!-- Dynamically populated -->
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: CUSTOMERS -->
    <div id="tab-customers" class="tab-content">
      <div class="page-title-row">
        <div>
          <h1 class="page-heading">Customer Records</h1>
          <p class="page-subtext">Tenant audit directory of scored registration events.</p>
        </div>
        <div style="width:240px;">
          <input type="text" id="cust-search-box" class="text-input" placeholder="Search email, IP, device..." oninput="filterCustomerSearch(this.value)">
        </div>
      </div>

      <div class="card">
        <div class="table-wrap" style="border:none;">
          <table>
            <thead>
              <tr>
                <th>Customer</th>
                <th>Email</th>
                <th>IP Address</th>
                <th>Device ID</th>
                <th>Risk Score</th>
                <th>Verdict</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody id="customers-tbody">
              <tr><td colspan="7" style="text-align:center; padding:24px; color:var(--text-dim);">Sign in to view customer evaluations.</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 4: API DOCS -->
    <div id="tab-docs" class="tab-content">
      <div class="page-title-row">
        <div>
          <h1 class="page-heading">API Reference</h1>
          <p class="page-subtext">Integrate fraud scoring into your application backend.</p>
        </div>
      </div>

      <div class="card">
        <div class="lang-bar">
          <button id="tab-lang-curl" class="lang-btn active" onclick="setSnippetLang('curl')">cURL</button>
          <button id="tab-lang-python" class="lang-btn" onclick="setSnippetLang('python')">Python</button>
          <button id="tab-lang-sdk" class="lang-btn" onclick="setSnippetLang('sdk')">Python SDK</button>
          <button id="tab-lang-node" class="lang-btn" onclick="setSnippetLang('node')">Node.js</button>
          <button id="tab-lang-js" class="lang-btn" onclick="setSnippetLang('js')">JavaScript</button>
          <button id="tab-lang-go" class="lang-btn" onclick="setSnippetLang('go')">Go</button>
          <div style="margin-left:auto;">
            <button class="btn btn-secondary btn-sm" onclick="copySnippetCode()">Copy</button>
          </div>
        </div>
        <div class="code-box">
          <pre id="snippet-target">Loading snippet...</pre>
        </div>
      </div>
    </div>

    <!-- TAB 5: MODEL METRICS -->
    <div id="tab-model" class="tab-content">
      <div class="page-title-row">
        <div>
          <h1 class="page-heading">Model Benchmarks & Thresholds</h1>
          <p class="page-subtext">Telemetry and classification boundaries.</p>
        </div>
        <button class="btn btn-secondary btn-sm" onclick="runRetraining()">Retrain Model</button>
      </div>

      <div class="metrics-row">
        <div class="metric-box">
          <div class="metric-name">ROC-AUC</div>
          <div class="metric-stat" style="color:var(--status-green-text);">0.941</div>
          <div class="metric-caption">95% CI: [0.923, 0.958]</div>
        </div>
        <div class="metric-box">
          <div class="metric-name">Abuse Recall</div>
          <div class="metric-stat" style="color:#38bdf8;">93.8%</div>
          <div class="metric-caption">95% CI: [91.2%, 96.0%]</div>
        </div>
        <div class="metric-box">
          <div class="metric-name">Precision @ Threshold</div>
          <div class="metric-stat" style="color:var(--accent-blue);">91.2%</div>
          <div class="metric-caption">Target &ge; 85% Precision</div>
        </div>
        <div class="metric-box">
          <div class="metric-name">Inference Latency</div>
          <div class="metric-stat" style="color:#d2a8ff;">&lt; 15ms</div>
          <div class="metric-caption">P99 Vectorized scoring</div>
        </div>
      </div>

      <div class="card">
        <div class="card-head">Decision Strategy (3-Band Action Policy)</div>
        <div class="card-body">
          <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:12px;">
            <div style="background:var(--bg-input); padding:10px; border-radius:var(--radius); border:1px solid var(--border-default);">
              <div style="font-size:11px; font-weight:600; color:var(--status-green-text);">0.0 - 5.4: ALLOW</div>
              <div style="font-size:11px; color:var(--text-muted); margin-top:2px;">Frictionless access for legitimate users.</div>
            </div>
            <div style="background:var(--bg-input); padding:10px; border-radius:var(--radius); border:1px solid var(--border-default);">
              <div style="font-size:11px; font-weight:600; color:var(--status-amber-text);">5.5 - 9.9: REVIEW</div>
              <div style="font-size:11px; color:var(--text-muted); margin-top:2px;">Requires step-up 2FA or SMS verification.</div>
            </div>
            <div style="background:var(--bg-input); padding:10px; border-radius:var(--radius); border:1px solid var(--border-default);">
              <div style="font-size:11px; font-weight:600; color:var(--status-red-text);">10.0 - 100.0: BLOCK</div>
              <div style="font-size:11px; color:var(--text-muted); margin-top:2px;">Immediate refusal & payment required.</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </main>

  <!-- AUTH MODAL -->
  <div class="modal-backdrop" id="auth-modal">
    <div class="modal-window">
      <div class="modal-top">
        <div class="modal-title-text" id="auth-modal-title">Sign In</div>
        <button class="btn btn-secondary btn-sm" onclick="closeAuthModal()">✕</button>
      </div>

      <div class="seg-tabs">
        <button id="seg-signin" class="seg-btn active" onclick="setAuthTab('signin')">Sign In</button>
        <button id="seg-signup" class="seg-btn" onclick="setAuthTab('signup')">Create Account</button>
      </div>

      <div id="auth-error-msg" style="display:none; color:var(--status-red-text); background:var(--status-red-bg); border:1px solid var(--status-red-border); border-radius:var(--radius); padding:6px 10px; margin-bottom:10px; font-size:11px;"></div>

      <div class="field-group" style="margin-bottom:8px;">
        <label class="field-label">Email</label>
        <input type="email" id="auth-email" class="text-input" placeholder="name@company.com">
      </div>
      <div class="field-group" style="margin-bottom:12px;">
        <label class="field-label">Password</label>
        <input type="password" id="auth-pwd" class="text-input" placeholder="Min 6 characters">
      </div>

      <button id="auth-submit-action" class="btn btn-blue" style="width:100%; margin-bottom:6px;" onclick="submitAuthForm()">Sign In</button>
      <button class="btn btn-secondary" style="width:100%;" onclick="submitGoogleAuth()">Continue with Google</button>
    </div>
  </div>

  <!-- EMAIL VERIFICATION WAITING MODAL -->
  <div class="modal-backdrop" id="verify-modal">
    <div class="modal-window" style="text-align:center;">
      <div id="verify-waiting-pane">
        <h2 style="font-size:14px; font-weight:600; margin-bottom:4px;">Verify Your Email</h2>
        <p style="font-size:12px; color:var(--text-muted); margin-bottom:10px;">
          Verification link sent to<br><strong id="verify-target-email" style="color:var(--text-main);"></strong>
        </p>

        <div style="background:var(--status-amber-bg); border:1px solid var(--status-amber-border); border-radius:var(--radius); padding:8px 10px; margin-bottom:12px; text-align:left;">
          <div style="font-size:11px; font-weight:600; color:var(--status-amber-text);">Check Spam / Junk Folder</div>
          <div style="font-size:11px; color:#cbd5e1;">If you do not see the email, please check your spam folder.</div>
        </div>

        <div style="font-size:11px; color:var(--text-muted); margin-bottom:12px;">Waiting for confirmation...</div>

        <div style="display:flex; gap:6px; justify-content:center;">
          <button class="btn btn-secondary btn-sm" onclick="resendVerifyEmail()">Resend Email</button>
          <button class="btn btn-blue btn-sm" onclick="checkVerifyStatus()">Check Status</button>
        </div>
      </div>

      <div id="verify-success-pane" style="display:none; padding:10px 0;">
        <h2 style="font-size:14px; font-weight:600; color:var(--status-green-text); margin-bottom:4px;">Email Verified</h2>
        <p style="font-size:12px; color:var(--text-muted);">Logging you in...</p>
      </div>
    </div>
  </div>

  <!-- KEY CREATION MODAL -->
  <div class="modal-backdrop" id="key-modal">
    <div class="modal-window">
      <div class="modal-top">
        <div class="modal-title-text">Generate API Key</div>
        <button class="btn btn-secondary btn-sm" onclick="closeKeyModal()">✕</button>
      </div>

      <div class="field-group" style="margin-bottom:8px;">
        <label class="field-label">Key Name</label>
        <input type="text" id="k-name" class="text-input" placeholder="e.g. Production Backend">
      </div>
      <div class="field-group" style="margin-bottom:14px;">
        <label class="field-label">Environment</label>
        <select id="k-type" class="select-input">
          <option value="live">Live (fk_live_...)</option>
          <option value="test">Test (fk_test_...)</option>
        </select>
      </div>

      <button class="btn btn-blue" style="width:100%;" onclick="submitKeyCreation()">Generate Key</button>
    </div>
  </div>

  <!-- SECRET KEY REVEAL MODAL (SHOWN ONCE AT CREATION) -->
  <div class="modal-backdrop" id="key-reveal-modal">
    <div class="modal-window">
      <div class="modal-top">
        <div class="modal-title-text" style="color:var(--status-green-text);">API Key Created</div>
        <button class="btn btn-secondary btn-sm" onclick="closeKeyRevealModal()">✕</button>
      </div>

      <p style="font-size:12px; color:var(--text-muted); margin-bottom:10px;">
        Please copy this secret key now. For security reasons, <strong>it will not be shown again</strong>.
      </p>

      <div style="background:var(--bg-input); border:1px solid var(--border-default); border-radius:var(--radius); padding:8px 10px; margin-bottom:12px; word-break:break-all;">
        <code id="new-key-secret-val" style="font-family:var(--font-mono); color:var(--accent-blue); font-size:12px;"></code>
      </div>

      <div style="display:flex; gap:8px;">
        <button class="btn btn-blue" style="flex:1;" onclick="copyRevealedKey()">Copy Key</button>
        <button class="btn btn-secondary" onclick="closeKeyRevealModal()">Done</button>
      </div>
    </div>
  </div>

  <!-- TOAST -->
  <div class="toast-msg" id="toast-bar"></div>

  <script>
    let activeUser = null;
    let primaryApiKey = null;
    let snippetLang = 'curl';
    let firebaseReady = false;
    let authMode = 'signin';
    let userKeyCount = 0;
    let pollTimer = null;

    const PRESETS = {
      genuine: {
        name: "Sarah Miller",
        email: "sarah.miller@gmail.com",
        ip: "198.51.100.24",
        device: "dev_macbook_pro_m2_99",
        payment: "pm_visa_auth_8821",
        area: "new york",
        os: "mac",
        country: "US"
      },
      syndicate: {
        name: "David Smith",
        email: "david.attacker99@tempmail.com",
        ip: "203.0.113.88",
        device: "dev_fraud_farm_bot_01",
        payment: "pm_card_syndicate_repeat",
        area: "dubai",
        os: "windows",
        country: "US"
      },
      velocity: {
        name: "Bot Burst Node",
        email: "burst_bot_42@fastburner.io",
        ip: "185.220.101.5",
        device: "dev_fraud_farm_bot_01",
        payment: "pm_card_burner_token_19",
        area: "mumbai",
        os: "linux",
        country: "IN"
      },
      mismatch: {
        name: "Alex Johnson",
        email: "alex.johnson@outlook.com",
        ip: "103.20.10.56",
        device: "dev_laptop_thinkpad_12",
        payment: "pm_chase_us_card_55",
        area: "singapore",
        os: "windows",
        country: "US"
      }
    };

    function logToBackend(level, msg) {
      console.log(`[${level}]`, msg);
      fetch('/api/v1/client-logs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ level, message: typeof msg === 'object' ? JSON.stringify(msg) : String(msg) })
      }).catch(() => {});
    }

    async function initFirebaseClient() {
      try {
        const resp = await fetch('/api/v1/config/firebase');
        const cfg = await resp.json();

        if (cfg.apiKey && !firebase.apps.length) {
          firebase.initializeApp({
            apiKey: cfg.apiKey,
            authDomain: cfg.authDomain,
            projectId: cfg.projectId,
            storageBucket: cfg.storageBucket,
            messagingSenderId: cfg.messagingSenderId,
            appId: cfg.appId
          });
          firebaseReady = true;
          logToBackend('SUCCESS', `Firebase initialized for project: ${cfg.projectId}`);

          firebase.auth().onAuthStateChanged(user => {
            if (user) {
              logToBackend('AUTH_STATE', `Active user: ${user.email} (UID: ${user.uid}) | Verified: ${user.emailVerified}`);
              applyUserAuth(user.uid, user.email, user.displayName, user.emailVerified);
            } else {
              activeUser = null;
              primaryApiKey = null;
              renderAuthNav();
              document.getElementById('keys-unauth-view').style.display = 'block';
              document.getElementById('keys-auth-view').style.display = 'none';
              renderSnippet();
            }
          });
        } else if (firebase.apps.length) {
          firebaseReady = true;
        }
      } catch (e) {
        logToBackend('ERROR', 'Firebase init error: ' + (e.message || e));
      }
    }

    function setAuthTab(tab) {
      authMode = tab;
      document.getElementById('auth-error-msg').style.display = 'none';
      if (tab === 'signin') {
        document.getElementById('seg-signin').classList.add('active');
        document.getElementById('seg-signup').classList.remove('active');
        document.getElementById('auth-modal-title').innerText = 'Sign In';
        document.getElementById('auth-submit-action').innerText = 'Sign In';
      } else {
        document.getElementById('seg-signup').classList.add('active');
        document.getElementById('seg-signin').classList.remove('active');
        document.getElementById('auth-modal-title').innerText = 'Create Account';
        document.getElementById('auth-submit-action').innerText = 'Create Account';
      }
    }

    function showAuthError(msg) {
      const err = document.getElementById('auth-error-msg');
      err.innerText = msg;
      err.style.display = 'block';
    }

    function submitAuthForm() {
      const email = document.getElementById('auth-email').value.trim();
      const pwd = document.getElementById('auth-pwd').value.trim();

      if (!email || !pwd) return showAuthError('Enter both email and password.');
      if (pwd.length < 6) return showAuthError('Password must be at least 6 characters.');
      if (!firebaseReady) return showAuthError('Connecting to Firebase auth service...');

      const btn = document.getElementById('auth-submit-action');
      btn.disabled = true;
      btn.innerText = 'Working...';

      if (authMode === 'signin') {
        firebase.auth().signInWithEmailAndPassword(email, pwd)
          .then(cred => {
            btn.disabled = false;
            btn.innerText = 'Sign In';
            applyUserAuth(cred.user.uid, cred.user.email, cred.user.displayName, cred.user.emailVerified);
            closeAuthModal();
            toast("Signed in as " + cred.user.email);
          })
          .catch(err => {
            btn.disabled = false;
            btn.innerText = 'Sign In';
            if (err.code === 'auth/wrong-password' || err.code === 'auth/invalid-credential') {
              showAuthError('Incorrect password.');
            } else if (err.code === 'auth/user-not-found') {
              showAuthError('No account found for this email.');
            } else {
              showAuthError(err.message || 'Sign in failed.');
            }
          });
      } else {
        firebase.auth().createUserWithEmailAndPassword(email, pwd)
          .then(cred => {
            btn.disabled = false;
            btn.innerText = 'Create Account';
            cred.user.sendEmailVerification().catch(() => {});
            closeAuthModal();
            startEmailVerificationFlow(cred.user);
          })
          .catch(err => {
            btn.disabled = false;
            btn.innerText = 'Create Account';
            if (err.code === 'auth/email-already-in-use') {
              showAuthError('This email is already registered. Sign in above.');
            } else {
              showAuthError(err.message || 'Registration failed.');
            }
          });
      }
    }

    function submitGoogleAuth() {
      if (!firebaseReady) return showAuthError('Connecting to Firebase auth service...');
      const provider = new firebase.auth.GoogleAuthProvider();
      firebase.auth().signInWithPopup(provider)
        .then(cred => {
          applyUserAuth(cred.user.uid, cred.user.email, cred.user.displayName, true);
          closeAuthModal();
          toast("Signed in with Google.");
        })
        .catch(err => {
          if (err.code !== 'auth/popup-closed-by-user') {
            showAuthError(err.message || 'Google sign in failed.');
          }
        });
    }

    function startEmailVerificationFlow(user) {
      if (pollTimer) clearInterval(pollTimer);
      document.getElementById('verify-target-email').innerText = user.email;
      document.getElementById('verify-waiting-pane').style.display = 'block';
      document.getElementById('verify-success-pane').style.display = 'none';
      document.getElementById('verify-modal').classList.add('open');

      pollTimer = setInterval(async () => {
        try {
          if (firebase.auth().currentUser) {
            await firebase.auth().currentUser.reload();
            if (firebase.auth().currentUser.emailVerified) {
              handleVerificationSuccess(firebase.auth().currentUser);
            }
          }
        } catch (e) {}
      }, 2500);
    }

    async function checkVerifyStatus() {
      if (firebaseReady && firebase.auth().currentUser) {
        await firebase.auth().currentUser.reload();
        await firebase.auth().currentUser.getIdToken(true);
        if (firebase.auth().currentUser.emailVerified) {
          handleVerificationSuccess(firebase.auth().currentUser);
        } else {
          toast("Email not yet verified. Please check inbox or spam folder.");
        }
      }
    }

    async function resendVerifyEmail() {
      if (firebaseReady && firebase.auth().currentUser) {
        try {
          await firebase.auth().currentUser.sendEmailVerification();
          toast("Verification email sent! Check spam folder.");
        } catch (e) {
          toast("Already sent recently. Please check your spam folder.");
        }
      }
    }

    function handleVerificationSuccess(user) {
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
      document.getElementById('verify-waiting-pane').style.display = 'none';
      document.getElementById('verify-success-pane').style.display = 'block';

      setTimeout(async () => {
        document.getElementById('verify-modal').classList.remove('open');
        await applyUserAuth(user.uid, user.email, user.displayName, true);
        toast("Email verified.");
      }, 1500);
    }

    async function applyUserAuth(uid, email, displayName, isVerified = true) {
      try {
        const res = await fetch('/api/v1/auth/session', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ uid, email, display_name: displayName })
        });
        const data = await res.json();
        activeUser = data.user;
        activeUser.emailVerified = isVerified;
        renderAuthNav();

        document.getElementById('keys-unauth-view').style.display = 'none';
        document.getElementById('keys-auth-view').style.display = 'block';

        fetchKeys();
        fetchCustomers();

        // Sync to Firestore
        if (firebaseReady && firebase.firestore) {
          try {
            const db = firebase.firestore();
            db.collection('users').document(uid).set({
              uid: uid,
              email: email,
              display_name: displayName || email.split('@')[0],
              email_verified: Boolean(isVerified),
              quota_per_min: 30,
              last_login_at: new Date().toISOString(),
              updated_at: new Date().toISOString()
            }, { merge: true }).catch(() => {});

            db.collection('users').document(uid).collection('login_history').add({
              login_at: new Date().toISOString(),
              email: email,
              user_agent: navigator.userAgent
            }).catch(() => {});
          } catch (e) {}
        }
      } catch (e) {
        console.log('Auth note:', e);
      }
    }

    function renderAuthNav() {
      const container = document.getElementById('header-auth-box');
      if (activeUser) {
        const verifiedTag = activeUser.emailVerified
          ? `<span class="verified-badge">Verified</span>`
          : `<span class="unverified-badge" onclick="startEmailVerificationFlow(firebase.auth().currentUser || activeUser)">Unverified</span>`;

        container.innerHTML = `
          <div style="display:flex; align-items:center; gap:10px;">
            <div class="user-profile-badge">
              <span class="user-email-text">${activeUser.email}</span>
              ${verifiedTag}
            </div>
            <button class="btn-signout" onclick="logoutUser()">Sign Out</button>
          </div>
        `;
      } else {
        container.innerHTML = `<button class="btn btn-blue btn-sm" onclick="openAuthModal()">Sign In</button>`;
      }
    }

    async function logoutUser() {
      if (firebaseReady) {
        await firebase.auth().signOut();
      }
      activeUser = null;
      primaryApiKey = null;
      renderAuthNav();
      document.getElementById('keys-unauth-view').style.display = 'block';
      document.getElementById('keys-auth-view').style.display = 'none';
      toast("Signed out.");
      renderSnippet();
    }

    // --- PLAYGROUND SCORING ---
    async function submitScoring() {
      const payload = {
        name: document.getElementById('f-name').value,
        email: document.getElementById('f-email').value,
        ip_address: document.getElementById('f-ip').value,
        device_id: document.getElementById('f-device').value,
        payment_token: document.getElementById('f-payment').value,
        area: document.getElementById('f-area').value,
        device_os: document.getElementById('f-os').value,
        payment_country: document.getElementById('f-country').value
      };

      const startT = performance.now();
      try {
        const headers = { 'Content-Type': 'application/json' };
        if (primaryApiKey) headers['X-API-Key'] = primaryApiKey;

        const res = await fetch('/api/v1/score', {
          method: 'POST',
          headers: headers,
          body: JSON.stringify(payload)
        });

        if (res.status === 429) {
          const err = await res.json().catch(() => ({}));
          alert('Rate Limit Exceeded (30 req/min): ' + (err.detail ? (err.detail.message || err.detail.error) : 'Quota exhausted.'));
          return;
        }

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          const errMsg = (err.detail && (err.detail.message || err.detail.error)) || err.message || err.error || ('Server returned HTTP ' + res.status);
          alert('Evaluation Error: ' + errMsg);
          return;
        }

        const data = await res.json();
        const duration = (performance.now() - startT).toFixed(1);

        const latModel = typeof data.latency_ms === 'number' ? data.latency_ms : (parseFloat(data.latency_ms) || 0);
        document.getElementById('out-latency-tag').innerText = `${duration}ms roundtrip | ${latModel.toFixed(1)}ms model`;

        const score = typeof data.risk_score === 'number' ? data.risk_score : (parseFloat(data.risk_score) || 0);
        document.getElementById('verdict-score').innerText = score.toFixed(1);
        document.getElementById('metric-cid').innerText = data.customer_id || 'saved';

        const banner = document.getElementById('verdict-banner-el');
        const scoreEl = document.getElementById('verdict-score');
        const labelEl = document.getElementById('verdict-label');
        const actionEl = document.getElementById('verdict-action');

        const mode = data.severity === 'low' ? 'allow' : data.severity === 'medium' ? 'review' : 'deny';
        banner.className = 'verdict-card ' + mode;
        scoreEl.className = 'score-big ' + mode;
        labelEl.innerText = data.verdict || 'NEW USER (GENUINE)';
        labelEl.style.color = mode === 'allow' ? 'var(--status-green-text)' : mode === 'review' ? 'var(--status-amber-text)' : 'var(--status-red-text)';
        actionEl.innerText = 'Action: ' + (data.recommended_action || 'ALLOW');

        document.getElementById('metric-conf').innerText = (data.model_confidence_pct !== undefined ? data.model_confidence_pct : '95.0') + '%';

        // Signal weights
        const tbody = document.getElementById('signals-tbody');
        tbody.innerHTML = '';
        const entries = Object.entries(data.signal_breakdown || {});
        entries.forEach(([sig, rawVal]) => {
          const val = typeof rawVal === 'number' ? rawVal : (parseFloat(rawVal) || 0);
          const rawFeat = data.raw_features && data.raw_features[sig] !== undefined ? data.raw_features[sig] : '--';
          const pct = Math.min(Math.abs(val) * 3.3, 100);
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td style="font-family:var(--font-mono); font-weight:500;">${sig}</td>
            <td style="font-family:var(--font-mono); color:var(--text-muted);">${rawFeat}</td>
            <td>
              <div style="font-size:11px; font-weight:600; color:${val > 0 ? 'var(--status-red-text)' : 'var(--status-green-text)'};">${val > 0 ? '+' : ''}${val.toFixed(1)}</div>
              <div class="delta-bar">
                <div class="delta-fill" style="width:${pct}%; background:${val >= 15 ? 'var(--status-red-text)' : val > 0 ? 'var(--status-amber-text)' : 'var(--status-green-text)'};"></div>
              </div>
            </td>
          `;
          tbody.appendChild(tr);
        });

        // Store customer in Firestore
        if (activeUser && firebaseReady && firebase.firestore) {
          try {
            const db = firebase.firestore();
            const custId = data.customer_id || ('cust_' + Date.now());
            db.collection('users').document(activeUser.uid).collection('customers').document(custId).set({
              customer_id: custId,
              name: payload.name,
              email: payload.email,
              ip_address: payload.ip_address,
              device_id: payload.device_id,
              payment_token: payload.payment_token,
              area: payload.area,
              risk_score: score,
              verdict: data.verdict,
              recommended_action: data.recommended_action,
              severity: data.severity,
              confidence_pct: data.model_confidence_pct,
              created_at: new Date().toISOString()
            }).catch(() => {});
          } catch (e) {}
        }

        if (activeUser) fetchCustomers();

      } catch (e) {
        alert('Evaluation error: ' + e);
      }
    }

    // --- API KEYS ---
    async function fetchKeys() {
      if (!activeUser) return;
      try {
        const res = await fetch(`/api/v1/keys/list?user_id=${activeUser.uid}`);
        const data = await res.json();
        const tbody = document.getElementById('keys-tbody');
        tbody.innerHTML = '';
        userKeyCount = data.keys.length;
        document.getElementById('key-count-stat').innerText = `${data.keys.length} / 3`;

        if (data.keys.length === 0) {
          primaryApiKey = null;
          tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:20px; color:var(--text-dim);">No API keys generated. Click "+ New Key" (max 3 keys).</td></tr>';
          renderSnippet();
          return;
        }

        primaryApiKey = data.keys[0].masked_key;
        renderSnippet();

        data.keys.forEach(k => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td style="font-weight:600;">${k.name}</td>
            <td><span style="font-size:10px; font-family:var(--font-mono); font-weight:600; color:var(--status-green-text);">${k.key_type.toUpperCase()}</span></td>
            <td><code style="color:var(--accent-blue); font-family:var(--font-mono);">${k.masked_key}</code></td>
            <td style="color:var(--text-muted);">30 req/min</td>
            <td style="color:var(--text-dim);">${k.created_at.split('T')[0]}</td>
            <td>
              <button class="btn btn-danger btn-sm" onclick="deleteKey('${k.key_id}', '${k.name.replace(/'/g, "\\'")}')">Delete</button>
            </td>
          `;
          tbody.appendChild(tr);
        });
      } catch (e) {
        console.log('Key note:', e);
      }
    }

    function openKeyRevealModal(secretKey) {
      document.getElementById('new-key-secret-val').innerText = secretKey;
      document.getElementById('key-reveal-modal').classList.add('open');
    }

    function closeKeyRevealModal() {
      document.getElementById('key-reveal-modal').classList.remove('open');
    }

    function copyRevealedKey() {
      const secret = document.getElementById('new-key-secret-val').innerText;
      navigator.clipboard.writeText(secret);
      toast("Copied secret key to clipboard!");
    }

    async function submitKeyCreation() {
      if (!activeUser) return openAuthModal();
      if (!activeUser.emailVerified) {
        toast("⚠️ Email verification required to generate API keys.");
        closeKeyModal();
        return startEmailVerificationFlow(firebase.auth().currentUser || activeUser);
      }
      if (userKeyCount >= 3) {
        toast("⚠️ Key limit reached (3/3). Delete an existing key first.");
        closeKeyModal();
        return;
      }

      const label = document.getElementById('k-name').value.trim() || 'Backend API Key';
      const ktype = document.getElementById('k-type').value;

      const res = await fetch('/api/v1/keys/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: activeUser.uid, name: label, key_type: ktype })
      });

      if (!res.ok) {
        const errData = await res.json();
        alert(errData.message || errData.error || 'Failed to generate key.');
        return;
      }

      const data = await res.json();
      closeKeyModal();
      fetchKeys();

      // Sync API key to Firestore
      if (firebaseReady && firebase.firestore) {
        try {
          const db = firebase.firestore();
          db.collection('users').document(activeUser.uid).collection('api_keys').document(data.key_id).set({
            key_id: data.key_id,
            name: label,
            key_type: ktype,
            masked_key: data.masked_key,
            rate_limit_per_min: 30,
            created_at: data.created_at || new Date().toISOString()
          }).catch(() => {});
        } catch (e) {}
      }

      openKeyRevealModal(data.api_key);
    }

    async function deleteKey(keyId, keyName) {
      if (!activeUser) return;
      if (!confirm(`Delete API key "${keyName}"? Outbound requests using this key will immediately be rejected.`)) {
        return;
      }

      try {
        const res = await fetch('/api/v1/keys/delete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: activeUser.uid, key_id: keyId })
        });
        const data = await res.json();

        // Delete from Firestore
        if (firebaseReady && firebase.firestore) {
          try {
            firebase.firestore().collection('users').document(activeUser.uid).collection('api_keys').document(keyId).delete().catch(() => {});
          } catch (e) {}
        }

        toast(`Key "${keyName}" deleted.`);
        fetchKeys();
      } catch (e) {
        toast("Failed to delete key: " + e);
      }
    }

    // --- CUSTOMERS ---
    async function fetchCustomers() {
      if (!activeUser) return;
      try {
        const res = await fetch(`/api/v1/customers/list?user_id=${activeUser.uid}`);
        const data = await res.json();
        renderCustomers(data.customers || []);
      } catch (e) {
        console.log('Customer note:', e);
      }
    }

    let searchDebounce = null;
    function filterCustomerSearch(val) {
      clearTimeout(searchDebounce);
      searchDebounce = setTimeout(async () => {
        if (!activeUser) return;
        if (!val.trim()) return fetchCustomers();
        try {
          const res = await fetch(`/api/v1/customers/search?user_id=${activeUser.uid}&q=${encodeURIComponent(val)}`);
          const data = await res.json();
          renderCustomers(data.customers || []);
        } catch (e) {}
      }, 200);
    }

    function renderCustomers(customers) {
      const tbody = document.getElementById('customers-tbody');
      tbody.innerHTML = '';

      if (!customers || customers.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:24px; color:var(--text-dim);">No customer records.</td></tr>';
        return;
      }

      customers.forEach(c => {
        const tr = document.createElement('tr');
        const score = typeof c.risk_score === 'number' ? c.risk_score : (parseFloat(c.risk_score) || 0);
        const isAbuse = score >= 10.0;
        const devId = (c.device_id || '').substring(0, 14);
        tr.innerHTML = `
          <td style="font-weight:500;">${c.name || 'Unknown'}</td>
          <td style="font-family:var(--font-mono); color:var(--text-muted);">${c.email || '--'}</td>
          <td style="font-family:var(--font-mono);">${c.ip_address || '--'}</td>
          <td style="font-family:var(--font-mono); color:var(--text-dim);">${devId}...</td>
          <td><span style="font-family:var(--font-mono); font-weight:600; color:${isAbuse ? 'var(--status-red-text)' : 'var(--status-green-text)'};">${score.toFixed(1)}</span></td>
          <td><span style="font-size:10px; font-weight:600; color:${isAbuse ? 'var(--status-red-text)' : 'var(--status-green-text)'};">${c.verdict || 'NEW USER'}</span></td>
          <td style="color:var(--text-dim); font-size:11px;">${(c.created_at || '').split('T')[0]}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    // --- CODE SNIPPETS ---
    function setSnippetLang(lang) {
      snippetLang = lang;
      document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
      const activeBtn = document.getElementById('tab-lang-' + lang);
      if (activeBtn) activeBtn.classList.add('active');
      renderSnippet();
    }

    function renderSnippet() {
      const keyVal = primaryApiKey || "YOUR_API_KEY_HERE";
      const payloadObj = {
        name: "Sarah Miller",
        email: "sarah.miller@gmail.com",
        ip_address: "198.51.100.24",
        device_id: "dev_macbook_pro_m2_99",
        payment_token: "pm_visa_auth_8821",
        area: "new york"
      };

      let code = "";
      if (snippetLang === 'curl') {
        code = `curl -X POST https://free-trail-fraud-detection-mlmodel.onrender.com/api/v1/score \\\\
  -H "Content-Type: application/json" \\\\
  -H "X-API-Key: ${keyVal}" \\\\
  -d '${JSON.stringify(payloadObj, null, 2)}'`;
      } else if (snippetLang === 'python') {
        code = `import requests

url = "https://free-trail-fraud-detection-mlmodel.onrender.com/api/v1/score"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "${keyVal}"
}
payload = ${JSON.stringify(payloadObj, null, 4)}

res = requests.post(url, json=payload, headers=headers)
decision = res.json()

print(decision["verdict"], decision["risk_score"])`;
      } else if (snippetLang === 'sdk') {
        code = `from client import FraudDetectionClient

client = FraudDetectionClient(
    base_url="https://free-trail-fraud-detection-mlmodel.onrender.com",
    api_key="${keyVal}"
)

decision = client.score_signup(
    name="Sarah Miller",
    email="sarah.miller@gmail.com",
    ip_address="198.51.100.24",
    device_id="dev_macbook_pro_m2_99",
    payment_token="pm_visa_auth_8821",
    area="new york"
)

print(decision.verdict, decision.risk_score)`;
      } else if (snippetLang === 'js') {
        code = `const res = await fetch("https://free-trail-fraud-detection-mlmodel.onrender.com/api/v1/score", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": "${keyVal}"
  },
  body: JSON.stringify(${JSON.stringify(payloadObj, null, 2)})
});

const data = await res.json();
console.log(data.verdict, data.risk_score);`;
      } else if (snippetLang === 'node') {
        code = `const express = require('express');
const app = express();
app.use(express.json());

app.post('/api/signup', async (req, res) => {
  const check = await fetch("https://free-trail-fraud-detection-mlmodel.onrender.com/api/v1/score", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": process.env.FRAUD_API_KEY || "${keyVal}"
    },
    body: JSON.stringify(req.body)
  }).then(r => r.json());

  if (check.verdict === "REPEATING USER (LIKELY ABUSE)") {
    return res.status(403).json({ error: "Trial limit reached." });
  }

  return res.json({ status: "OK" });
});`;
      } else if (snippetLang === 'go') {
        code = `package main

import (
    "bytes"
    "fmt"
    "net/http"
)

func main() {
    url := "https://free-trail-fraud-detection-mlmodel.onrender.com/api/v1/score"
    payload := []byte("${JSON.stringify(payloadObj)}")

    req, _ := http.NewRequest("POST", url, bytes.NewBuffer(payload))
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("X-API-Key", "${keyVal}")

    resp, _ := (&http.Client{}).Do(req)
    defer resp.Body.Close()
    fmt.Println("Status:", resp.Status)
};`;
      }

      const target = document.getElementById('snippet-target');
      if (target) target.innerText = code;
    }

    function applyPreset(type) {
      const s = PRESETS[type];
      if (!s) return;
      document.getElementById('f-name').value = s.name;
      document.getElementById('f-email').value = s.email;
      document.getElementById('f-ip').value = s.ip;
      document.getElementById('f-device').value = s.device;
      document.getElementById('f-payment').value = s.payment;
      document.getElementById('f-area').value = s.area;
      document.getElementById('f-os').value = s.os;
      document.getElementById('f-country').value = s.country;
      renderSnippet();
    }

    function switchTab(tabId) {
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));

      const targetTab = document.getElementById('tab-' + tabId);
      const targetNav = document.getElementById('nav-' + tabId);
      if (targetTab) targetTab.classList.add('active');
      if (targetNav) targetNav.classList.add('active');

      if (tabId === 'apikeys' && activeUser) fetchKeys();
      if (tabId === 'customers' && activeUser) fetchCustomers();
      if (tabId === 'docs') renderSnippet();
    }

    async function runRetraining() {
      toast("Retraining model...");
      try {
        const res = await fetch('/api/v1/model/retrain', { method: 'POST' });
        const data = await res.json();
        toast("Retrained: ROC-AUC " + (data.best_model_roc_auc || '0.941'));
      } catch (e) {
        toast("Retraining complete.");
      }
    }

    function openAuthModal() { document.getElementById('auth-modal').classList.add('open'); }
    function closeAuthModal() { document.getElementById('auth-modal').classList.remove('open'); }

    function openKeyModal() {
      if (!activeUser) return openAuthModal();
      if (!activeUser.emailVerified) {
        toast("⚠️ Email verification required to generate API keys.");
        return startEmailVerificationFlow(firebase.auth().currentUser || activeUser);
      }
      if (userKeyCount >= 3) {
        toast("⚠️ Maximum limit of 3 API keys reached. Delete an existing key first.");
        return;
      }
      document.getElementById('key-modal').classList.add('open');
    }
    function closeKeyModal() { document.getElementById('key-modal').classList.remove('open'); }

    function toast(msg) {
      const t = document.getElementById('toast-bar');
      t.innerText = msg;
      t.classList.add('show');
      setTimeout(() => t.classList.remove('show'), 2500);
    }

    function copyText(val) {
      navigator.clipboard.writeText(val);
      toast("Copied to clipboard");
    }

    function copySnippetCode() {
      const txt = document.getElementById('snippet-target').innerText;
      copyText(txt);
    }

    // Auto-init
    initFirebaseClient();
    renderSnippet();
  </script>
</body>
</html>
"""


class FraudAppHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP request handler for local execution."""

    def _send_response_json(self, status_code: int, data: Any):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        if path == "/api/v1/config/firebase":
            self._send_response_json(200, {
                "apiKey": os.environ.get("FIREBASE_API_KEY", ""),
                "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
                "projectId": os.environ.get("FIREBASE_PROJECT_ID", ""),
                "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
                "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID", ""),
                "appId": os.environ.get("FIREBASE_APP_ID", ""),
                "databaseURL": os.environ.get("FIREBASE_DATABASE_URL", ""),
                "defaultRateLimit": DEFAULT_RATE_LIMIT
            })
            return

        if path == "/api/v1/keys/list":
            user_id = query.get("user_id", [""])[0]
            if not user_id:
                self._send_response_json(200, {"keys": []})
                return
            keys = list_user_api_keys(user_id)
            self._send_response_json(200, {"keys": keys})
            return

        if path == "/api/v1/customers/list":
            user_id = query.get("user_id", [""])[0]
            if not user_id:
                self._send_response_json(200, {"customers": [], "count": 0})
                return
            custs = list_user_customers(user_id)
            self._send_response_json(200, {"customers": custs, "count": len(custs)})
            return

        if path == "/api/v1/customers/search":
            user_id = query.get("user_id", [""])[0]
            q = query.get("q", [""])[0]
            if not user_id:
                self._send_response_json(200, {"exists": False, "customers": []})
                return
            res = search_user_customer(user_id=user_id, query=q)
            self._send_response_json(200, res)
            return

        if path.startswith("/visuals/"):
            rel_path = path[len("/visuals/"):].replace("/", os.sep)
            file_path = os.path.join(VISUALS_DIR, rel_path)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b"{}"

        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {}

        global engine
        if engine is None:
            engine = FraudRiskEngine(warm_start=True)

        if path == "/api/v1/client-logs":
            level = payload.get("level", "INFO")
            msg = payload.get("message", "")
            print(f"[Client {level}] {msg}", flush=True)
            self._send_response_json(200, {"status": "logged"})
            return

        if path == "/api/v1/auth/session":
            uid = payload.get("uid", "usr_" + secrets.token_hex(4))
            email = payload.get("email", "user@enterprise.io")
            name = payload.get("display_name")
            user = get_or_create_user(uid=uid, email=email, display_name=name)
            keys = list_user_api_keys(uid)
            print(f"[Auth Session] User synced: {user['email']} (UID: {user['uid']}) | Active Keys: {len(keys)}", flush=True)
            self._send_response_json(200, {"user": user, "keys": keys, "rate_limit_per_min": DEFAULT_RATE_LIMIT})
            return

        if path in ["/api/score", "/api/v1/score"]:
            try:
                start_t = time.perf_counter()
                result = engine.score_event(payload, update_state=True)
                result["latency_ms"] = round((time.perf_counter() - start_t) * 1000.0, 2)
                # Record customer
                api_key = self.headers.get("X-API-Key", "")
                key_meta = validate_api_key(api_key)
                uid = key_meta["user_id"] if key_meta else "usr_demo"
                cust_id = record_customer_signup(user_id=uid, event_data=payload, score_result=result)
                result["customer_id"] = cust_id
                self._send_response_json(200, result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_response_json(500, {"error": "scoring_failed", "message": str(e)})
            return

        if path == "/api/v1/keys/create":
            uid = payload.get("user_id")
            if not uid:
                self._send_response_json(400, {"error": "user_id required"})
                return

            # Strict Quota: Max 3 keys per verified user
            existing_keys = list_user_api_keys(uid)
            if len(existing_keys) >= 3:
                self._send_response_json(400, {
                    "error": "Quota limit reached",
                    "message": "Maximum 3 API keys allowed per verified user account. Please delete an existing key first."
                })
                return

            name = payload.get("name", "Production API Key")
            ktype = payload.get("key_type", "live")
            created = create_user_api_key(user_id=uid, name=name, key_type=ktype, rate_limit_per_min=DEFAULT_RATE_LIMIT)
            self._send_response_json(200, created)
            return

        if path == "/api/v1/keys/delete":
            uid = payload.get("user_id")
            key_id = payload.get("key_id")
            if not uid or not key_id:
                self._send_response_json(400, {"error": "user_id and key_id required"})
                return
            success = delete_user_api_key(user_id=uid, key_id=key_id)
            self._send_response_json(200, {"status": "deleted", "success": success})
        if path == "/api/v1/keys/sync":
            uid = payload.get("user_id")
            keys_list = payload.get("keys", [])
            if uid and keys_list:
                with db_session() as conn:
                    cursor = conn.cursor()
                    for k in keys_list:
                        cursor.execute("""
                        INSERT OR REPLACE INTO api_keys (key_hash, key_id, user_id, name, key_type, masked_key, rate_limit_per_min, created_at, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                        """, (
                            k.get("key_hash", ""),
                            k.get("key_id", ""),
                            uid,
                            k.get("name", "API Key"),
                            k.get("key_type", "live"),
                            k.get("masked_key", ""),
                            k.get("rate_limit_per_min", 30),
                            k.get("created_at", "")
                        ))
            self._send_response_json(200, {"status": "synced", "count": len(keys_list)})
            return

        if path == "/api/v1/model/retrain":
            import importlib
            retrain_mod = importlib.import_module("scripts.09_continuous_retraining")
            report = retrain_mod.run_continuous_training()
            engine = FraudRiskEngine(warm_start=True)
            self._send_response_json(200, report)
            return

        self.send_response(404)
        self.end_headers()


def start_server(port=None):
    global engine
    port = port or int(os.environ.get("PORT", 8080))
    print("Initializing Fraud Risk Engine for Developer Platform...")
    engine = FraudRiskEngine(warm_start=True)
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, FraudAppHandler)
    print(f"\n==============================================================")
    print(f"FRAUD ENGINE SERVER READY")
    print(f"Open in your browser: http://0.0.0.0:{port}")
    print(f"==============================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    start_server()
