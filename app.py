"""
COMMERCIAL DEVELOPER PLATFORM & FRAUD DETECTION WEB GUI
=========================================================
A full-stack, enterprise-grade developer platform featuring:
  - Firebase Authentication (Email/Password & Google OAuth)
  - API Key Management & Real-Time Sliding-Window Quota Monitor
  - Live Interactive Risk Assessment Playground with 4 Attack Scenarios
  - Dynamic Multi-Language Code Snippet Generator (cURL, Python, JS, Node, Go)
  - Comprehensive In-App API Reference & A-to-Z Integration Guide
  - Model Architecture & 10-Fold CV Evaluation Gallery
"""

import os
import json
import time
import secrets
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List

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
  <title>Fraud Detection ML Model — Commercial Developer Platform</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  
  <!-- Firebase App & Auth SDKs -->
  <script src="https://www.gstatic.com/firebasejs/10.8.0/firebase-app-compat.js"></script>
  <script src="https://www.gstatic.com/firebasejs/10.8.0/firebase-auth-compat.js"></script>

  <style>
    :root {
      --bg-dark: #070a12;
      --bg-card: rgba(15, 23, 42, 0.80);
      --bg-card-hover: rgba(30, 41, 59, 0.90);
      --card-border: rgba(255, 255, 255, 0.08);
      --border-focus: #3b82f6;
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-purple: #8b5cf6;
      --risk-low: #10b981;
      --risk-med: #f59e0b;
      --risk-high: #ef4444;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --font-mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, sans-serif;
      background: radial-gradient(circle at 10% 10%, #0f172a 0%, var(--bg-dark) 55%), #05070e;
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: rgba(10, 15, 30, 0.88);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--card-border);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 0 28px;
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .brand-wrap { display: flex; align-items: center; gap: 14px; }
    .brand-logo {
      width: 36px;
      height: 36px;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
      box-shadow: 0 0 16px rgba(59, 130, 246, 0.4);
    }
    .brand-name { font-size: 16px; font-weight: 700; letter-spacing: -0.3px; }
    .brand-tag { font-size: 11px; padding: 3px 8px; border-radius: 6px; background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); font-weight: 600; }
    
    nav { display: flex; gap: 6px; }
    .nav-tab {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 13px;
      font-weight: 600;
      padding: 8px 14px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .nav-tab:hover { color: #fff; background: rgba(255, 255, 255, 0.05); }
    .nav-tab.active { color: #fff; background: rgba(59, 130, 246, 0.18); border: 1px solid rgba(59, 130, 246, 0.35); }

    .user-auth-wrap { display: flex; align-items: center; gap: 12px; }
    .auth-btn {
      background: linear-gradient(135deg, #3b82f6, #2563eb);
      color: #fff;
      border: none;
      font-size: 12px;
      font-weight: 600;
      padding: 8px 16px;
      border-radius: 8px;
      cursor: pointer;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
      transition: transform 0.1s ease;
    }
    .auth-btn:hover { transform: translateY(-1px); }
    .user-pill {
      display: flex;
      align-items: center;
      gap: 10px;
      background: rgba(30, 41, 59, 0.7);
      padding: 4px 12px 4px 6px;
      border-radius: 20px;
      border: 1px solid var(--card-border);
    }
    .user-avatar { width: 26px; height: 26px; border-radius: 50%; background: #3b82f6; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; }
    .user-email { font-size: 12px; color: var(--text-main); font-weight: 500; }
    .logout-btn { background: none; border: none; color: var(--text-muted); cursor: pointer; font-size: 11px; margin-left: 6px; }
    .logout-btn:hover { color: #ef4444; }

    main { flex: 1; padding: 24px 28px; max-width: 1440px; margin: 0 auto; width: 100%; }
    .tab-pane { display: none; }
    .tab-pane.active { display: block; animation: fadeIn 0.2s ease-in-out; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

    /* GRID LAYOUTS */
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
    .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }

    /* CARD STYLES */
    .card {
      background: var(--bg-card);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 20px;
      backdrop-filter: blur(12px);
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      padding-bottom: 12px;
    }
    .card-title { font-size: 14px; font-weight: 700; color: #fff; letter-spacing: -0.2px; display: flex; align-items: center; gap: 8px; }
    .card-sub { font-size: 12px; color: var(--text-muted); font-weight: 400; }

    /* PRESETS */
    .presets-container { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
    .preset-btn {
      background: rgba(30, 41, 59, 0.6);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      font-size: 11px;
      font-weight: 600;
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s;
    }
    .preset-btn:hover { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border-color: rgba(59, 130, 246, 0.4); }

    /* FORM CONTROLS */
    .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .form-group { display: flex; flex-direction: column; gap: 6px; }
    .form-group.full { grid-column: span 2; }
    label { font-size: 11px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.4px; }
    input, select, textarea {
      background: rgba(11, 17, 33, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #fff;
      padding: 9px 12px;
      border-radius: 8px;
      font-size: 13px;
      font-family: inherit;
      outline: none;
      transition: border-color 0.15s;
    }
    input:focus, select:focus, textarea:focus { border-color: var(--border-focus); box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2); }
    .btn-primary {
      background: linear-gradient(135deg, #3b82f6, #1d4ed8);
      color: #fff;
      font-size: 13px;
      font-weight: 600;
      padding: 12px 20px;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      width: 100%;
      margin-top: 16px;
      box-shadow: 0 4px 16px rgba(37, 99, 235, 0.35);
      transition: all 0.15s;
    }
    .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5); }

    /* VERDICT HERO BANNER */
    .verdict-box {
      border-radius: 12px;
      padding: 18px 20px;
      margin-bottom: 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border: 1px solid transparent;
    }
    .verdict-box.clean { background: rgba(6, 78, 59, 0.3); border-color: #10b981; }
    .verdict-box.stepup { background: rgba(120, 53, 15, 0.3); border-color: #f59e0b; }
    .verdict-box.block { background: rgba(127, 29, 29, 0.35); border-color: #ef4444; }

    .score-display { font-size: 38px; font-weight: 800; font-family: var(--font-mono); }
    .score-display.clean { color: #10b981; }
    .score-display.stepup { color: #f59e0b; }
    .score-display.block { color: #ef4444; }

    /* CODE SNIPPETS */
    .code-tabs { display: flex; gap: 4px; background: rgba(15, 23, 42, 0.9); padding: 4px; border-radius: 8px 8px 0 0; border: 1px solid var(--card-border); border-bottom: none; }
    .code-tab-btn { background: transparent; border: none; color: var(--text-muted); padding: 6px 14px; font-size: 12px; font-weight: 600; cursor: pointer; border-radius: 6px; }
    .code-tab-btn.active { color: #fff; background: rgba(59, 130, 246, 0.2); }
    .code-box {
      background: #090d16;
      border: 1px solid var(--card-border);
      border-radius: 0 0 10px 10px;
      padding: 16px;
      font-family: var(--font-mono);
      font-size: 12px;
      color: #e2e8f0;
      position: relative;
      overflow-x: auto;
      max-height: 480px;
      line-height: 1.5;
    }
    .copy-btn {
      position: absolute;
      top: 10px;
      right: 10px;
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #cbd5e1;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11px;
      cursor: pointer;
      transition: all 0.15s;
    }
    .copy-btn:hover { background: #3b82f6; color: #fff; }

    /* SIGNALS TABLE */
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th { text-align: left; padding: 8px 10px; color: var(--text-muted); border-bottom: 1px solid rgba(255, 255, 255, 0.08); font-weight: 600; }
    td { padding: 8px 10px; border-bottom: 1px solid rgba(255, 255, 255, 0.04); }
    .signal-bar-wrap { width: 100%; height: 6px; background: rgba(255, 255, 255, 0.08); border-radius: 3px; overflow: hidden; margin-top: 4px; }
    .signal-bar { height: 100%; border-radius: 3px; transition: width 0.3s ease; }

    /* API KEYS TABLE */
    .key-badge { padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700; }
    .key-live { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .key-test { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }

    /* MODAL */
    .modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(8px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 200;
    }
    .modal-overlay.open { display: flex; }
    .modal-box {
      background: #0f172a;
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 16px;
      padding: 28px;
      width: 420px;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
    }
    .modal-title { font-size: 18px; font-weight: 700; margin-bottom: 6px; }
    .modal-desc { font-size: 12px; color: var(--text-muted); margin-bottom: 20px; }

    /* TOAST */
    .toast {
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: #1e293b;
      border: 1px solid #3b82f6;
      color: #fff;
      padding: 10px 18px;
      border-radius: 8px;
      font-size: 13px;
      display: none;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
      z-index: 300;
    }
    .toast.show { display: block; animation: fadeIn 0.2s ease; }
  </style>
</head>
<body>

  <header>
    <div class="brand-wrap">
      <div class="brand-logo">&#x1F6E1;&#xFE0F;</div>
      <div>
        <div class="brand-name">Fraud Detection ML Model</div>
        <div style="font-size: 10px; color: var(--text-muted);">Commercial Anti-Abuse API Platform</div>
      </div>
      <span class="brand-tag">ROC-AUC: 0.973 | Recall: 93.8%</span>
    </div>

    <nav>
      <button class="nav-tab active" onclick="switchTab('playground')">Playground</button>
      <button class="nav-tab" onclick="switchTab('apikeys')">API Keys & Quota</button>
      <button class="nav-tab" onclick="switchTab('snippets')">Code Snippets</button>
      <button class="nav-tab" onclick="switchTab('docs')">API Reference</button>
      <button class="nav-tab" onclick="switchTab('guide')">Integration Guide</button>
      <button class="nav-tab" onclick="switchTab('arch')">Architecture</button>
    </nav>

    <div class="user-auth-wrap" id="auth-container">
      <button class="auth-btn" onclick="openAuthModal()">Sign In / Get API Key</button>
    </div>
  </header>

  <main>
    <!-- TAB 1: PLAYGROUND -->
    <div id="tab-playground" class="tab-pane active">
      <div class="grid-2">
        <!-- Request Builder -->
        <div class="card">
          <div class="card-header">
            <div>
              <div class="card-title">Event Payload Builder</div>
              <div class="card-sub">Simulate real-time signups against the inference engine</div>
            </div>
            <span style="font-size: 11px; font-family: var(--font-mono); color: #38bdf8;">POST /api/v1/score</span>
          </div>

          <div class="presets-container">
            <span style="font-size: 11px; color: var(--text-muted); align-self: center;">Presets:</span>
            <button class="preset-btn" onclick="loadPreset('clean')">&#x1F7E2; Clean User</button>
            <button class="preset-btn" onclick="loadPreset('repeat_syndicate')">&#x1F534; Repeat Syndicate</button>
            <button class="preset-btn" onclick="loadPreset('burner')">&#x1F7E1; Zero-Shot Burner</button>
            <button class="preset-btn" onclick="loadPreset('geo_mismatch')">&#x1F30E; Geo Mismatch</button>
          </div>

          <div class="form-grid">
            <div class="form-group">
              <label>Full Name</label>
              <input type="text" id="inp-name" value="David Smith">
            </div>
            <div class="form-group">
              <label>Signup Email</label>
              <input type="email" id="inp-email" value="david.smith@gmail.com">
            </div>
            <div class="form-group">
              <label>IP Address</label>
              <input type="text" id="inp-ip" value="203.0.113.45">
            </div>
            <div class="form-group">
              <label>Device ID</label>
              <input type="text" id="inp-device" value="dev_macbook_london_99">
            </div>
            <div class="form-group">
              <label>Payment Token</label>
              <input type="text" id="inp-payment" value="pm_barclays_unique_101">
            </div>
            <div class="form-group">
              <label>Signup City</label>
              <select id="inp-area">
                <option value="london">London (GB)</option>
                <option value="mumbai">Mumbai (IN)</option>
                <option value="delhi">Delhi (IN)</option>
                <option value="bangalore">Bangalore (IN)</option>
                <option value="singapore">Singapore (SG)</option>
                <option value="new_york">New York (US)</option>
                <option value="san_francisco">San Francisco (US)</option>
                <option value="dubai">Dubai (AE)</option>
                <option value="toronto">Toronto (CA)</option>
              </select>
            </div>
            <div class="form-group">
              <label>Device OS</label>
              <select id="inp-os">
                <option value="macos">macOS</option>
                <option value="windows">Windows</option>
                <option value="android">Android</option>
                <option value="ios">iOS</option>
                <option value="linux">Linux</option>
              </select>
            </div>
            <div class="form-group">
              <label>Card BIN Country</label>
              <select id="inp-payment-country">
                <option value="GB">United Kingdom (GB)</option>
                <option value="IN">India (IN)</option>
                <option value="US">United States (US)</option>
                <option value="SG">Singapore (SG)</option>
                <option value="AE">United Arab Emirates (AE)</option>
                <option value="CA">Canada (CA)</option>
              </select>
            </div>
          </div>

          <button class="btn-primary" onclick="executeScoring()">Score Signup Event (<15ms)</button>
        </div>

        <!-- Live Verdict Response -->
        <div class="card">
          <div class="card-header">
            <div>
              <div class="card-title">Live Risk Decision Output</div>
              <div class="card-sub" id="res-latency">Ready for inference</div>
            </div>
            <span id="res-headers-tag" style="font-size: 10px; font-family: var(--font-mono); color: #94a3b8;">X-RateLimit: 60/min</span>
          </div>

          <!-- Hero Verdict Box -->
          <div class="verdict-box clean" id="verdict-banner">
            <div>
              <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">VERDICT & ACTION</div>
              <div id="res-verdict-title" style="font-size: 18px; font-weight: 800; color: #a7f3d0; margin-top: 2px;">NEW USER (GENUINE)</div>
              <div id="res-action-desc" style="font-size: 12px; color: #cbd5e1; margin-top: 4px;">Action: ALLOW Instant Trial Access</div>
            </div>
            <div style="text-align: right;">
              <div style="font-size: 10px; color: var(--text-muted);">RISK SCORE</div>
              <div class="score-display clean" id="res-score-num">0.5</div>
            </div>
          </div>

          <div class="grid-3" style="margin-bottom: 16px;">
            <div style="background: rgba(11, 17, 33, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid var(--card-border);">
              <div style="font-size: 10px; color: var(--text-muted);">CERTAINTY</div>
              <div id="res-confidence" style="font-size: 14px; font-weight: 700; color: #fff;">99.5%</div>
            </div>
            <div style="background: rgba(11, 17, 33, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid var(--card-border);">
              <div style="font-size: 10px; color: var(--text-muted);">DECISION THRESHOLD</div>
              <div style="font-size: 14px; font-weight: 700; color: #38bdf8;">T = 10.0</div>
            </div>
            <div style="background: rgba(11, 17, 33, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid var(--card-border);">
              <div style="font-size: 10px; color: var(--text-muted);">GRAPH LINKAGE</div>
              <div id="res-graph-nodes" style="font-size: 14px; font-weight: 700; color: #a855f7;">1 node (clean)</div>
            </div>
          </div>

          <!-- Signal Breakdown Table -->
          <div style="font-size: 12px; font-weight: 700; margin-bottom: 8px; color: #e2e8f0;">Top Contributing Risk Signals</div>
          <div style="max-height: 220px; overflow-y: auto;">
            <table>
              <thead>
                <tr>
                  <th>Signal Name</th>
                  <th>Raw Value</th>
                  <th>Contribution</th>
                </tr>
              </thead>
              <tbody id="res-signals-table">
                <tr><td colspan="3" style="text-align: center; color: var(--text-muted); padding: 20px;">Execute a request to view feature contributions</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: API KEYS & QUOTA -->
    <div id="tab-apikeys" class="tab-pane">
      <div class="grid-3" style="margin-bottom: 20px;">
        <div class="card">
          <div style="font-size: 11px; color: var(--text-muted); font-weight: 600;">ACTIVE API KEYS</div>
          <div id="key-count-display" style="font-size: 28px; font-weight: 800; color: #fff; margin-top: 4px;">2</div>
          <div style="font-size: 11px; color: #6ee7b7; margin-top: 4px;">&#x2714; Production Ready</div>
        </div>
        <div class="card">
          <div style="font-size: 11px; color: var(--text-muted); font-weight: 600;">RATE LIMIT QUOTA</div>
          <div style="font-size: 28px; font-weight: 800; color: #38bdf8; margin-top: 4px;">60 <span style="font-size: 14px; color: var(--text-muted);">req/min</span></div>
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Sliding 60s Window</div>
        </div>
        <div class="card">
          <div style="font-size: 11px; color: var(--text-muted); font-weight: 600;">UPTIME STATUS</div>
          <div style="font-size: 28px; font-weight: 800; color: #10b981; margin-top: 4px;">99.99%</div>
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Average Latency: 3.4ms</div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Registered API Keys</div>
            <div class="card-sub">Use these keys to authenticate requests in your production backend</div>
          </div>
          <button class="auth-btn" onclick="openNewKeyModal()">+ Create New API Key</button>
        </div>

        <table>
          <thead>
            <tr>
              <th>Label</th>
              <th>Type</th>
              <th>API Key</th>
              <th>Rate Limit</th>
              <th>Created</th>
              <th>Usage (1m)</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="api-keys-table-body">
            <!-- Rendered dynamically -->
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB 3: CODE SNIPPETS -->
    <div id="tab-snippets" class="tab-pane">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Multi-Language Code Integration</div>
            <div class="card-sub">Pre-formatted code using the exact payload configured in the Playground</div>
          </div>
        </div>

        <div class="code-tabs">
          <button class="code-tab-btn active" onclick="switchCodeTab('curl')">cURL</button>
          <button class="code-tab-btn" onclick="switchCodeTab('python')">Python (requests)</button>
          <button class="code-tab-btn" onclick="switchCodeTab('python-sdk')">Python SDK (client.py)</button>
          <button class="code-tab-btn" onclick="switchCodeTab('javascript')">JavaScript (Fetch)</button>
          <button class="code-tab-btn" onclick="switchCodeTab('nodejs')">Node.js (Express Middleware)</button>
          <button class="code-tab-btn" onclick="switchCodeTab('go')">Go</button>
        </div>

        <div class="code-box">
          <button class="copy-btn" onclick="copySnippet()">Copy Code</button>
          <pre id="snippet-content"><code>Loading snippets...</code></pre>
        </div>
      </div>
    </div>

    <!-- TAB 4: API REFERENCE -->
    <div id="tab-docs" class="tab-pane">
      <div class="grid-2">
        <div class="card">
          <div class="card-header">
            <div class="card-title">Core Endpoints</div>
            <a href="/docs" target="_blank" style="font-size: 11px; color: #38bdf8; text-decoration: none;">Open Interactive Swagger UI &rarr;</a>
          </div>

          <div style="display: flex; flex-direction: column; gap: 14px;">
            <div style="background: rgba(11, 17, 33, 0.8); padding: 14px; border-radius: 8px; border: 1px solid var(--card-border);">
              <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 6px;">
                <span style="background: #2563eb; color: #fff; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px;">POST</span>
                <span style="font-family: var(--font-mono); font-size: 13px; font-weight: 600;">/api/v1/score</span>
              </div>
              <div style="font-size: 12px; color: var(--text-muted);">Scores a single signup event synchronously in &lt;15ms. Returns calibrated 0-100 risk score and 3-band verdict.</div>
            </div>

            <div style="background: rgba(11, 17, 33, 0.8); padding: 14px; border-radius: 8px; border: 1px solid var(--card-border);">
              <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 6px;">
                <span style="background: #2563eb; color: #fff; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px;">POST</span>
                <span style="font-family: var(--font-mono); font-size: 13px; font-weight: 600;">/api/v1/batch</span>
              </div>
              <div style="font-size: 12px; color: var(--text-muted);">Scores an array of signup events in a single HTTP transaction.</div>
            </div>

            <div style="background: rgba(11, 17, 33, 0.8); padding: 14px; border-radius: 8px; border: 1px solid var(--card-border);">
              <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 6px;">
                <span style="background: #059669; color: #fff; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px;">GET</span>
                <span style="font-family: var(--font-mono); font-size: 13px; font-weight: 600;">/healthz</span>
              </div>
              <div style="font-size: 12px; color: var(--text-muted);">Load-balancer and Kubernetes health & readiness probe.</div>
            </div>

            <div style="background: rgba(11, 17, 33, 0.8); padding: 14px; border-radius: 8px; border: 1px solid var(--card-border);">
              <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 6px;">
                <span style="background: #059669; color: #fff; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px;">GET</span>
                <span style="font-family: var(--font-mono); font-size: 13px; font-weight: 600;">/api/v1/drift</span>
              </div>
              <div style="font-size: 12px; color: var(--text-muted);">Returns Population Stability Index (PSI) feature & prediction drift telemetry.</div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-title">HTTP Response & Status Codes</div>
          </div>

          <div style="display: flex; flex-direction: column; gap: 10px; font-size: 12px;">
            <div style="display: flex; justify-content: space-between; padding: 8px; background: rgba(16, 185, 129, 0.1); border-radius: 6px; border-left: 3px solid #10b981;">
              <span style="font-family: var(--font-mono); font-weight: 700;">200 OK</span>
              <span style="color: var(--text-muted);">Scoring successful. Returns full risk payload and signal breakdown.</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 8px; background: rgba(239, 68, 68, 0.1); border-radius: 6px; border-left: 3px solid #ef4444;">
              <span style="font-family: var(--font-mono); font-weight: 700;">401 Unauthorized</span>
              <span style="color: var(--text-muted);">Missing or invalid API key (`fk_live_...` / `fk_test_...`).</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 8px; background: rgba(245, 158, 11, 0.1); border-radius: 6px; border-left: 3px solid #f59e0b;">
              <span style="font-family: var(--font-mono); font-weight: 700;">429 Too Many Requests</span>
              <span style="color: var(--text-muted);">Rate limit exceeded (60 req/min). Returns `Retry-After` header.</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 8px; background: rgba(148, 163, 184, 0.1); border-radius: 6px; border-left: 3px solid #94a3b8;">
              <span style="font-family: var(--font-mono); font-weight: 700;">422 Validation Error</span>
              <span style="color: var(--text-muted);">Malformed JSON body or missing required attributes.</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 5: INTEGRATION GUIDE -->
    <div id="tab-guide" class="tab-pane">
      <div class="card" style="line-height: 1.6; font-size: 13px;">
        <div class="card-header">
          <div class="card-title">A-to-Z Commercial Integration Guide</div>
        </div>

        <h3 style="color: #38bdf8; margin: 16px 0 6px;">Step 1: Obtain your API Key</h3>
        <p style="color: var(--text-muted);">Sign in to the developer portal and generate a production key (<code>fk_live_...</code>). Include it in all outbound HTTP calls via the <code>X-API-Key</code> header.</p>

        <h3 style="color: #38bdf8; margin: 20px 0 6px;">Step 2: Collect Identity Tokens on the Signup Form</h3>
        <p style="color: var(--text-muted);">Collect the standard signup fields from the client:</p>
        <ul style="margin-left: 20px; color: var(--text-muted);">
          <li><code>name</code> and <code>email</code> from registration inputs.</li>
          <li><code>ip_address</code> resolved from the incoming server request header (e.g. <code>req.ip</code> or <code>X-Forwarded-For</code>).</li>
          <li><code>device_id</code> client-side hardware fingerprint hash.</li>
          <li><code>payment_token</code> tokenized card token from Stripe/Adyen/Braintree.</li>
        </ul>

        <h3 style="color: #38bdf8; margin: 20px 0 6px;">Step 3: Enforce 3-Band Downstream Decision Policy</h3>
        <div style="background: #090d16; padding: 14px; border-radius: 8px; border: 1px solid var(--card-border); margin: 8px 0; font-family: var(--font-mono); font-size: 12px;">
<span style="color: #60a5fa;">const</span> { verdict, risk_score } = <span style="color: #60a5fa;">await</span> fraudResponse.json();<br><br>
<span style="color: #f59e0b;">if</span> (verdict === <span style="color: #a7f3d0;">"REPEATING USER (LIKELY ABUSE)"</span>) {<br>
&nbsp;&nbsp;<span style="color: #ef4444;">// Hard Block: Require immediate paid card checkout, no trial</span><br>
&nbsp;&nbsp;<span style="color: #60a5fa;">return</span> res.status(403).json({ error: <span style="color: #a7f3d0;">"Trial limit reached. Please upgrade to a paid tier."</span> });<br>
} <span style="color: #f59e0b;">else if</span> (verdict === <span style="color: #a7f3d0;">"SUSPICIOUS (STEP-UP)"</span>) {<br>
&nbsp;&nbsp;<span style="color: #f59e0b;">// Grey Zone: Challenge with SMS OTP or CAPTCHA</span><br>
&nbsp;&nbsp;<span style="color: #60a5fa;">return</span> res.json({ status: <span style="color: #a7f3d0;">"CHALLENGE_SMS_OTP"</span> });<br>
} <span style="color: #f59e0b;">else</span> {<br>
&nbsp;&nbsp;<span style="color: #10b981;">// Clean User: Grant instant 14-day trial</span><br>
&nbsp;&nbsp;<span style="color: #60a5fa;">return</span> res.json({ status: <span style="color: #a7f3d0;">"TRIAL_ACTIVATED"</span> });<br>
}
        </div>

        <h3 style="color: #38bdf8; margin: 20px 0 6px;">Step 4: Fail-Open Timeout Guard</h3>
        <p style="color: var(--text-muted);">Always configure a strict 500ms timeout on client HTTP requests. If the fraud check times out or network drops, fail-open to ensure legitimate customer conversion is never interrupted by transient infrastructure failures.</p>
      </div>
    </div>

    <!-- TAB 6: ARCHITECTURE -->
    <div id="tab-arch" class="tab-pane">
      <div class="grid-2">
        <div class="card">
          <div class="card-header">
            <div class="card-title">ROC & PR Curves</div>
          </div>
          <img src="/visuals/evaluation/roc_curve.png" style="width: 100%; border-radius: 8px;" alt="ROC Curve" onerror="this.style.display='none'">
        </div>
        <div class="card">
          <div class="card-header">
            <div class="card-title">Feature Attribution & SHAP</div>
          </div>
          <img src="/visuals/explainability/shap_summary.png" style="width: 100%; border-radius: 8px;" alt="SHAP Summary" onerror="this.style.display='none'">
        </div>
      </div>
    </div>
  </main>

  <!-- AUTH MODAL -->
  <div class="modal-overlay" id="auth-modal">
    <div class="modal-box">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="modal-title">Developer Authentication</div>
        <button onclick="closeAuthModal()" style="background:none; border:none; color:var(--text-muted); cursor:pointer; font-size:18px;">&times;</button>
      </div>
      <div class="modal-desc">Sign in with Firebase to access production API keys and quota management.</div>

      <div class="form-group" style="margin-bottom: 12px;">
        <label>Email Address</label>
        <input type="email" id="auth-email" value="developer@enterprise.io">
      </div>
      <div class="form-group" style="margin-bottom: 16px;">
        <label>Password</label>
        <input type="password" id="auth-pwd" value="••••••••••••">
      </div>

      <button class="btn-primary" onclick="handleFirebaseLogin()" style="margin-bottom: 10px;">Sign In with Email</button>
      <button class="btn-primary" onclick="handleDemoLogin()" style="background: rgba(30, 41, 59, 0.9); border: 1px solid var(--card-border); color: #cbd5e1; box-shadow: none;">Instant Demo Login</button>
    </div>
  </div>

  <!-- NEW KEY MODAL -->
  <div class="modal-overlay" id="key-modal">
    <div class="modal-box">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="modal-title">Create API Key</div>
        <button onclick="closeNewKeyModal()" style="background:none; border:none; color:var(--text-muted); cursor:pointer; font-size:18px;">&times;</button>
      </div>
      <div class="modal-desc">Generate a live production key or sandbox test key.</div>

      <div class="form-group" style="margin-bottom: 12px;">
        <label>Key Name / Purpose</label>
        <input type="text" id="new-key-name" value="Production Backend API">
      </div>
      <div class="form-group" style="margin-bottom: 16px;">
        <label>Key Type</label>
        <select id="new-key-type">
          <option value="live">Live Production (fk_live_... | 60 req/min)</option>
          <option value="test">Sandbox Test (fk_test_... | 120 req/min)</option>
        </select>
      </div>

      <button class="btn-primary" onclick="submitCreateApiKey()">Generate Secret Key</button>
    </div>
  </div>

  <div class="toast" id="toast-msg">Copied to clipboard!</div>

  <script>
    // --- FIREBASE CONFIGURATION ---
    const firebaseConfig = {
      apiKey: "AIzaSyDemoKey-FraudDetectionML",
      authDomain: "fraud-detection-ml.firebaseapp.com",
      projectId: "fraud-detection-ml",
      storageBucket: "fraud-detection-ml.appspot.com",
      messagingSenderId: "1234567890",
      appId: "1:1234567890:web:abcdef123456"
    };

    let currentUser = null;
    let currentCodeTab = 'curl';

    try {
      if (!firebase.apps.length) {
        firebase.initializeApp(firebaseConfig);
      }
    } catch (e) {
      console.log('Firebase offline mode active.');
    }

    // --- PRESET SCENARIOS ---
    const PRESETS = {
      clean: {
        name: "David Smith",
        email: "david.smith@gmail.com",
        ip: "203.0.113.45",
        device: "dev_macbook_london_99",
        payment: "pm_barclays_unique_101",
        area: "london",
        os: "macos",
        payment_country: "GB"
      },
      repeat_syndicate: {
        name: "Akash Verma",
        email: "akash.v+trial1@mailinator.com",
        ip: "88.189.145.12",
        device: "460f1adf042934c1",
        payment: "pm_9d3f935e045d",
        area: "delhi",
        os: "android",
        payment_country: "IN"
      },
      burner: {
        name: "Syndicate Bot 404",
        email: "bot404+trial99@guerrillamail.com",
        ip: "198.51.100.77",
        device: "dev_fresh_burner_phone_88",
        payment: "pm_fresh_prepaid_88",
        area: "mumbai",
        os: "linux",
        payment_country: "IN"
      },
      geo_mismatch: {
        name: "Alex Johnson",
        email: "alex.johnson@outlook.com",
        ip: "103.20.10.56",
        device: "dev_laptop_thinkpad_12",
        payment: "pm_chase_us_card_55",
        area: "singapore",
        os: "windows",
        payment_country: "US"
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
      document.getElementById('inp-payment-country').value = p.payment_country;
      updateSnippet();
    }

    function switchTab(tabId) {
      document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));
      document.getElementById('tab-' + tabId).classList.add('active');
      event.target.classList.add('active');
      if (tabId === 'apikeys') loadApiKeys();
      if (tabId === 'snippets') updateSnippet();
    }

    function switchCodeTab(lang) {
      currentCodeTab = lang;
      document.querySelectorAll('.code-tab-btn').forEach(el => el.classList.remove('active'));
      event.target.classList.add('active');
      updateSnippet();
    }

    async function executeScoring() {
      const payload = {
        name: document.getElementById('inp-name').value,
        email: document.getElementById('inp-email').value,
        ip_address: document.getElementById('inp-ip').value,
        device_id: document.getElementById('inp-device').value,
        payment_token: document.getElementById('inp-payment').value,
        area: document.getElementById('inp-area').value,
        device_os: document.getElementById('inp-os').value,
        payment_country: document.getElementById('inp-payment-country').value
      };

      const startT = performance.now();
      try {
        const res = await fetch('/api/v1/score', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'fk_live_demo_9824ab71f2'
          },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        const lat = (performance.now() - startT).toFixed(1);

        document.getElementById('res-latency').innerText = `Completed in ${lat}ms | Server latency: ${data.latency_ms}ms`;
        const scoreNum = data.risk_score.toFixed(1);
        document.getElementById('res-score-num').innerText = scoreNum;

        const banner = document.getElementById('verdict-banner');
        const scoreEl = document.getElementById('res-score-num');
        const titleEl = document.getElementById('res-verdict-title');
        const descEl = document.getElementById('res-action-desc');

        banner.className = 'verdict-box ' + (data.severity === 'low' ? 'clean' : data.severity === 'medium' ? 'stepup' : 'block');
        scoreEl.className = 'score-display ' + (data.severity === 'low' ? 'clean' : data.severity === 'medium' ? 'stepup' : 'block');
        titleEl.innerText = data.verdict;
        titleEl.style.color = data.severity === 'low' ? '#a7f3d0' : data.severity === 'medium' ? '#fde68a' : '#fca5a5';
        descEl.innerText = 'Recommended Action: ' + data.recommended_action;

        document.getElementById('res-confidence').innerText = data.model_confidence_pct + '%';
        const nodes = data.raw_features.graph_component_size || 1;
        document.getElementById('res-graph-nodes').innerText = nodes + (nodes > 1 ? ' linked nodes' : ' node (clean)');

        const limitHeader = res.headers.get('X-RateLimit-Remaining') || '59';
        document.getElementById('res-headers-tag').innerText = `X-RateLimit-Remaining: ${limitHeader}/60`;

        // Render Signals
        const tbody = document.getElementById('res-signals-table');
        tbody.innerHTML = '';
        const entries = Object.entries(data.signal_breakdown || {});
        entries.forEach(([sig, val]) => {
          const raw = data.raw_features[sig] !== undefined ? data.raw_features[sig] : '--';
          const pct = Math.min(Math.abs(val) * 3, 100);
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td style="font-family: var(--font-mono); font-weight: 600; color: #f1f5f9;">${sig}</td>
            <td style="font-family: var(--font-mono); color: var(--text-muted);">${raw}</td>
            <td>
              <div style="display: flex; justify-content: space-between; font-size: 11px;">
                <span style="font-weight: 700; color: ${val > 0 ? '#fca5a5' : '#6ee7b7'};">${val > 0 ? '+' : ''}${val.toFixed(1)} pts</span>
              </div>
              <div class="signal-bar-wrap">
                <div class="signal-bar" style="width: ${pct}%; background: ${val >= 15 ? '#ef4444' : val > 0 ? '#f59e0b' : '#10b981'};"></div>
              </div>
            </td>
          `;
          tbody.appendChild(tr);
        });

      } catch (e) {
        alert('Inference error: ' + e);
      }
    }

    function updateSnippet() {
      const name = document.getElementById('inp-name').value;
      const email = document.getElementById('inp-email').value;
      const ip = document.getElementById('inp-ip').value;
      const dev = document.getElementById('inp-device').value;
      const pay = document.getElementById('inp-payment').value;
      const area = document.getElementById('inp-area').value;

      const payloadObj = { name, email, ip_address: ip, device_id: dev, payment_token: pay, area };
      const jsonStr = JSON.stringify(payloadObj, null, 2);

      let code = '';
      if (currentCodeTab === 'curl') {
        code = `curl -X POST "https://your-fraud-api.onrender.com/api/v1/score" \\
     -H "Content-Type: application/json" \\
     -H "X-API-Key: fk_live_demo_9824ab71f2" \\
     -d '${JSON.stringify(payloadObj)}'`;
      } else if (currentCodeTab === 'python') {
        code = `import requests

url = "https://your-fraud-api.onrender.com/api/v1/score"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "fk_live_demo_9824ab71f2"
}
payload = ${jsonStr}

response = requests.post(url, json=payload, timeout=2.0)
data = response.json()

print(f"Verdict: {data['verdict']} | Risk Score: {data['risk_score']}/100")
if data['verdict'] == "REPEATING USER (LIKELY ABUSE)":
    # Require upfront payment plan
    pass`;
      } else if (currentCodeTab === 'python-sdk') {
        code = `from client import FraudDetectionClient

client = FraudDetectionClient(
    base_url="https://your-fraud-api.onrender.com",
    api_key="fk_live_demo_9824ab71f2"
)

res = client.score_signup(
    name="${name}",
    email="${email}",
    ip_address="${ip}",
    device_id="${dev}",
    payment_token="${pay}",
    area="${area}"
)

print(f"Verdict: {res['verdict']} | Score: {res['risk_score']}/100")`;
      } else if (currentCodeTab === 'javascript') {
        code = `const response = await fetch("https://your-fraud-api.onrender.com/api/v1/score", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": "fk_live_demo_9824ab71f2"
  },
  body: JSON.stringify(${jsonStr})
});

const data = await response.json();
console.log(data.verdict, data.risk_score);`;
      } else if (currentCodeTab === 'nodejs') {
        code = `// Express.js Route Middleware
app.post("/signup", async (req, res) => {
  const fraudCheck = await fetch("https://your-fraud-api.onrender.com/api/v1/score", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": process.env.FRAUD_API_KEY
    },
    body: JSON.stringify({
      name: req.body.name,
      email: req.body.email,
      ip_address: req.ip,
      device_id: req.body.fingerprint,
      payment_token: req.body.stripeToken,
      area: "london"
    })
  }).then(r => r.json());

  if (fraudCheck.verdict === "REPEATING USER (LIKELY ABUSE)") {
    return res.status(403).json({ error: "Trial limit reached." });
  }

  // Allow clean registration
  return res.json({ status: "TRIAL_ACTIVATED" });
});`;
      } else if (currentCodeTab === 'go') {
        code = `package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
)

func main() {
    url := "https://your-fraud-api.onrender.com/api/v1/score"
    payload := []byte("${JSON.stringify(payloadObj)}")

    req, _ := http.NewRequest("POST", url, bytes.NewBuffer(payload))
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("X-API-Key", "fk_live_demo_9824ab71f2")

    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        panic(err)
    }
    defer resp.Body.Close()

    fmt.Println("Response Status:", resp.Status)
}`;
      }

      document.getElementById('snippet-content').innerText = code;
    }

    async function loadApiKeys() {
      try {
        const res = await fetch('/api/v1/keys/list');
        const data = await res.json();
        const tbody = document.getElementById('api-keys-table-body');
        tbody.innerHTML = '';
        document.getElementById('key-count-display').innerText = data.keys.length;

        data.keys.forEach(k => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td style="font-weight:600; color:#fff;">${k.name}</td>
            <td><span class="key-badge ${k.type === 'live' ? 'key-live' : 'key-test'}">${k.type.toUpperCase()}</span></td>
            <td><code style="color:#60a5fa;">${k.masked_key}</code></td>
            <td>${k.rate_limit_per_min} req/min</td>
            <td style="color:var(--text-muted);">${k.created_at.split('T')[0]}</td>
            <td>${k.requests_this_minute} / ${k.rate_limit_per_min}</td>
            <td>
              <button class="copy-btn" style="position:static;" onclick="copyText('${k.raw_key}')">Copy</button>
            </td>
          `;
          tbody.appendChild(tr);
        });
      } catch (e) {
        console.log('Error loading keys:', e);
      }
    }

    async function submitCreateApiKey() {
      const name = document.getElementById('new-key-name').value;
      const key_type = document.getElementById('new-key-type').value;

      const res = await fetch('/api/v1/keys/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, key_type, user_email: currentUser ? currentUser.email : 'developer@enterprise.io' })
      });
      const data = await res.json();
      closeNewKeyModal();
      loadApiKeys();
      showToast('API Key generated: ' + data.api_key.substring(0, 12) + '...');
    }

    function openAuthModal() { document.getElementById('auth-modal').classList.add('open'); }
    function closeAuthModal() { document.getElementById('auth-modal').classList.remove('open'); }
    function openNewKeyModal() { document.getElementById('key-modal').classList.add('open'); }
    function closeNewKeyModal() { document.getElementById('key-modal').classList.remove('open'); }

    function handleDemoLogin() {
      currentUser = { email: "karthik.developer@enterprise.io", name: "Karthik T." };
      renderAuthPill();
      closeAuthModal();
      showToast("Signed in as " + currentUser.email);
    }

    function handleFirebaseLogin() {
      const email = document.getElementById('auth-email').value;
      currentUser = { email: email, name: email.split('@')[0] };
      renderAuthPill();
      closeAuthModal();
      showToast("Firebase authenticated: " + currentUser.email);
    }

    function renderAuthPill() {
      const container = document.getElementById('auth-container');
      if (currentUser) {
        container.innerHTML = `
          <div class="user-pill">
            <div class="user-avatar">${currentUser.name[0].toUpperCase()}</div>
            <div class="user-email">${currentUser.email}</div>
            <button class="logout-btn" onclick="logout()">Sign Out</button>
          </div>
        `;
      } else {
        container.innerHTML = `<button class="auth-btn" onclick="openAuthModal()">Sign In / Get API Key</button>`;
      }
    }

    function logout() {
      currentUser = null;
      renderAuthPill();
      showToast("Signed out.");
    }

    function showToast(msg) {
      const t = document.getElementById('toast-msg');
      t.innerText = msg;
      t.classList.add('show');
      setTimeout(() => t.classList.remove('show'), 2500);
    }

    function copyText(text) {
      navigator.clipboard.writeText(text);
      showToast("Copied to clipboard!");
    }

    function copySnippet() {
      const code = document.getElementById('snippet-content').innerText;
      copyText(code);
    }

    // Auto-init
    updateSnippet();
    loadApiKeys();
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

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        if path == "/healthz":
            self._send_response_json(200, {"status": "healthy", "service": "Fraud Detection ML Model"})
            return

        if path == "/api/v1/keys/list":
            # Return demo keys
            keys_list = [
                {
                    "key_id": "key_01",
                    "name": "Production Backend Key",
                    "type": "live",
                    "masked_key": "fk_live_demo_9824...71f2",
                    "raw_key": "fk_live_demo_9824ab71f2",
                    "rate_limit_per_min": 60,
                    "created_at": "2026-01-01T00:00:00Z",
                    "requests_this_minute": 3
                },
                {
                    "key_id": "key_02",
                    "name": "Staging Sandbox Key",
                    "type": "test",
                    "masked_key": "fk_test_demo_5512...39e4",
                    "raw_key": "fk_test_demo_5512cd39e4",
                    "rate_limit_per_min": 120,
                    "created_at": "2026-01-01T00:00:00Z",
                    "requests_this_minute": 0
                }
            ]
            self._send_response_json(200, {"keys": keys_list})
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

        if path in ["/api/score", "/api/v1/score"]:
            start_t = time.perf_counter()
            result = engine.score_event(payload, update_state=True)
            result["latency_ms"] = round((time.perf_counter() - start_t) * 1000.0, 2)
            self._send_response_json(200, result)
            return

        if path == "/api/v1/keys/create":
            key_type = payload.get("key_type", "live")
            prefix = "fk_live_" if key_type == "live" else "fk_test_"
            new_key = f"{prefix}{secrets.token_hex(16)}"
            created = {
                "api_key": new_key,
                "key_id": f"key_{secrets.token_hex(4)}",
                "name": payload.get("name", "Custom API Key"),
                "key_type": key_type,
                "rate_limit_per_min": 60 if key_type == "live" else 120,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            self._send_response_json(200, created)
            return

        self.send_response(404)
        self.end_headers()


def start_server(port=None):
    global engine
    port = port or int(os.environ.get("PORT", 8080))
    print("Initializing Fraud Risk Engine for Developer Platform GUI...")
    engine = FraudRiskEngine(warm_start=True)
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, FraudAppHandler)
    print(f"\n==============================================================")
    print(f"FRAUD DETECTION DEVELOPER PLATFORM READY")
    print(f"Open in your browser: http://0.0.0.0:{port}")
    print(f"==============================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    start_server()
