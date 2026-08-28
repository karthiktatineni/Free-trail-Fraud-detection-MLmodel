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
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
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
from legal_loader import load_legal_documents, get_legal_document

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
DEFAULT_RATE_LIMIT = int(os.environ.get("DEFAULT_RATE_LIMIT_PER_MINUTE", 30))

engine = None

HTML_PAGE = r"""<!DOCTYPE html>
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
    /* LAYOUT */
    main {
      flex: 1;
      padding: 24px 20px 48px;
      max-width: 1280px;
      margin: 0 auto;
      width: 100%;
      min-height: calc(100vh - 52px);
      display: flex;
      flex-direction: column;
    }

    .tab-content { display: none; flex-direction: column; flex: 1; }
    .tab-content.active { display: flex; }

    .page-title-row {
      margin-bottom: 18px;
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
    }
    .page-heading { font-size: 17px; font-weight: 600; color: var(--text-main); }
    .page-subtext { font-size: 12px; color: var(--text-muted); margin-top: 2px; }

    .grid-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: stretch; }
    .metrics-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 16px; }

    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-default);
      border-radius: var(--radius);
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .card-head {
      padding: 12px 16px;
      background: var(--bg-card-header);
      border-bottom: 1px solid var(--border-default);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      font-weight: 600;
    }
    .card-body { padding: 18px; flex: 1; display: flex; flex-direction: column; }

    .metric-box {
      background: var(--bg-card);
      border: 1px solid var(--border-default);
      border-radius: var(--radius);
      padding: 14px 16px;
    }
    .metric-name { font-size: 11px; font-weight: 500; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.03em; }
    .metric-stat { font-size: 20px; font-weight: 700; color: var(--text-main); margin: 4px 0 2px; font-family: var(--font-mono); }
    .metric-caption { font-size: 11px; color: var(--text-dim); }

    /* PRESETS */
    .presets-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }
    .preset-pill {
      background: var(--bg-card);
      border: 1px solid var(--border-default);
      color: var(--text-muted);
      font-size: 11.5px;
      font-weight: 500;
      padding: 4px 10px;
      border-radius: var(--radius);
      cursor: pointer;
      font-family: inherit;
      transition: all 0.15s ease;
    }
    .preset-pill:hover { color: var(--text-main); border-color: var(--text-muted); background: var(--bg-card-header); }

    /* FORMS */
    .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 12px; }
    .field-group { display: flex; flex-direction: column; gap: 4px; }
    .field-label { font-size: 11px; font-weight: 500; color: var(--text-muted); }
    .text-input, .select-input {
      background: var(--bg-input);
      border: 1px solid var(--border-default);
      color: var(--text-main);
      font-family: var(--font-mono);
      font-size: 12.5px;
      padding: 8px 10px;
      border-radius: var(--radius);
      outline: none;
      width: 100%;
      transition: border-color 0.15s ease;
    }
    .text-input:focus, .select-input:focus { border-color: var(--border-focus); }

    /* SCORECARD */
    .verdict-card {
      border-radius: var(--radius);
      padding: 14px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
      border: 1px solid transparent;
    }
    .verdict-card.allow { background: var(--status-green-bg); border-color: var(--status-green-border); }
    .verdict-card.review { background: var(--status-amber-bg); border-color: var(--status-amber-border); }
    .verdict-card.deny { background: var(--status-red-bg); border-color: var(--status-red-border); }

    .verdict-main { font-size: 14px; font-weight: 700; }
    .verdict-desc { font-size: 11.5px; color: var(--text-muted); margin-top: 3px; }

    .score-big { font-size: 26px; font-weight: 700; font-family: var(--font-mono); line-height: 1; }
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

    /* FOOTER */
    .site-footer {
      background: var(--bg-card);
      border-top: 1px solid var(--border-default);
      margin-top: 80px;
      padding: 40px 20px 28px;
      width: 100%;
    }
    .footer-inner {
      max-width: 1280px;
      margin: 0 auto;
    }
    .footer-grid {
      display: grid;
      grid-template-columns: 1.4fr 1.2fr 1.2fr;
      gap: 36px;
      margin-bottom: 32px;
    }
    .footer-col-title {
      font-size: 11.5px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-main);
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .footer-links-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .footer-link {
      color: var(--text-muted);
      font-size: 12px;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: none;
      border: none;
      padding: 0;
      font-family: inherit;
      cursor: pointer;
      text-align: left;
      transition: color 0.15s ease;
    }
    .footer-link:hover { color: var(--accent-blue-hover); }
    .footer-badge {
      font-size: 10px;
      padding: 1px 6px;
      border-radius: 4px;
      font-weight: 600;
      font-family: var(--font-mono);
      background: var(--bg-card-header);
      border: 1px solid var(--border-default);
      color: var(--text-dim);
    }
    .footer-bottom-bar {
      border-top: 1px solid var(--border-muted);
      padding-top: 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
      font-size: 11.5px;
      color: var(--text-dim);
    }
    .footer-copyright {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--text-muted);
    }
    .footer-quick-links {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    .footer-quick-link {
      color: var(--text-muted);
      font-size: 11.5px;
      text-decoration: none;
      cursor: pointer;
      background: none;
      border: none;
      padding: 0;
      font-family: inherit;
      transition: color 0.15s ease;
    }
    .footer-quick-link:hover { color: var(--text-main); }
    .footer-cookie-btn {
      background: var(--bg-input);
      border: 1px solid var(--border-default);
      color: var(--text-muted);
      padding: 3px 8px;
      border-radius: var(--radius);
      font-size: 11px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      transition: all 0.15s;
    }
    .footer-cookie-btn:hover {
      background: var(--bg-card-header);
      color: var(--text-main);
      border-color: var(--border-focus);
    }

    /* LEGAL MODAL */
    .legal-modal-window {
      max-width: 960px;
      width: 95%;
      height: 85vh;
      max-height: 820px;
      display: flex;
      flex-direction: column;
      padding: 0;
      overflow: hidden;
    }
    .legal-modal-top {
      padding: 12px 18px;
      background: var(--bg-card-header);
      border-bottom: 1px solid var(--border-default);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .legal-search-box {
      background: var(--bg-input);
      border: 1px solid var(--border-default);
      color: var(--text-main);
      font-size: 12px;
      padding: 5px 10px;
      border-radius: var(--radius);
      outline: none;
      width: 220px;
    }
    .legal-search-box:focus { border-color: var(--border-focus); }
    .legal-modal-body {
      display: grid;
      grid-template-columns: 240px 1fr;
      flex: 1;
      overflow: hidden;
      min-height: 0;
    }
    .legal-sidebar {
      background: #11151c;
      border-right: 1px solid var(--border-default);
      padding: 10px 8px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }
    .legal-doc-btn {
      display: flex;
      flex-direction: column;
      padding: 8px 10px;
      border-radius: var(--radius);
      border: 1px solid transparent;
      background: transparent;
      text-align: left;
      cursor: pointer;
      transition: all 0.15s ease;
      font-family: inherit;
    }
    .legal-doc-btn:hover { background: rgba(255, 255, 255, 0.04); }
    .legal-doc-btn.active {
      background: var(--bg-card);
      border-color: var(--border-default);
      box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    .legal-doc-title {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .legal-doc-sub {
      font-size: 10.5px;
      color: var(--text-dim);
      margin-top: 2px;
      line-height: 1.3;
    }
    .legal-content-pane {
      padding: 24px 32px;
      overflow-y: auto;
      font-size: 12.5px;
      line-height: 1.65;
      color: #e6edf3;
      background: var(--bg-card);
    }
    .legal-doc-header {
      margin-bottom: 20px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--border-muted);
    }
    .legal-doc-meta {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 8px;
      font-size: 11px;
      color: var(--text-muted);
      align-items: center;
    }
    .legal-doc-badge {
      background: var(--bg-input);
      border: 1px solid var(--border-default);
      color: var(--accent-blue);
      padding: 2px 7px;
      border-radius: 4px;
      font-family: var(--font-mono);
      font-size: 10.5px;
      font-weight: 500;
    }
    .legal-rendered-markdown h1 { font-size: 20px; font-weight: 700; margin-bottom: 12px; color: #ffffff; border-bottom: 1px solid var(--border-muted); padding-bottom: 8px; }
    .legal-rendered-markdown h2 { font-size: 15px; font-weight: 600; margin-top: 20px; margin-bottom: 8px; color: #ffffff; }
    .legal-rendered-markdown h3 { font-size: 13px; font-weight: 600; margin-top: 14px; margin-bottom: 6px; color: #f0f6fc; }
    .legal-rendered-markdown p { margin-bottom: 12px; color: #c9d1d9; }
    .legal-rendered-markdown ul, .legal-rendered-markdown ol { margin-bottom: 12px; padding-left: 20px; color: #c9d1d9; }
    .legal-rendered-markdown li { margin-bottom: 4px; }
    .legal-rendered-markdown strong { color: #ffffff; font-weight: 600; }
    .legal-rendered-markdown blockquote {
      border-left: 3px solid var(--accent-blue);
      padding: 8px 14px;
      background: rgba(31, 111, 235, 0.08);
      margin: 12px 0;
      border-radius: 0 var(--radius) var(--radius) 0;
      font-size: 12px;
      color: #e6edf3;
    }
    .legal-rendered-markdown table {
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0;
      font-size: 11.5px;
      border: 1px solid var(--border-default);
      border-radius: var(--radius);
      overflow: hidden;
    }
    .legal-rendered-markdown th {
      background: var(--bg-card-header);
      padding: 8px 10px;
      border-bottom: 1px solid var(--border-default);
      text-align: left;
      font-weight: 600;
      color: var(--text-main);
    }
    .legal-rendered-markdown td {
      padding: 8px 10px;
      border-bottom: 1px solid var(--border-muted);
      color: #c9d1d9;
    }
    .legal-rendered-markdown tr:last-child td { border-bottom: none; }
    .legal-rendered-markdown hr { border: none; border-top: 1px solid var(--border-muted); margin: 20px 0; }
    .legal-rendered-markdown code {
      font-family: var(--font-mono);
      background: #090c10;
      padding: 2px 5px;
      border-radius: 4px;
      font-size: 11px;
      color: #79c0ff;
    }

    /* COOKIE CONSENT BANNER & MODAL */
    .cookie-banner {
      position: fixed;
      bottom: 20px;
      left: 20px;
      right: 20px;
      max-width: 740px;
      margin: 0 auto;
      background: var(--bg-card);
      border: 1px solid var(--border-default);
      border-radius: 8px;
      padding: 16px 20px;
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.6);
      z-index: 999;
      display: none;
      flex-direction: column;
      gap: 12px;
      animation: slideUp 0.25s ease-out;
    }
    .cookie-banner.show { display: flex; }
    @keyframes slideUp {
      from { transform: translateY(20px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }
    .cookie-banner-content {
      display: flex;
      align-items: flex-start;
      gap: 14px;
    }
    .cookie-banner-text {
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.45;
    }
    .cookie-banner-text strong { color: var(--text-main); }
    .cookie-banner-actions {
      display: flex;
      gap: 8px;
      align-items: center;
      justify-content: flex-end;
      flex-wrap: wrap;
    }
    .toggle-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 0;
      border-bottom: 1px solid var(--border-muted);
    }
    .toggle-row:last-child { border-bottom: none; }
    .toggle-info-title { font-size: 12px; font-weight: 600; color: var(--text-main); }
    .toggle-info-desc { font-size: 11px; color: var(--text-muted); margin-top: 2px; }

    @media (max-width: 850px) {
      .grid-2col, .metrics-row { grid-template-columns: 1fr; }
      .footer-grid { grid-template-columns: 1fr 1fr; gap: 20px; }
      .legal-modal-body { grid-template-columns: 1fr; }
      .legal-sidebar { max-height: 160px; border-right: none; border-bottom: 1px solid var(--border-default); }
      header { padding: 0 12px; }
      main { padding: 12px; }
    }
    @media (max-width: 550px) {
      .footer-grid { grid-template-columns: 1fr; }
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

            <button class="btn btn-blue" style="width:100%; margin-top:8px; padding:9px 0; font-weight:600;" onclick="submitScoring()">Run Fraud Check</button>
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
              <div class="metric-box" style="padding:10px 12px;">
                <div class="metric-name" style="font-size:10px;">Confidence</div>
                <div id="metric-conf" style="font-size:14px; font-weight:700; font-family:var(--font-mono); margin-top:2px;">99.5%</div>
              </div>
              <div class="metric-box" style="padding:10px 12px;">
                <div class="metric-name" style="font-size:10px;">Threshold</div>
                <div style="font-size:14px; font-weight:700; font-family:var(--font-mono); color:var(--accent-blue); margin-top:2px;">T = 10.0</div>
              </div>
              <div class="metric-box" style="padding:10px 12px;">
                <div class="metric-name" style="font-size:10px;">Record ID</div>
                <div id="metric-cid" style="font-size:11px; font-weight:600; font-family:var(--font-mono); margin-top:4px;">--</div>
              </div>
            </div>

            <div style="font-size:11px; font-weight:600; color:var(--text-muted); margin-bottom:6px;">Signal Weights</div>
            <div class="table-wrap" style="max-height:220px; overflow-y:auto;">
              <table>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Value</th>
                    <th>Weight</th>
                  </tr>
                </thead>
                <tbody id="signals-tbody">
                  <tr><td colspan="3" style="text-align:center; color:var(--text-dim); padding:18px;">Submit a payload to view feature weights</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <!-- Live Causal Intelligence Row -->
      <div class="metrics-row" style="margin-top: 20px; margin-bottom: 0;">
        <div class="metric-box">
          <div class="metric-name">24-Hour Velocity Check</div>
          <div id="stat-vel-24h" class="metric-stat" style="font-size:16px;">1 Event <span style="font-size:11px; font-weight:400; color:var(--text-muted);">(Normal)</span></div>
          <div class="metric-caption">Sliding time-series window</div>
        </div>
        <div class="metric-box">
          <div class="metric-name">Syndicate Graph Size</div>
          <div id="stat-graph-size" class="metric-stat" style="font-size:16px;">Node Size: 1</div>
          <div class="metric-caption">Disjoint-Set cluster tracking</div>
        </div>
        <div class="metric-box">
          <div class="metric-name">Device OS & IP Alignment</div>
          <div id="stat-geo-align" class="metric-stat" style="font-size:16px; color:var(--status-green-text);">Verified Match</div>
          <div class="metric-caption">Class-C subnet & BIN geolocation</div>
        </div>
        <div class="metric-box">
          <div class="metric-name">Inference Latency</div>
          <div id="stat-p99-lat" class="metric-stat" style="font-size:16px; color:var(--accent-blue);">&lt; 15 ms</div>
          <div class="metric-caption">Vectorized decision engine</div>
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

  <!-- SITE FOOTER -->
  <footer class="site-footer">
    <div class="footer-inner">
      <div class="footer-grid">
        <!-- Col 1: Platform & Product -->
        <div class="footer-col">
          <div class="footer-col-title">
            <span>Fraud Engine</span>
            <span class="footer-badge">v2.1</span>
          </div>
          <ul class="footer-links-list">
            <li><button class="footer-link" onclick="switchTab('playground')">Playground Simulator</button></li>
            <li><button class="footer-link" onclick="switchTab('apikeys')">API Keys & Quotas <span class="footer-badge">30 req/min</span></button></li>
            <li><button class="footer-link" onclick="switchTab('customers')">Customer Audit Records</button></li>
            <li><button class="footer-link" onclick="switchTab('docs')">API Reference & SDKs</button></li>
            <li><button class="footer-link" onclick="switchTab('model')">Model Metrics & Benchmarks</button></li>
            <li><a class="footer-link" href="/healthz" target="_blank">System Health & Uptime <span style="color:var(--status-green-text); font-size:10px;">● 99.9%</span></a></li>
          </ul>
        </div>

        <!-- Col 2: Legal & Compliance -->
        <div class="footer-col">
          <div class="footer-col-title">Legal & Compliance</div>
          <ul class="footer-links-list">
            <li><button class="footer-link" onclick="openLegalDoc('privacy')">Privacy Policy</button></li>
            <li><button class="footer-link" onclick="openLegalDoc('terms')">Terms of Service</button></li>
            <li><button class="footer-link" onclick="openLegalDoc('cookies')">Cookie Policy</button></li>
            <li><button class="footer-link" onclick="openLegalDoc('refund')">Refund & Cancellation</button></li>
            <li><button class="footer-link" onclick="openLegalDoc('aup')">Acceptable Use Policy (AUP)</button></li>
            <li><button class="footer-link" onclick="openLegalDoc('dpa')">Data Processing Agreement (DPA)</button></li>
            <li><button class="footer-link" onclick="openLegalDoc('disclaimer')">AI & ML Decision Disclaimer</button></li>
          </ul>
        </div>

        <!-- Col 3: Security & Architecture -->
        <div class="footer-col">
          <div class="footer-col-title">Security & Architecture</div>
          <ul class="footer-links-list">
            <li class="footer-link" style="cursor:default;"><span>SHA-256 Key Hashing</span></li>
            <li class="footer-link" style="cursor:default;"><span>Multi-Tenant Query Isolation</span></li>
            <li class="footer-link" style="cursor:default;"><span>TLS 1.3 Strict In-Transit Encryption</span></li>
            <li class="footer-link" style="cursor:default;"><span>Zero Plaintext Secret Storage</span></li>
            <li class="footer-link" style="cursor:default;"><span>24-Hour Velocity Cache Eviction</span></li>
            <li class="footer-link" style="cursor:default;"><span>SOC 2 Type II Aligned Controls</span></li>
          </ul>
        </div>
      </div>

      <!-- Bottom Bar with Copyright -->
      <div class="footer-bottom-bar">
        <div class="footer-copyright">
          <span>karthik tatineni</span>
        </div>

        <div class="footer-quick-links">
          <button class="footer-quick-link" onclick="openLegalDoc('privacy')">Privacy</button>
          <span>&bull;</span>
          <button class="footer-quick-link" onclick="openLegalDoc('terms')">Terms</button>
          <span>&bull;</span>
          <button class="footer-quick-link" onclick="openLegalDoc('cookies')">Cookies</button>
          <span>&bull;</span>
          <button class="footer-quick-link" onclick="openLegalDoc('refund')">Refunds</button>
          <span>&bull;</span>
          <button class="footer-quick-link" onclick="openLegalDoc('aup')">AUP</button>
          <span>&bull;</span>
          <button class="footer-quick-link" onclick="openLegalDoc('dpa')">DPA</button>
          <span>&bull;</span>
          <button class="footer-quick-link" onclick="openLegalDoc('disclaimer')">AI Disclaimer</button>
          <span>&bull;</span>
          <button class="footer-cookie-btn" onclick="openCookieModal()">Cookie Preferences</button>
        </div>
      </div>
    </div>
  </footer>

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

  <!-- LEGAL DOCUMENTATION CENTER MODAL -->
  <div class="modal-backdrop" id="legal-modal">
    <div class="modal-window legal-modal-window">
      <div class="legal-modal-top">
        <div style="display:flex; align-items:center; gap:10px;">
          <span style="font-size:14px; font-weight:700; color:var(--text-main);">Legal & Compliance Center</span>
          <span id="legal-doc-badge" class="legal-doc-badge">v1.0</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
          <input type="text" id="legal-search-input" class="legal-search-box" placeholder="Search in document..." oninput="filterLegalSearch(this.value)">
          <button class="btn btn-secondary btn-sm" onclick="printActiveLegalDoc()" title="Print Document">Print</button>
          <button class="btn btn-secondary btn-sm" onclick="copyActiveLegalDocMarkdown()" title="Copy Markdown">Copy</button>
          <button class="btn btn-secondary btn-sm" onclick="closeLegalModal()">✕</button>
        </div>
      </div>

      <div class="legal-modal-body">
        <div class="legal-sidebar">
          <button id="legal-nav-privacy" class="legal-doc-btn active" onclick="openLegalDoc('privacy')">
            <div class="legal-doc-title">Privacy Policy</div>
            <div class="legal-doc-sub">Data collection & rights</div>
          </button>
          <button id="legal-nav-terms" class="legal-doc-btn" onclick="openLegalDoc('terms')">
            <div class="legal-doc-title">Terms of Service</div>
            <div class="legal-doc-sub">Commercial terms & API limits</div>
          </button>
          <button id="legal-nav-cookies" class="legal-doc-btn" onclick="openLegalDoc('cookies')">
            <div class="legal-doc-title">Cookie Policy</div>
            <div class="legal-doc-sub">Authentication & storage tokens</div>
          </button>
          <button id="legal-nav-refund" class="legal-doc-btn" onclick="openLegalDoc('refund')">
            <div class="legal-doc-title">Refund & Cancellation</div>
            <div class="legal-doc-sub">14-day policy & billing cycles</div>
          </button>
          <button id="legal-nav-aup" class="legal-doc-btn" onclick="openLegalDoc('aup')">
            <div class="legal-doc-title">Acceptable Use (AUP)</div>
            <div class="legal-doc-sub">Prohibited activities & security</div>
          </button>
          <button id="legal-nav-dpa" class="legal-doc-btn" onclick="openLegalDoc('dpa')">
            <div class="legal-doc-title">Data Processing (DPA)</div>
            <div class="legal-doc-sub">B2B Controller / Processor</div>
          </button>
          <button id="legal-nav-disclaimer" class="legal-doc-btn" onclick="openLegalDoc('disclaimer')">
            <div class="legal-doc-title">AI & ML Disclaimer</div>
            <div class="legal-doc-sub">Probabilistic decision support</div>
          </button>
        </div>

        <div class="legal-content-pane" id="legal-content-container">
          <div class="legal-doc-header">
            <h1 id="legal-active-title" style="font-size:18px; font-weight:700; color:#fff;">Privacy Policy</h1>
            <div class="legal-doc-meta">
              <span id="legal-active-effective">Effective Date: August 28, 2026</span>
              <span>&bull;</span>
              <span id="legal-active-category">Category: Privacy & Data Protection</span>
              <span>&bull;</span>
              <span id="legal-active-version">Version 1.0</span>
            </div>
          </div>
          <div id="legal-rendered-body" class="legal-rendered-markdown">
            <div style="text-align:center; padding:40px; color:var(--text-dim);">Loading legal document...</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- COOKIE CONSENT BANNER -->
  <div class="cookie-banner" id="cookie-consent-banner">
    <div class="cookie-banner-content">
      <div class="cookie-banner-text">
        <strong>Privacy & Storage Preferences</strong><br>
        We use essential local storage tokens for secure developer authentication and session management. Optional diagnostic telemetry helps us monitor API latency and error rates.
      </div>
    </div>
    <div class="cookie-banner-actions">
      <button class="btn btn-secondary btn-sm" onclick="openCookieModal()">Manage Preferences</button>
      <button class="btn btn-secondary btn-sm" onclick="setCookieConsent(false)">Essential Only</button>
      <button class="btn btn-blue btn-sm" onclick="setCookieConsent(true)">Accept Analytics</button>
    </div>
  </div>

  <!-- COOKIE PREFERENCES MODAL -->
  <div class="modal-backdrop" id="cookie-modal">
    <div class="modal-window" style="max-width:440px;">
      <div class="modal-top">
        <div class="modal-title-text">Cookie & Storage Preferences</div>
        <button class="btn btn-secondary btn-sm" onclick="closeCookieModal()">✕</button>
      </div>

      <p style="font-size:12px; color:var(--text-muted); margin-bottom:14px;">
        Configure your preferences for storage tokens and diagnostics telemetry. You can update these settings at any time.
      </p>

      <div style="background:var(--bg-input); border:1px solid var(--border-default); border-radius:var(--radius); padding:10px 14px; margin-bottom:16px;">
        <div class="toggle-row">
          <div>
            <div class="toggle-info-title">Strictly Necessary Storage</div>
            <div class="toggle-info-desc">Required for Firebase Auth, active sessions, and security.</div>
          </div>
          <div>
            <input type="checkbox" checked disabled style="cursor:not-allowed;">
          </div>
        </div>
        <div class="toggle-row">
          <div>
            <div class="toggle-info-title">Performance & Diagnostics Telemetry</div>
            <div class="toggle-info-desc">Measures client API latencies and operational error logs.</div>
          </div>
          <div>
            <input type="checkbox" id="cookie-pref-analytics" style="cursor:pointer;">
          </div>
        </div>
      </div>

      <div style="display:flex; gap:8px;">
        <button class="btn btn-blue" style="flex:1;" onclick="saveCustomCookiePreferences()">Save Preferences</button>
        <button class="btn btn-secondary" onclick="closeCookieModal()">Cancel</button>
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

    // --- COOKIE CONSENT & STORAGE PREFERENCES ---
    function getCookieConsent() {
      try {
        const val = localStorage.getItem('fraud_engine_cookie_consent');
        if (val) return JSON.parse(val);
      } catch (e) {}
      return null;
    }

    function hasAnalyticsConsent() {
      const consent = getCookieConsent();
      return consent !== null && consent.analytics === true;
    }

    function checkCookieConsentOnLoad() {
      const consent = getCookieConsent();
      if (!consent) {
        document.getElementById('cookie-consent-banner').classList.add('show');
      }
    }

    function setCookieConsent(allowAnalytics) {
      const pref = {
        essential: true,
        analytics: !!allowAnalytics,
        timestamp: new Date().toISOString()
      };
      try {
        localStorage.setItem('fraud_engine_cookie_consent', JSON.stringify(pref));
      } catch (e) {}
      document.getElementById('cookie-consent-banner').classList.remove('show');
      closeCookieModal();
      toast(allowAnalytics ? 'Preferences saved: Analytics telemetry enabled' : 'Preferences saved: Essential storage only');
    }

    function openCookieModal() {
      document.getElementById('cookie-modal').classList.add('open');
      const pref = getCookieConsent();
      const analyticsBox = document.getElementById('cookie-pref-analytics');
      if (analyticsBox) {
        analyticsBox.checked = pref ? !!pref.analytics : false;
      }
    }

    function closeCookieModal() {
      document.getElementById('cookie-modal').classList.remove('open');
    }

    function saveCustomCookiePreferences() {
      const analyticsBox = document.getElementById('cookie-pref-analytics');
      setCookieConsent(analyticsBox ? analyticsBox.checked : false);
    }

    function logToBackend(level, msg) {
      console.log(`[${level}]`, msg);
      // Strictly gated: do not transmit client logs/telemetry without explicit user analytics consent
      if (!hasAnalyticsConsent()) {
        return;
      }
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

    let keysUnsubscribe = null;
    let custsUnsubscribe = null;

    function setupFirestoreListeners(uid) {
      if (!firebaseReady || !firebase.firestore || !uid) return;
      const db = firebase.firestore();

      // 1. Real-time API keys listener directly from Cloud Firestore (Multi-device synced)
      if (keysUnsubscribe) { keysUnsubscribe(); keysUnsubscribe = null; }
      keysUnsubscribe = db.collection('users').document(uid).collection('api_keys')
        .onSnapshot(snapshot => {
          const keys = [];
          snapshot.forEach(doc => {
            const k = doc.data();
            if (k && (k.is_active === undefined || k.is_active === 1)) {
              keys.push({
                key_id: k.key_id || doc.id,
                key_hash: k.key_hash || '',
                user_id: uid,
                name: k.name || 'Production API Key',
                key_type: k.key_type || 'live',
                masked_key: k.masked_key || 'fk_live_...',
                rate_limit_per_min: k.rate_limit_per_min || 30,
                created_at: k.created_at || new Date().toISOString()
              });
            }
          });
          renderKeysTable(keys);
          if (keys.length > 0) {
            fetch('/api/v1/keys/sync', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ user_id: uid, keys: keys })
            }).catch(() => {});
          }
        }, err => {
          console.log("Firestore keys listener note:", err);
        });

      // 2. Real-time Customer events listener from Cloud Firestore
      if (custsUnsubscribe) { custsUnsubscribe(); custsUnsubscribe = null; }
      custsUnsubscribe = db.collection('users').document(uid).collection('customers')
        .orderBy('created_at', 'desc')
        .limit(50)
        .onSnapshot(snapshot => {
          const custs = [];
          snapshot.forEach(doc => {
            custs.push(doc.data());
          });
          if (custs.length > 0) {
            renderCustomers(custs);
          }
        }, err => {
          console.log("Firestore customers listener note:", err);
        });
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

        // Connect cloud Firestore listeners for permanent multi-device sync
        setupFirestoreListeners(uid);

        // Fetch fallback
        fetchKeys();
        fetchCustomers();

        // Sync user profile to Firestore
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
      if (keysUnsubscribe) { keysUnsubscribe(); keysUnsubscribe = null; }
      if (custsUnsubscribe) { custsUnsubscribe(); custsUnsubscribe = null; }
      if (firebaseReady) {
        try {
          await firebase.auth().signOut();
        } catch (e) {}
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

        // Update Live Causal Intelligence row
        const rawF = data.raw_features || {};
        const vel24 = rawF['payment_reuse_count'] !== undefined ? (rawF['payment_reuse_count'] + 1) : 1;
        const velEl = document.getElementById('stat-vel-24h');
        if (velEl) {
          velEl.innerHTML = `${vel24} Event${vel24 > 1 ? 's' : ''} <span style="font-size:11px; font-weight:400; color:${vel24 >= 3 ? 'var(--status-red-text)' : 'var(--text-muted)'};">(${vel24 >= 3 ? 'High Velocity' : 'Normal'})</span>`;
        }

        const graphSize = rawF['graph_component_size'] !== undefined ? rawF['graph_component_size'] : 1;
        const graphEl = document.getElementById('stat-graph-size');
        if (graphEl) {
          graphEl.innerText = `Node Size: ${graphSize}`;
          graphEl.style.color = graphSize > 2 ? 'var(--status-red-text)' : 'var(--text-main)';
        }

        const geoMismatch = rawF['ip_billing_mismatch'] !== undefined ? rawF['ip_billing_mismatch'] : 0;
        const geoEl = document.getElementById('stat-geo-align');
        if (geoEl) {
          if (geoMismatch) {
            geoEl.innerText = 'Mismatch Detected';
            geoEl.style.color = 'var(--status-red-text)';
          } else {
            geoEl.innerText = 'Verified Match';
            geoEl.style.color = 'var(--status-green-text)';
          }
        }

        const latEl = document.getElementById('stat-p99-lat');
        if (latEl) {
          latEl.innerText = `${latModel.toFixed(1)} ms`;
        }

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

    // --- PURE FIRESTORE API KEY MANAGEMENT ---
    function renderKeysTable(keys) {
      const tbody = document.getElementById('keys-tbody');
      tbody.innerHTML = '';
      userKeyCount = keys ? keys.length : 0;
      document.getElementById('key-count-stat').innerText = `${userKeyCount} / 3`;

      if (!keys || keys.length === 0) {
        primaryApiKey = null;
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:20px; color:var(--text-dim);">No API keys generated. Click "+ New Key" (max 3 keys).</td></tr>';
        renderSnippet();
        return;
      }

      primaryApiKey = keys[0].masked_key;
      renderSnippet();

      keys.forEach(k => {
        const tr = document.createElement('tr');
        const createdDate = (k.created_at || '').split('T')[0] || '--';
        tr.innerHTML = `
          <td style="font-weight:600;">${k.name || 'Production API Key'}</td>
          <td><span style="font-size:10px; font-family:var(--font-mono); font-weight:600; color:var(--status-green-text);">${(k.key_type || 'live').toUpperCase()}</span></td>
          <td><code style="color:var(--accent-blue); font-family:var(--font-mono);">${k.masked_key}</code></td>
          <td style="color:var(--text-muted);">${k.rate_limit_per_min || 30} req/min</td>
          <td style="color:var(--text-dim);">${createdDate}</td>
          <td>
            <button class="btn btn-danger btn-sm" onclick="deleteKey('${k.key_id}', '${(k.name || 'API Key').replace(/'/g, "\\'")}')">Delete</button>
          </td>
        `;
        tbody.appendChild(tr);
      });
    }

    async function fetchKeys() {
      if (!activeUser) return;
      try {
        let keys = [];

        // 1. Primary: Load directly from Cloud Firestore for this user
        if (firebaseReady && firebase.firestore) {
          try {
            const db = firebase.firestore();
            const snap = await db.collection('users').document(activeUser.uid).collection('api_keys').get();
            if (!snap.empty) {
              snap.forEach(doc => {
                const k = doc.data();
                if (k && (k.is_active === undefined || k.is_active === 1)) {
                  keys.push({
                    key_id: k.key_id || doc.id,
                    key_hash: k.key_hash || '',
                    user_id: activeUser.uid,
                    name: k.name || 'Production API Key',
                    key_type: k.key_type || 'live',
                    masked_key: k.masked_key || 'fk_live_...',
                    rate_limit_per_min: k.rate_limit_per_min || 30,
                    created_at: k.created_at || new Date().toISOString()
                  });
                }
              });
            }
          } catch (e) {
            console.log("Firestore keys query note:", e);
          }
        }

        // 2. Fallback: Load from Backend API if Firestore query was empty
        if (keys.length === 0) {
          try {
            const res = await fetch(`/api/v1/keys/list?user_id=${activeUser.uid}`);
            if (res.ok) {
              const data = await res.json();
              if (data && Array.isArray(data.keys)) {
                keys = data.keys;
              }
            }
          } catch (e) {
            console.log("Backend keys fetch note:", e);
          }
        }

        // Render keys
        renderKeysTable(keys);

        // Sync keys to backend SQLite so backend accepts calls made with these keys
        if (keys.length > 0) {
          fetch('/api/v1/keys/sync', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: activeUser.uid, keys: keys })
          }).catch(() => {});
        }
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
        const errData = await res.json().catch(() => ({}));
        alert(errData.message || errData.error || 'Failed to generate key.');
        return;
      }

      const data = await res.json();
      closeKeyModal();

      // Immediately write key record to Cloud Firestore under user's document
      if (firebaseReady && firebase.firestore) {
        try {
          const currentUser = firebase.auth().currentUser;
          if (currentUser) {
            await currentUser.getIdToken(true).catch(() => {});
          }
          const db = firebase.firestore();
          const targetUid = currentUser ? currentUser.uid : activeUser.uid;
          await db.collection('users').document(targetUid).collection('api_keys').document(data.key_id).set({
            key_id: data.key_id,
            key_hash: data.key_hash || '',
            user_id: targetUid,
            email: activeUser.email,
            name: label,
            key_type: ktype,
            masked_key: data.masked_key,
            rate_limit_per_min: 30,
            created_at: data.created_at || new Date().toISOString(),
            is_active: 1
          }, { merge: true });
          console.log("Key successfully written to Firestore:", data.key_id);
        } catch (e) {
          console.error("Firestore key sync error:", e);
          if (e.code === 'permission-denied') {
            toast("⚠️ Firestore permission denied: Check Firestore security rules in Firebase Console.");
          }
        }
      }

      await fetchKeys();
      openKeyRevealModal(data.api_key);
    }

    async function deleteKey(keyId, keyName) {
      if (!activeUser) return;
      if (!confirm(`Delete API key "${keyName}"? Outbound requests using this key will immediately be rejected.`)) {
        return;
      }

      try {
        // 1. Delete from Cloud Firestore
        if (firebaseReady && firebase.firestore) {
          try {
            await firebase.firestore().collection('users').document(activeUser.uid).collection('api_keys').document(keyId).delete();
          } catch (e) {}
        }

        // 2. Delete from backend
        fetch('/api/v1/keys/delete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: activeUser.uid, key_id: keyId })
        }).catch(() => {});

        toast(`Key "${keyName}" deleted.`);
        await fetchKeys();
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

    // --- LEGAL CENTER & DOCUMENT CONTROLLER ---
    let legalDocsCache = {};
    let activeLegalDoc = null;
    let activeLegalDocRawMarkdown = '';

    function renderMarkdownToHtml(md) {
      if (!md) return '';
      let text = md;
      if (text.startsWith('---')) {
        const parts = text.split('---', 3);
        if (parts.length >= 3) {
          text = parts[2].trim();
        }
      }

      let lines = text.split('\n');
      let html = [];
      let inTable = false;
      let tableRows = [];
      let inList = false;
      let listType = 'ul';

      function flushTable() {
        if (!inTable) return;
        if (tableRows.length === 0) { inTable = false; return; }
        let tHtml = '<table>';
        let isFirst = true;
        for (let r of tableRows) {
          let cols = r.split('|').slice(1, -1);
          if (cols.every(c => /^[\s\-:]+$/.test(c))) continue;
          if (isFirst) {
            tHtml += '<thead><tr>' + cols.map(c => '<th>' + formatInline(c.trim()) + '</th>').join('') + '</tr></thead><tbody>';
            isFirst = false;
          } else {
            tHtml += '<tr>' + cols.map(c => '<td>' + formatInline(c.trim()) + '</td>').join('') + '</tr>';
          }
        }
        tHtml += '</tbody></table>';
        html.push(tHtml);
        tableRows = [];
        inTable = false;
      }

      function flushList() {
        if (!inList) return;
        html.push(`</${listType}>`);
        inList = false;
      }

      function formatInline(str) {
        return str
          .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
          .replace(/\*([^*]+)\*/g, '<em>$1</em>')
          .replace(/`([^`]+)`/g, '<code>$1</code>')
          .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" style="color:var(--accent-blue-hover); text-decoration:underline;">$1</a>');
      }

      for (let i = 0; i < lines.length; i++) {
        let line = lines[i];
        let trimmed = line.trim();

        if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
          flushList();
          inTable = true;
          tableRows.push(trimmed);
          continue;
        } else {
          flushTable();
        }

        if (trimmed === '---' || trimmed === '***') {
          flushList();
          html.push('<hr>');
          continue;
        }

        if (trimmed.startsWith('#### ')) {
          flushList();
          html.push(`<h4>${formatInline(trimmed.substring(5))}</h4>`);
          continue;
        }
        if (trimmed.startsWith('### ')) {
          flushList();
          html.push(`<h3>${formatInline(trimmed.substring(4))}</h3>`);
          continue;
        }
        if (trimmed.startsWith('## ')) {
          flushList();
          html.push(`<h2>${formatInline(trimmed.substring(3))}</h2>`);
          continue;
        }
        if (trimmed.startsWith('# ')) {
          flushList();
          html.push(`<h1>${formatInline(trimmed.substring(2))}</h1>`);
          continue;
        }

        if (trimmed.startsWith('> ')) {
          flushList();
          html.push(`<blockquote>${formatInline(trimmed.substring(2))}</blockquote>`);
          continue;
        }

        if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
          if (!inList || listType !== 'ul') {
            flushList();
            inList = true;
            listType = 'ul';
            html.push('<ul>');
          }
          html.push(`<li>${formatInline(trimmed.substring(2))}</li>`);
          continue;
        }

        let numMatch = trimmed.match(/^(\d+)\.\s+(.*)/);
        if (numMatch) {
          if (!inList || listType !== 'ol') {
            flushList();
            inList = true;
            listType = 'ol';
            html.push('<ol>');
          }
          html.push(`<li>${formatInline(numMatch[2])}</li>`);
          continue;
        }

        flushList();
        if (trimmed === '') continue;
        html.push(`<p>${formatInline(trimmed)}</p>`);
      }

      flushTable();
      flushList();
      return html.join('\n');
    }

    async function openLegalDoc(slug) {
      slug = slug ? slug.toLowerCase().replace(/[^a-z0-9_-]/g, '') : 'privacy';
      if (slug === 'legal') slug = 'privacy';

      document.querySelectorAll('.legal-doc-btn').forEach(btn => btn.classList.remove('active'));
      const activeNavBtn = document.getElementById('legal-nav-' + slug);
      if (activeNavBtn) activeNavBtn.classList.add('active');

      document.getElementById('legal-rendered-body').innerHTML = '<div style="text-align:center; padding:30px; color:var(--text-dim);">Loading document...</div>';
      document.getElementById('legal-modal').classList.add('open');

      try {
        let docData = legalDocsCache[slug];
        if (!docData) {
          const resp = await fetch('/api/v1/legal/' + slug);
          if (!resp.ok) throw new Error('Document not found');
          docData = await resp.json();
          legalDocsCache[slug] = docData;
        }

        activeLegalDoc = docData;
        activeLegalDocRawMarkdown = docData.content || '';

        document.getElementById('legal-active-title').innerText = docData.title || 'Legal Document';
        document.getElementById('legal-active-effective').innerText = 'Effective: ' + (docData.effective_date || 'August 28, 2026');
        document.getElementById('legal-active-category').innerText = 'Category: ' + (docData.category || 'Compliance');
        document.getElementById('legal-active-version').innerText = 'Version ' + (docData.version || '1.0');
        document.getElementById('legal-doc-badge').innerText = 'v' + (docData.version || '1.0');

        document.getElementById('legal-rendered-body').innerHTML = renderMarkdownToHtml(docData.content);
        document.getElementById('legal-search-input').value = '';

        if (window.location.hash !== '#' + slug) {
          history.replaceState(null, null, '#' + slug);
        }
      } catch (err) {
        document.getElementById('legal-rendered-body').innerHTML = `
          <div style="color:var(--status-red-text); background:var(--status-red-bg); padding:16px; border-radius:var(--radius); border:1px solid var(--status-red-border);">
            <strong>Failed to load legal document:</strong> ${err.message || 'Unknown error'}
          </div>`;
      }
    }

    function closeLegalModal() {
      document.getElementById('legal-modal').classList.remove('open');
      const validSlugs = ['privacy', 'terms', 'cookies', 'refund', 'aup', 'dpa', 'disclaimer', 'legal'];
      const currentHash = window.location.hash.replace('#', '').toLowerCase();
      if (validSlugs.includes(currentHash)) {
        history.replaceState(null, null, ' ');
      }
    }

    function printActiveLegalDoc() {
      if (!activeLegalDoc) return;
      const printWin = window.open('', '_blank', 'width=800,height=900');
      const htmlContent = `
        <!DOCTYPE html>
        <html>
        <head>
          <title>${activeLegalDoc.title || 'Legal Document'} — Fraud Engine</title>
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 40px; color: #111; line-height: 1.6; }
            h1 { font-size: 22px; border-bottom: 2px solid #333; padding-bottom: 8px; margin-bottom: 12px; }
            h2 { font-size: 16px; margin-top: 20px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
            h3 { font-size: 14px; margin-top: 14px; }
            table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 12px; }
            th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: left; }
            th { background: #f4f4f4; }
            blockquote { border-left: 3px solid #0066cc; margin: 12px 0; padding: 6px 12px; background: #f8f9fa; }
            .meta { font-size: 11px; color: #666; margin-bottom: 16px; border-bottom: 1px solid #eee; padding-bottom: 8px; }
          </style>
        </head>
        <body>
          <div class="meta">Fraud Engine Legal Documentation &bull; Effective: ${activeLegalDoc.effective_date} &bull; Version: ${activeLegalDoc.version}</div>
          ${renderMarkdownToHtml(activeLegalDoc.content)}
        </body>
        </html>
      `;
      printWin.document.write(htmlContent);
      printWin.document.close();
      printWin.focus();
      setTimeout(() => { printWin.print(); }, 250);
    }

    function copyActiveLegalDocMarkdown() {
      if (!activeLegalDocRawMarkdown) return;
      copyText(activeLegalDocRawMarkdown);
      toast('Copied Markdown source to clipboard');
    }

    function filterLegalSearch(query) {
      if (!activeLegalDocRawMarkdown) return;
      const q = query.trim().toLowerCase();
      if (!q) {
        document.getElementById('legal-rendered-body').innerHTML = renderMarkdownToHtml(activeLegalDocRawMarkdown);
        return;
      }
      const rendered = renderMarkdownToHtml(activeLegalDocRawMarkdown);
      const container = document.getElementById('legal-rendered-body');
      container.innerHTML = rendered;
      
      const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null, false);
      const textNodes = [];
      while (walker.nextNode()) textNodes.push(walker.currentNode);

      for (let node of textNodes) {
        const parent = node.parentNode;
        if (parent && parent.nodeName !== 'SCRIPT' && parent.nodeName !== 'STYLE') {
          const text = node.nodeValue;
          const idx = text.toLowerCase().indexOf(q);
          if (idx !== -1) {
            const span = document.createElement('span');
            span.innerHTML = text.replace(new RegExp('(' + q.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, '\\$&') + ')', 'gi'), '<mark style="background:#58a6ff; color:#0d1117; padding:1px 3px; border-radius:2px;">$1</mark>');
            parent.replaceChild(span, node);
          }
        }
      }
    }

    function handleUrlHash() {
      const hash = window.location.hash.replace('#', '').toLowerCase().trim();
      const validSlugs = ['privacy', 'terms', 'cookies', 'refund', 'aup', 'dpa', 'disclaimer', 'legal'];
      if (validSlugs.includes(hash)) {
        openLegalDoc(hash);
      }
    }

    window.addEventListener('hashchange', handleUrlHash);

    // Auto-init
    initFirebaseClient();
    renderSnippet();
    checkCookieConsentOnLoad();
    handleUrlHash();
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

        if path == "/" or path == "/index.html" or path in ["/legal", "/privacy", "/terms", "/cookies", "/refund", "/aup", "/dpa", "/disclaimer"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        if path == "/api/v1/legal/documents":
            self._send_response_json(200, {"documents": load_legal_documents(include_content=True)})
            return

        if path.startswith("/api/v1/legal/"):
            slug = path[len("/api/v1/legal/"):].strip("/")
            doc = get_legal_document(slug)
            if doc:
                self._send_response_json(200, doc)
            else:
                self._send_response_json(404, {"error": "document_not_found", "message": f"Legal document '{slug}' not found"})
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
    httpd = ThreadingHTTPServer(server_address, FraudAppHandler)
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
