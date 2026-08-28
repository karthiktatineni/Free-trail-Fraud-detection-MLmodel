---
slug: cookies
title: Cookie Policy
version: "1.0"
effective_date: "2026-08-28"
last_updated: "2026-08-28"
category: "Privacy & Data Protection"
summary: "Explains how cookies and local storage tokens are used for authentication and preferences."
---

# Cookie Policy & Storage Technologies

**Effective Date:** August 28, 2026  
**Last Updated:** August 28, 2026  
**Application:** Fraud Engine Developer Dashboard ("Fraud Engine", "we", "us", or "our")

---

## 1. What Are Cookies and Local Storage?

Cookies and web storage technologies (`localStorage` and `sessionStorage`) are small data items placed on your browser when you access web applications. They allow our developer dashboard to maintain your authenticated session, remember your UI preferences, and store your cookie choices.

---

## 2. Categories of Storage We Use

We maintain a minimal storage footprint organized into two primary categories:

### A. Strictly Necessary & Essential Storage (Always Active)
Essential tokens are required for core platform functionality, security, and developer authentication:

| Storage Key / Cookie | Provider | Purpose | Duration |
| :--- | :--- | :--- | :--- |
| `firebase:authUser:...` | Firebase Auth | Maintains verified developer authentication state across browser sessions. | Persistent (until logout) |
| `fraud_engine_cookie_consent` | Fraud Engine | Stores your cookie consent configuration (Essential vs Analytics). | 1 Year |
| `session_state` | Fraud Engine | Preserves temporary form states during playground testing. | Session |

*Essential storage cannot be turned off because the developer portal cannot function securely without it.*

### B. Analytics & Performance Telemetry (Optional — Requires Consent)
These diagnostics help us measure UI rendering performance, API error rates, and latency:

| Storage Key / Cookie | Provider | Purpose | Duration |
| :--- | :--- | :--- | :--- |
| `_perf_telemetry` | Fraud Engine | Measures client-side inference latency and dashboard error logs (`POST /api/v1/client-logs`). | 30 Days |

*We do not use advertising trackers, marketing pixels, or third-party behavioral profiling cookies.*

---

## 3. Cookie Consent & Granular Management

When you visit the developer dashboard, you are presented with our **Cookie Consent Notice**:
- **Accept Analytics:** Permits optional performance and error logging.
- **Essential Only (Default):** Restricts storage strictly to authentication and security mechanisms. Non-essential telemetry is completely disabled.
- **Manage Preferences:** You can view and update your choices at any time via the "Cookie Preferences" link in the footer.

---

## 4. Browser-Level Controls

You can also manage or clear cookies through your browser settings (Chrome, Firefox, Safari, Edge). Note that clearing essential authentication tokens will log you out of your developer session.

---

## 5. Contact Us

If you have questions regarding our cookie practices:
- **Email:** `privacy@fraudengine.io`
- **Address:** Fraud Engine Technologies, Inc., 1209 Orange Street, Wilmington, DE 19801, USA
