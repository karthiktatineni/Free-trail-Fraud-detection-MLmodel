---
slug: aup
title: Acceptable Use Policy
version: "1.0"
effective_date: "2026-08-28"
last_updated: "2026-08-28"
category: "Security & Platform Rules"
summary: "Defines prohibited activities, security testing guidelines, and API usage standards."
---

# Acceptable Use Policy (AUP)

**Effective Date:** August 28, 2026  
**Last Updated:** August 28, 2026  
**Application:** Fraud Engine API, Developer Portal & Services ("Fraud Engine", "we", "us", or "our")

---

## 1. Purpose & Scope

This Acceptable Use Policy ("AUP") defines rules and prohibited actions when interacting with the Fraud Engine API, developer dashboard, and related services. This policy protects platform availability, model integrity, and tenant security.

All registered users, API key holders, and developers must adhere strictly to this policy.

---

## 2. Prohibited Activities

When using the Service, you agree **not** to:

### A. API Abuse & Platform Disruption
- Exceed, bypass, or tamper with plan-specific rate limits (e.g., standard 30 req/min sliding-window) or key quotas (max 3 keys per verified tenant) using distributed proxies, botnets, or unauthorized concurrency.
- Launch denial-of-service (DoS/DDoS) attacks, flood requests, or stress-test production infrastructure without prior authorization.
- Interfere with other tenants' access or degrade shared multi-tenant infrastructure.

### B. Machine Learning & Model Extraction
- Attempt to reverse engineer, decompile, extract weights, or replicate the underlying machine learning models (XGBoost/LightGBM pipelines, feature transformers).
- Submit automated adversarial probes or poisoned payloads designed to corrupt continuous learning loops or degrade fraud detection efficacy for other tenants.
- Use API outputs or data to train or benchmark a competing fraud detection service.

### C. Fraudulent, Malicious, or Illegal Operations
- Submit stolen identity data, compromised payment credentials, or unauthorized personal records to the API.
- Use the platform to facilitate identity theft, carding syndicates, credential stuffing, account takeover (ATO), or illegal money laundering.
- Transmit malware, trojans, viruses, or harmful code via API event payloads.

### D. Security Vulnerability Exploitation
- Scan, probe, or test the vulnerability of the Service without written authorization under our responsible disclosure program.
- Breach authentication mechanisms, forge JWT/session tokens, or forge `X-API-Key` headers to access other tenants' private customer directories.

---

## 3. Responsible Security Disclosure

We welcome responsible security research. If you identify a security vulnerability:
- Email full details to **`security@fraudengine.io`**.
- Provide reasonable time for remediation prior to public disclosure.
- Do not access, modify, or destroy another tenant's data during testing.

---

## 4. Enforcement & Account Termination

- **Monitoring:** We monitor API traffic for anomalous velocity bursts, abnormal error rates, and policy violations.
- **Remediation:** Violations of this AUP may result in immediate rate limiting, API key revocation, account suspension, or permanent termination without notice or refund.
- **Legal Recourse:** We reserve the right to report unlawful activities to law enforcement and pursue appropriate civil remedies.

---

## 5. Contact Us

To report suspected abuse or inquire about permitted security testing:
- **Email:** `security@fraudengine.io` / `abuse@fraudengine.io`
- **Address:** Fraud Engine Technologies, Inc., 1209 Orange Street, Wilmington, DE 19801, USA
