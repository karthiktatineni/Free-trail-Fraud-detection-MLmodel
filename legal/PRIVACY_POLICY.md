---
slug: privacy
title: Privacy Policy
version: "1.0"
effective_date: "2026-08-28"
last_updated: "2026-08-28"
category: "Privacy & Data Protection"
summary: "Details our data collection, processing practices, retention schedules, subprocessors, and user rights."
---

# Privacy Policy

**Effective Date:** August 28, 2026  
**Last Updated:** August 28, 2026  
**Application / Service:** Fraud Engine Platform ("Fraud Engine", "we", "us", or "our")

---

## 1. Introduction

Fraud Engine provides real-time fraud risk scoring, velocity tracking, and syndicate graph analysis APIs for online businesses and developers. We respect your privacy and are committed to transparently describing how data is handled across our platform.

This Privacy Policy explains what information we collect from account holders ("Tenants" or "Developers") and end-user data submitted through our risk evaluation APIs ("Customer Event Payloads"), how we store and process that data, and the options available to you.

---

## 2. Information We Collect

### A. Account & Developer Profile Information
When you register for an account or manage your developer profile:
- **Contact Details:** Email address, display name, and organization metadata.
- **Authentication Data:** Firebase Authentication user identifiers (UIDs), email verification status, and session authentication tokens.
- **API Credentials:** Salted and hashed API keys (`fk_live_...` / `fk_test_...`). *We only store cryptographic SHA-256 hashes of API keys; raw secret keys are never retained in plaintext on our backend.*
- **Subscription & Usage Metadata:** Selected subscription tier, active API key count (maximum 3 keys per verified tenant), and rate limit utilization metrics.

### B. Customer Event Payloads (API Scored Data)
When a tenant submits a transaction or registration event to our inference endpoints (`POST /api/v1/score` or `POST /api/v1/batch`):
- **User Identity Fields:** Full name, email address, and event timestamps.
- **Network Metadata:** IPv4 / IPv6 addresses, Class-C subnet classifications, and billing city/country codes.
- **Device Identifiers:** Hardware/browser device fingerprint hashes and operating system labels.
- **Payment Tokens:** Tokenized payment references (e.g., `pm_visa_auth_8821`). *We never collect, store, or process raw Primary Account Numbers (PANs) or Card Verification Values (CVVs).*

### C. System Telemetry & Logs
- If performance analytics consent is granted, we collect client-side performance latencies, API error codes, and operational status logs (`/api/v1/client-logs`) to monitor system reliability.

---

## 3. How We Use and Process Data

We use the collected information to:
- Generate real-time risk scores (0.0 to 100.0) using trained machine learning models (XGBoost, LightGBM, Random Forest ensembles).
- Maintain causal feature stores, including 24-hour velocity counters and disjoint-set graph connectivity.
- Enforce plan-specific rate limits (e.g., 30 requests per minute sliding window) and tenant key quotas.
- Provide tenants with a private audit directory of scored customer records.
- Train and improve model accuracy through continuous learning scripts using aggregated, pseudo-anonymized data.

---

## 4. Third Parties & Subprocessors

We share data only with service providers strictly necessary to operate our infrastructure:

| Provider | Purpose | Data Handled | Location |
| :--- | :--- | :--- | :--- |
| **Google Cloud / Firebase** | Authentication & Cloud Firestore persistence | User profiles, audit records | USA / Global |
| **Redis Labs** | In-memory sliding velocity cache | Transient velocity keys | Multi-Region |
| **Cloud Hosting / Compute** | API runtime execution & load balancing | API traffic, telemetry | USA / EU |

We do not sell, rent, or trade personal data to third parties or data brokers.

---

## 5. Data Retention & Deletion

- **Tenant Account Data:** Retained for the duration of your active account.
- **Customer Audit Records:** Stored in tenant-isolated partitions and retained for operational dispute review, up to **90 days** by default, or until deleted by the tenant.
- **Sliding Velocity Caches:** In-memory velocity counts are transient and automatically evict entries older than **24 hours**.
- **Account Deletion:** Upon account closure, all associated API keys, profile data, and customer audit entries are queued for permanent deletion within **30 business days**.

---

## 6. Security Measures

We employ industry-standard technical and organizational security measures:
- Encryption in transit using TLS 1.3 for all HTTP/API communications.
- Cryptographic SHA-256 hashing for all API secret tokens.
- Strict tenant-level database isolation (`WHERE user_id = :tenant_id`).
- Granular role-based access control (RBAC) on internal administrative systems.

---

## 7. Your Data Rights

Depending on your jurisdiction and applicable data protection laws, you and your end-users may have the right to:
- Request access to personal data held by us.
- Request correction of inaccurate information.
- Request deletion of your personal data.
- Request an export of your account data in standard machine-readable format.
- Withdraw consent for optional telemetry and analytics at any time via cookie preferences.

To exercise these rights, please contact our privacy desk at **`privacy@fraudengine.io`**.

---

## 8. International Data Transfers

When data is transferred across international borders, we utilize recognized transfer mechanisms, including standard contractual clauses with our cloud infrastructure providers.

---

## 9. Contact Us

For privacy inquiries, data subject requests, or questions:
- **Email:** `privacy@fraudengine.io`
- **Address:** Fraud Engine Technologies, Inc., 1209 Orange Street, Wilmington, DE 19801, USA
