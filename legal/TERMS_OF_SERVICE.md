---
slug: terms
title: Terms of Service
version: "1.0"
effective_date: "2026-08-28"
last_updated: "2026-08-28"
category: "Commercial & Legal"
summary: "Defines the terms, acceptable usage, API limits, intellectual property, and service conditions."
---

# Terms of Service

**Effective Date:** August 28, 2026  
**Last Updated:** August 28, 2026  
**Contracting Entity:** Fraud Engine Technologies, Inc. ("Fraud Engine", "we", "us", or "our")

---

## 1. Acceptance of Terms

By accessing or using the Fraud Engine dashboard, developer platform, API endpoints (`/api/v1/*`), or associated software tools (collectively, the "Service"), you agree to be bound by these Terms of Service ("Terms"). If you access the Service on behalf of a company or organization, you represent that you have authority to bind that entity.

---

## 2. Developer Accounts & Service Limitations

### A. Account Registration & Email Verification
- To access developer tools and generate API keys, you must create an account and verify your email address.
- You are responsible for safeguarding your login credentials and API keys.

### B. API Service Limitations & Rate Limits
- API requests are subject to the applicable rate limits associated with the customer's plan.
- On the standard developer plan, usage is limited to **30 requests per minute** per API key, calculated over a sliding 60-second window. Requests exceeding this threshold will receive HTTP status `429 Too Many Requests`.
- Verified developer accounts are permitted a maximum allocation of **3 active API keys** across live and test environments.
- Higher throughput and custom quotas are available under enterprise service agreements.

---

## 3. Intellectual Property Rights

- **Platform Ownership:** Fraud Engine and its licensors retain all intellectual property rights in the Service, underlying machine learning models (pipelines, feature transformers, weights), APIs, and dashboard software.
- **License to Use:** We grant you a limited, non-exclusive, non-transferable, revocable license to access the API and dashboard for your internal fraud prevention purposes in accordance with these Terms.
- **Restrictions:** You agree not to reverse engineer, decompile, extract model weights, resell API access, or use the Service to build a competing fraud evaluation platform.

---

## 4. Customer Data & Tenant Responsibilities

- **Lawful Data Submission:** You represent that all end-user data submitted to the API (e.g., email addresses, IP addresses, device identifiers, and tokenized payment tokens) has been collected in accordance with applicable laws and privacy disclosures.
- **No Raw Cardholder Data:** You must never submit unencrypted primary account numbers (PAN) or card verification values (CVV) to the Service. Only tokenized references (e.g., `pm_...`) may be submitted.
- **Tenant Scope:** You agree only to access customer records generated under your own authenticated account.

---

## 5. Machine Learning Output & Risk Scoring Disclaimers

- **Decision-Support Nature:** The fraud risk score (0.0 to 100.0) and associated verdicts (ALLOW, REVIEW, BLOCK) are probabilistic statistical estimations designed as decision-support signals. They do not constitute a definitive or legal determination that an individual or transaction is fraudulent.
- **Customer Discretion:** You retain sole responsibility for determining the final business action taken on any end-user or transaction, including implementing appropriate human review workflows for high-risk flags.

---

## 6. Service Availability & Maintenance

- We strive for high service availability and reliability. However, the Service is provided on an **"AS IS"** and **"AS AVAILABLE"** basis.
- We may perform routine maintenance, security patches, and model updates with reasonable notice where feasible.

---

## 7. Limitation of Liability

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL FRAUD ENGINE TECHNOLOGIES, INC., ITS DIRECTORS, EMPLOYEES, OR AFFILIATES BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES (INCLUDING LOSS OF PROFITS, DATA, OR BUSINESS INTERRUPTION) ARISING FROM OR RELATED TO YOUR USE OF THE SERVICE OR DEPENDENCE ON FRAUD SCORE OUTPUTS. OUR TOTAL CUMULATIVE LIABILITY UNDER THESE TERMS SHALL BE LIMITED TO THE GREATER OF THE FEES ACTUALLY PAID BY YOU TO FRAUD ENGINE IN THE TWELVE (12) MONTHS PRECEDING THE EVENT GIVING RISE TO LIABILITY, OR ONE HUNDRED DOLLARS ($100 USD).

---

## 8. Suspension & Termination

We reserve the right to suspend or terminate your API access if you violate these Terms, exceed rate limits systematically, engage in abusive traffic, or fail to pay applicable subscription fees. You may terminate your account at any time by closing your profile in the dashboard.

---

## 9. Governing Law & Dispute Resolution

These Terms are governed by the laws of the **State of Delaware, USA**, without regard to conflict of law rules. Any dispute arising under these Terms shall be resolved through binding arbitration in Wilmington, Delaware, or another mutually agreed forum.

---

## 10. Contact

For legal inquiries or notices regarding these Terms, contact **`legal@fraudengine.io`**.
