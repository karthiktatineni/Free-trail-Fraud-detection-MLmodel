---
slug: dpa
title: Data Processing Agreement (DPA)
version: "1.0"
effective_date: "2026-08-28"
last_updated: "2026-08-28"
category: "B2B Compliance & Data Protection"
summary: "Defines Controller and Processor responsibilities, security measures, subprocessors, and data transfer terms."
---

# Data Processing Agreement (DPA)

**Effective Date:** August 28, 2026  
**Last Updated:** August 28, 2026  
**Applicability:** Business-to-Business (B2B) Customers and API Tenants

---

## 1. Scope & Relationship of Parties

This Data Processing Agreement ("DPA") governs the processing of personal data submitted by business customers ("Customer" or "Controller") to Fraud Engine Technologies, Inc. ("Fraud Engine" or "Processor") in connection with the Fraud Engine API and fraud evaluation service.

- **Customer as Controller:** Customer determines the purposes and means of collecting end-user signup and transaction data submitted for risk evaluation.
- **Fraud Engine as Processor:** Fraud Engine processes Customer Personal Data solely on behalf of and in accordance with the Customer’s documented instructions and the [Terms of Service](file:///legal/TERMS_OF_SERVICE.md).

---

## 2. Categories of Data & Processing Details

- **Subject Matter:** Real-time fraud scoring, identity risk assessment, velocity tracking, and syndicate graph analysis.
- **Duration of Processing:** For the duration of the Customer’s subscription agreement plus applicable retention periods (up to 90 days for audit records, or until deletion).
- **Categories of Data Subjects:** Customer’s end-users, subscribers, and transaction participants.
- **Types of Personal Data Processed:** End-user names, email addresses, IP addresses, Class-C subnets, device fingerprint hashes, operating system labels, and tokenized payment identifiers (`pm_...`).

---

## 3. Obligations of the Processor (Fraud Engine)

Fraud Engine agrees to:
1. **Processing Instructions:** Process Customer Personal Data solely on documented instructions from Customer, unless required to do so by applicable law.
2. **Confidentiality:** Ensure that personnel authorized to process Customer Personal Data have committed themselves to confidentiality.
3. **Security Standards:** Implement appropriate technical and organizational measures (TOMs), including TLS 1.3 encryption in transit, SHA-256 key hashing, tenant database isolation, and role-based access control.
4. **Subprocessors:** Maintain a list of vetted infrastructure subprocessors (Google Cloud, Redis Labs, Cloud Hosting). Provide notice of material subprocessor changes where commercially feasible.
5. **Assistance with Data Subject Requests:** Provide technical capabilities allowing Customer to fulfill data subject access, rectification, and erasure requests.
6. **Data Breach Notification:** Notify Customer without undue delay upon becoming aware of a confirmed personal data breach affecting Customer Personal Data.
7. **Deletion or Return:** At the choice of Customer, delete or return all Customer Personal Data upon termination of services, subject to statutory retention obligations.

---

## 4. Obligations of the Customer (Controller)

Customer agrees to:
1. Provide lawful notices to data subjects regarding third-party fraud evaluation.
2. Maintain a lawful basis under applicable privacy legislation for submitting event data to the Processor.
3. Refrain from submitting unencrypted primary card numbers (PAN) or sensitive special category data to the API.

---

## 5. Subprocessor Overview

| Subprocessor | Role | Data Processed | Location |
| :--- | :--- | :--- | :--- |
| **Google Cloud / Firebase** | Cloud Hosting & Database | Customer audit records, authentication | USA / Global |
| **Redis Labs** | In-Memory Sliding Cache | Temporary velocity keys (24h eviction) | Multi-Region |
| **Cloud Hosting Provider** | Compute & Load Balancing | API request packets | USA / EU |

---

## 6. International Data Transfers

To the extent that processing involves transfers of personal data subject to European, UK, or Swiss data protection laws outside the EEA/UK to countries not recognized as providing an adequate level of data protection, the parties agree to abide by the Standard Contractual Clauses (SCCs) adopted by the European Commission.

---

## 7. Contact Information

For DPA execution inquiries or data protection officer communication:
- **Email:** `dpo@fraudengine.io` / `legal@fraudengine.io`
- **Address:** Fraud Engine Technologies, Inc., 1209 Orange Street, Wilmington, DE 19801, USA
