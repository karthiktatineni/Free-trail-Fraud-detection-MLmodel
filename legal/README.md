# Legal & Compliance Documentation

This directory contains the canonical, single-source-of-truth legal policies, terms, and compliance disclosures for the Fraud Engine SaaS platform.

## Document Directory

| Document | Slug | Category | Description |
| :--- | :--- | :--- | :--- |
| [Privacy Policy](file:///legal/PRIVACY_POLICY.md) | `privacy` | Privacy & Data Protection | Details data collection, processing, retention, subprocessors, and user rights. |
| [Terms of Service](file:///legal/TERMS_OF_SERVICE.md) | `terms` | Commercial & Legal | Defines terms, acceptable usage, API limits, IP ownership, and conditions. |
| [Cookie Policy](file:///legal/COOKIE_POLICY.md) | `cookies` | Privacy & Data Protection | Explains how cookies and local storage tokens are used for authentication and preferences. |
| [Refund & Cancellation Policy](file:///legal/REFUND_CANCELLATION_POLICY.md) | `refund` | Billing & Subscriptions | Details commercial billing cycles, self-service cancellation, and 14-day refund window. |
| [Acceptable Use Policy](file:///legal/ACCEPTABLE_USE_POLICY.md) | `aup` | Security & Platform Rules | Defines prohibited activities, security testing guidelines, and API usage standards. |
| [Data Processing Agreement](file:///legal/DATA_PROCESSING_AGREEMENT.md) | `dpa` | B2B Compliance & Data Protection | Defines Controller and Processor responsibilities, security measures, and subprocessors. |
| [AI & ML Decision Disclaimer](file:///legal/AI_ML_DISCLAIMER.md) | `disclaimer` | Model Risk & Decision Support | Clarifies that fraud scores are probabilistic signals and provides guidance for human review. |

---

## Architecture & Single Source of Truth

```text
legal/*.md (Markdown with YAML Frontmatter)
       │
       ▼
  Python Legal Loader (`load_legal_documents()`)
       │
       ├─► GET /api/v1/legal/documents   (List all documents + metadata)
       ├─► GET /api/v1/legal/{doc_slug}  (Fetch individual document content)
       │
       └─► Webpage Footer & Interactive Legal Center (Dynamic Modal & Direct URLs)
```

## Note on Legal Review
These documents describe the platform's actual technical implementation, data flows, and commercial policies. Organizations deploying this software commercially should have their legal counsel review all documents to ensure alignment with their specific operating jurisdictions.
