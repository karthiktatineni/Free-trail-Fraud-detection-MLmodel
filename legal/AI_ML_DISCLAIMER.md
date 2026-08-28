---
slug: disclaimer
title: AI & ML Decision Disclaimer
version: "1.0"
effective_date: "2026-08-28"
last_updated: "2026-08-28"
category: "Model Risk & Decision Support"
summary: "Clarifies that fraud scores are probabilistic signals and provides operational guidance for human-in-the-loop review."
---

# AI & Machine Learning Decision Disclaimer

**Effective Date:** August 28, 2026  
**Last Updated:** August 28, 2026  
**Application:** Fraud Engine Risk Scoring Models, Algorithms & Decision Recommendations

---

## 1. Decision-Support Notice

**The fraud risk score is a decision-support signal and should not be treated as a definitive determination that a person, transaction, or account registration is fraudulent or illegitimate.**

Fraud Engine uses statistical machine learning models (including XGBoost, LightGBM, Random Forest classifiers, and heuristic graph clustering) to generate numerical risk scores (0.0 to 100.0) and recommended actions (ALLOW, REVIEW, BLOCK). These scores represent probabilistic estimations based on historical training data, behavioral velocity, and feature correlations.

---

## 2. Statistical Nature of Machine Learning Models

- **Probabilistic Outputs:** Machine learning models are statistical tools. An elevated risk score indicates an increased statistical similarity to historical abusive patterns, not guaranteed malicious intent.
- **Potential for Errors:** Like all automated classification systems, model inferences may produce false positives (flagging a legitimate user) or false negatives (failing to flag an abusive pattern).
- **Threshold Sensitivity:** The recommended action thresholds (e.g., ALLOW for &lt;5.5, REVIEW for 5.5–9.9, BLOCK for &ge;10.0) are default operational baselines. Tenants should calibrate thresholds to align with their business risk appetite.

---

## 3. Recommended Human-in-the-Loop Workflow

To maintain fair and accurate customer management, Fraud Engine strongly recommends:
1. **Tiered Decisioning:** Utilizing the intermediate "REVIEW" band to trigger step-up authentication (e.g., SMS verification, two-factor authentication, or manual agent review) rather than outright automated denial.
2. **Review Mechanisms:** Providing end-users with an avenue to dispute or request human re-evaluation of automated rejections where appropriate under applicable law or business policy.
3. **Continuous Monitoring:** Regularly monitoring population stability index (PSI) and feature drift metrics (available via `/api/v1/drift`) to detect shifts in incoming customer distributions.

---

## 4. Allocation of Responsibility

- **Final Decision Authority:** The tenant/customer maintains sole and exclusive authority over the final business decision (such as approving, holding, or cancelling an account or transaction).
- **Business Discretion:** Fraud Engine provides analytical signals to inform decisions; we do not execute unilateral account freezes or transaction cancellations on your behalf.

---

## 5. Regulatory Context & Applicability

The applicability of specific regulatory frameworks (including consumer protection, fair credit, or automated decision-making statutes) depends on the customer's jurisdiction, industry, and the nature of the decision being made. Customers are responsible for assessing their own legal obligations when integrating automated risk scoring into customer-facing decision workflows.

---

## 6. Questions & Model Governance

For inquiries regarding model explainability, feature weights, or evaluation metrics:
- **Email:** `ml-governance@fraudengine.io` / `legal@fraudengine.io`
- **Documentation:** Review technical benchmarks in the [Model Metrics Dashboard](file:///#model)
