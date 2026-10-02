# Problem Statement

## Business problem

Large companies receive high volumes of free-text complaints, support tickets, and user reports. Manually routing these records is slow, inconsistent, and expensive. Delays are especially costly when the complaint suggests fraud, identity risk, legal exposure, regulatory concerns, or customer harm.

## Machine learning problem

Given a consumer complaint narrative, build a transformer-based classifier that predicts:

1. The operational triage category.
2. Whether the complaint should be escalated for urgent specialist review.

## Inputs

The primary input is the `Consumer complaint narrative` field from the CFPB Consumer Complaint Database CSV export. Metadata fields such as `Product`, `Sub-product`, `Issue`, `Sub-issue`, `Company response to consumer`, `Timely response?`, and `Tags` are used during weak-label bootstrapping and analysis, but the main model is trained on text.

## Outputs

### Multi-class triage output

- Fraud / scam / identity risk
- Billing / payment dispute
- Account access / login / account management
- Documentation / processing / verification issue
- Customer service failure
- Legal / compliance / regulatory concern

### Binary escalation output

- Escalate
- Do not escalate

## Success criteria

A successful model should:

- achieve strong macro F1 for triage classification,
- achieve high recall for the `escalate` class,
- outperform a TF-IDF + Logistic Regression baseline,
- provide interpretable error-analysis outputs,
- support reproducible training and evaluation from raw CSV to final metrics.

## Operational value

The classifier can support:

- faster complaint routing,
- reduced manual triage load,
- lower risk of misrouting,
- prioritization of high-risk complaints,
- better compliance and customer-risk analytics.
