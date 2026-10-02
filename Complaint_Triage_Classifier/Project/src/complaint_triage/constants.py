"""Project-wide constants."""

from __future__ import annotations

RAW_REQUIRED_COLUMNS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company public response",
    "Company",
    "State",
    "ZIP code",
    "Tags",
    "Consumer consent provided?",
    "Submitted via",
    "Date sent to company",
    "Company response to consumer",
    "Timely response?",
    "Consumer disputed?",
    "Complaint ID",
]

TRIAGE_LABELS = [
    "fraud_identity_risk",
    "billing_payment_dispute",
    "account_access_management",
    "documentation_processing_verification",
    "customer_service_failure",
    "legal_compliance_regulatory",
]

# Human-readable names live in configs/label_schema.yaml and are read through
# LabelSchema.display_name(), so they are deliberately not duplicated here.

ESCALATION_LABELS = ["do_not_escalate", "escalate"]

TASK_TO_LABEL_COLUMN = {
    "triage": "triage_label",
    "escalation": "escalation_label",
}

TASK_TO_LABELS = {
    "triage": TRIAGE_LABELS,
    "escalation": ESCALATION_LABELS,
}

# Columns that quote the consumer or help identify them. They are stripped from
# anything written under reports/ so that generated artifacts can be shared or
# committed without leaking narratives. Derived length features are kept so
# error analysis still works.
SENSITIVE_COLUMNS = [
    "raw_text",
    "clean_text",
    "Consumer complaint narrative",
    "ZIP code",
    "State",
    "Company",
]
