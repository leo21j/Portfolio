"""Weak-label rules for escalation.

Escalation is scored additively from the narrative and CFPB metadata. A row
crosses into ``escalate`` at a score of 2, so a single keyword hit is not
enough on its own but a high-risk issue category is.
"""

from __future__ import annotations

import pandas as pd

from complaint_triage.labeling.schema import LabelSchema
from complaint_triage.labeling.text_rules import contains_phrase, join_fields, to_lower_text

# Fields searched for high-risk keywords.
ESCALATION_TEXT_COLUMNS = [
    "clean_text",
    "Issue",
    "Sub-issue",
    "Tags",
    "Company response to consumer",
    "Company public response",
    "Timely response?",
]

# A high-risk issue category is worth more than a keyword, because the consumer
# selected it from a fixed list rather than mentioning it in passing.
KEYWORD_POINTS = 1
ISSUE_POINTS = 2
COMPANY_RESPONSE_POINTS = 1
PUBLIC_RESPONSE_POINTS = 1
TAG_POINTS = 1
UNTIMELY_RESPONSE_POINTS = 1

ESCALATION_SCORE_THRESHOLD = 2


def escalation_score(row: pd.Series, schema: LabelSchema) -> tuple[int, list[str]]:
    """Compute an escalation risk score and the list of rules that fired."""
    spec = schema.escalation
    reasons: list[str] = []
    score = 0

    haystack = join_fields(row, ESCALATION_TEXT_COLUMNS)
    for keyword in spec.get("high_risk_keywords", []):
        if contains_phrase(haystack, keyword):
            score += KEYWORD_POINTS
            reasons.append(f"keyword:{keyword}")

    issue_text = to_lower_text(row.get("Issue", ""))
    sub_issue_text = to_lower_text(row.get("Sub-issue", ""))
    for issue in spec.get("high_risk_issues", []):
        issue_l = issue.lower()
        if issue_l in issue_text or issue_l in sub_issue_text:
            score += ISSUE_POINTS
            reasons.append(f"issue:{issue}")

    response = to_lower_text(row.get("Company response to consumer", ""))
    for item in spec.get("high_risk_company_responses", []):
        if item.lower() in response:
            score += COMPANY_RESPONSE_POINTS
            reasons.append(f"company_response:{item}")

    public_response = to_lower_text(row.get("Company public response", ""))
    for item in spec.get("high_risk_public_responses", []):
        if item.lower() in public_response:
            score += PUBLIC_RESPONSE_POINTS
            reasons.append(f"public_response:{item}")

    tags = to_lower_text(row.get("Tags", ""))
    for tag in spec.get("high_risk_tags", []):
        if tag.lower() in tags:
            score += TAG_POINTS
            reasons.append(f"tag:{tag}")

    if to_lower_text(row.get("Timely response?", "")) == "no":
        score += UNTIMELY_RESPONSE_POINTS
        reasons.append("timely_response:no")

    return score, reasons


def weak_escalation_label(row: pd.Series, schema: LabelSchema) -> tuple[str, int, str]:
    """Assign a candidate escalation label, its score, and the reason."""
    positive = schema.escalation.get("positive_label", "escalate")
    negative = schema.escalation.get("negative_label", "do_not_escalate")
    score, reasons = escalation_score(row, schema)

    label = positive if score >= ESCALATION_SCORE_THRESHOLD else negative
    return label, int(score), "; ".join(reasons) if reasons else "no_high_risk_rule_match"


def apply_weak_escalation_labels(df: pd.DataFrame, schema: LabelSchema) -> pd.DataFrame:
    """Add weak escalation labels, scores, and reasons to a DataFrame."""
    output = df.copy()
    results = output.apply(
        lambda row: weak_escalation_label(row, schema), axis=1, result_type="expand"
    )
    output["escalation_weak_label"] = results[0]
    output["escalation_score"] = results[1].astype(int)
    output["escalation_rule_reason"] = results[2]
    output["escalation_label"] = output["escalation_weak_label"]
    return output
