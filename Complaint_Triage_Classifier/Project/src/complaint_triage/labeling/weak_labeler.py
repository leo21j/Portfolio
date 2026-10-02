"""Weak-label bootstrapping for triage categories.

Labels are assigned by scoring each category against keyword and CFPB metadata
rules defined in ``configs/label_schema.yaml``. The output is a starting point
for manual review, not ground truth.
"""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

from complaint_triage.labeling.schema import LabelSchema
from complaint_triage.labeling.text_rules import contains_phrase, join_fields, to_lower_text

# Fields concatenated into the haystack used for keyword matching.
TEXT_COLUMNS_FOR_RULES = [
    "clean_text",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Company public response",
    "Company response to consumer",
    "Tags",
]

# Rule weights. Exact issue matches are the strongest signal because the CFPB
# issue taxonomy is already close to the triage categories; keyword hits are
# weaker because narratives often mention several topics.
KEYWORD_WEIGHT = 2.0
ISSUE_EXACT_WEIGHT = 2.0
ISSUE_PARTIAL_WEIGHT = 1.0
PRODUCT_EXACT_WEIGHT = 1.0
PRODUCT_PARTIAL_WEIGHT = 0.5

# Label used when no rule fires at all. These rows get confidence 0.0, which
# puts them at the front of the manual review queue.
FALLBACK_TRIAGE_LABEL = "documentation_processing_verification"
NO_MATCH_REASON = "no_rule_match_default"


def score_triage_labels(row: pd.Series, schema: LabelSchema) -> dict[str, float]:
    """Score each triage label using keyword and metadata rules."""
    full_text = join_fields(row, TEXT_COLUMNS_FOR_RULES)
    product_text = to_lower_text(row.get("Product", ""))
    sub_product_text = to_lower_text(row.get("Sub-product", ""))
    issue_text = to_lower_text(row.get("Issue", ""))
    sub_issue_text = to_lower_text(row.get("Sub-issue", ""))

    scores: dict[str, float] = defaultdict(float)

    for label_id, spec in schema.triage_labels.items():
        for keyword in spec.get("keywords", []):
            if contains_phrase(full_text, keyword):
                scores[label_id] += KEYWORD_WEIGHT

        for product in spec.get("products", []):
            product_l = product.lower()
            if product_l in (product_text, sub_product_text):
                scores[label_id] += PRODUCT_EXACT_WEIGHT
            elif product_l in product_text or product_l in sub_product_text:
                scores[label_id] += PRODUCT_PARTIAL_WEIGHT

        for issue in spec.get("issues", []):
            issue_l = issue.lower()
            if issue_l in (issue_text, sub_issue_text):
                scores[label_id] += ISSUE_EXACT_WEIGHT
            elif issue_l in issue_text or issue_l in sub_issue_text:
                scores[label_id] += ISSUE_PARTIAL_WEIGHT

    return {label_id: float(scores.get(label_id, 0.0)) for label_id in schema.triage_label_ids}


def confidence_from_scores(scores: dict[str, float]) -> float:
    """Map rule scores to a confidence value in [0, 0.98].

    Confidence rises with both the absolute score of the winning label and its
    margin over the runner-up, so a row that matches two categories equally
    well stays low-confidence and gets reviewed. The logistic constants were
    chosen so that a single keyword hit with no competition lands near 0.4 and
    a strong unambiguous match approaches the 0.98 ceiling. The ceiling exists
    because a keyword rule should never express certainty.
    """
    if not scores:
        return 0.0

    ordered = sorted(scores.values(), reverse=True)
    best = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0.0

    if best <= 0:
        return 0.0

    margin = best - second
    confidence = 1.0 / (1.0 + math.exp(-(0.65 * best + 0.8 * margin - 2.0)))
    return round(float(min(max(confidence, 0.0), 0.98)), 4)


def weak_triage_label(row: pd.Series, schema: LabelSchema) -> tuple[str, float, str]:
    """Assign one candidate triage label, its confidence, and the reason."""
    scores = score_triage_labels(row, schema)

    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score <= 0:
        return FALLBACK_TRIAGE_LABEL, 0.0, NO_MATCH_REASON

    confidence = confidence_from_scores(scores)
    top_three = sorted(scores.items(), key=lambda item: -item[1])[:3]
    reason = "; ".join(f"{label}={score:g}" for label, score in top_three)
    return best_label, confidence, reason


def apply_weak_triage_labels(df: pd.DataFrame, schema: LabelSchema) -> pd.DataFrame:
    """Add weak triage labels, confidences, and reasons to a DataFrame."""
    output = df.copy()
    results = output.apply(lambda row: weak_triage_label(row, schema), axis=1, result_type="expand")
    output["triage_weak_label"] = results[0]
    output["triage_confidence"] = results[1].astype(float)
    output["triage_rule_reason"] = results[2]
    output["triage_label"] = output["triage_weak_label"]
    return output
