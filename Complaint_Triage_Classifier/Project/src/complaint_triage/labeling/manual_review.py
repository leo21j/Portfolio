"""Manual review queue generation."""

from __future__ import annotations

import pandas as pd

from complaint_triage.constants import TRIAGE_LABELS


def build_manual_review_queue(
    df: pd.DataFrame,
    confidence_threshold: float = 0.72,
    sample_size: int | None = 8000,
    max_rows_per_label: int = 1500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Create a balanced manual review queue.

    The queue prioritizes:
    1. low-confidence triage labels,
    2. escalated complaints,
    3. balanced samples from every triage label.
    """
    required = ["triage_label", "triage_confidence", "escalation_label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required label columns for review queue: {missing}")

    low_confidence = df[df["triage_confidence"].astype(float) < confidence_threshold]
    escalated = df[df["escalation_label"].eq("escalate")]

    balanced_parts: list[pd.DataFrame] = []
    for label in TRIAGE_LABELS:
        label_rows = df[df["triage_label"].eq(label)]
        if len(label_rows) == 0:
            continue
        n = min(max_rows_per_label, len(label_rows))
        balanced_parts.append(label_rows.sample(n=n, random_state=random_seed))

    combined = pd.concat([low_confidence, escalated, *balanced_parts], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Complaint ID"], keep="first")

    if sample_size is not None and len(combined) > sample_size:
        # Preserve low-confidence and escalated rows first; sample the rest.
        priority_mask = (
            combined["triage_confidence"].astype(float).lt(confidence_threshold)
            | combined["escalation_label"].eq("escalate")
        )
        priority = combined[priority_mask]
        remainder = combined[~priority_mask]
        remaining_n = max(sample_size - len(priority), 0)

        if remaining_n > 0 and len(remainder) > 0:
            remainder = remainder.sample(
                n=min(remaining_n, len(remainder)),
                random_state=random_seed,
            )
            combined = pd.concat([priority, remainder], ignore_index=True)
        else:
            combined = priority.head(sample_size)

    review_columns = [
        "Complaint ID",
        "clean_text",
        "Product",
        "Sub-product",
        "Issue",
        "Sub-issue",
        "Company response to consumer",
        "Timely response?",
        "Tags",
        "triage_weak_label",
        "triage_confidence",
        "triage_rule_reason",
        "escalation_weak_label",
        "escalation_score",
        "escalation_rule_reason",
        "triage_label",
        "escalation_label",
    ]
    available_columns = [col for col in review_columns if col in combined.columns]
    combined = combined[available_columns].copy()
    combined["review_notes"] = ""
    return combined.reset_index(drop=True)
