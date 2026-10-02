"""Dataset validation utilities."""

from __future__ import annotations

import pandas as pd

from complaint_triage.constants import ESCALATION_LABELS, TASK_TO_LABELS, TRIAGE_LABELS


def validate_label_file(df: pd.DataFrame) -> None:
    """Validate the minimum columns and label values for training."""
    required = ["Complaint ID", "clean_text", "triage_label", "escalation_label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Label file is missing required columns: {missing}")

    empty_text = df["clean_text"].fillna("").astype(str).str.strip().eq("")
    if empty_text.any():
        raise ValueError(f"Found {int(empty_text.sum())} rows with empty clean_text.")

    invalid_triage = sorted(set(df["triage_label"].dropna()) - set(TRIAGE_LABELS))
    if invalid_triage:
        raise ValueError(f"Invalid triage labels: {invalid_triage}")

    invalid_escalation = sorted(set(df["escalation_label"].dropna()) - set(ESCALATION_LABELS))
    if invalid_escalation:
        raise ValueError(f"Invalid escalation labels: {invalid_escalation}")


def validate_task(task: str) -> None:
    """Validate a supported modeling task."""
    if task not in TASK_TO_LABELS:
        raise ValueError(f"Unsupported task '{task}'. Expected one of {list(TASK_TO_LABELS)}.")
