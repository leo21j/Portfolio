"""Text and tabular preprocessing for CFPB complaints."""

from __future__ import annotations

import re

import pandas as pd

from complaint_triage.constants import RAW_REQUIRED_COLUMNS
from complaint_triage.io import validate_columns

_RE_WHITESPACE = re.compile(r"\s+")
_RE_REDACTION = re.compile(r"\bX{2,}\b", re.IGNORECASE)
_RE_REPEATED_PUNCT = re.compile(r"([!?.,]){3,}")


def normalize_text(text: object, normalize_redactions: bool = True) -> str:
    """Normalize complaint narrative text without over-cleaning it."""
    if pd.isna(text):
        return ""

    value = str(text)
    value = value.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    value = _RE_WHITESPACE.sub(" ", value).strip()

    if normalize_redactions:
        value = _RE_REDACTION.sub("[REDACTED]", value)

    value = _RE_REPEATED_PUNCT.sub(r"\1\1", value)
    return value


def prepare_complaints(
    df: pd.DataFrame,
    min_text_chars: int = 40,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Prepare CFPB complaints for labeling and modeling."""
    validate_columns(df, RAW_REQUIRED_COLUMNS)

    prepared = df.copy()
    prepared["raw_text"] = prepared["Consumer complaint narrative"].fillna("").astype(str)
    prepared["clean_text"] = prepared["raw_text"].map(normalize_text)

    prepared = prepared[prepared["clean_text"].str.len() >= min_text_chars].copy()

    if drop_duplicates and "Complaint ID" in prepared.columns:
        prepared = prepared.drop_duplicates(subset=["Complaint ID"], keep="first")

    prepared["text_char_len"] = prepared["clean_text"].str.len()
    prepared["text_word_len"] = prepared["clean_text"].str.split().map(len)

    # Keep the canonical required columns first, then derived fields.
    ordered = [col for col in RAW_REQUIRED_COLUMNS if col in prepared.columns]
    derived = ["raw_text", "clean_text", "text_char_len", "text_word_len"]
    other = [col for col in prepared.columns if col not in ordered + derived]
    return prepared[ordered + other + derived].reset_index(drop=True)
