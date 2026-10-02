"""I/O utilities for CFPB complaint data."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from complaint_triage.constants import RAW_REQUIRED_COLUMNS, SENSITIVE_COLUMNS

LOGGER = logging.getLogger(__name__)

# The CFPB export is published as UTF-8, but re-saving it from a spreadsheet
# often introduces a BOM or a Windows codepage, so try the common variants.
_CANDIDATE_ENCODINGS = ("utf-8", "utf-8-sig", "latin1")


def read_complaints_csv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read the CFPB complaints CSV, trying the common encodings in turn.

    Malformed rows are skipped rather than aborting the read, because a single
    bad line in a multi-million-row export should not stop the pipeline. The
    number skipped is logged so the loss is visible instead of silent.

    Raises:
        ValueError: if the file cannot be decoded with any candidate encoding.
    """
    path = Path(path)
    failures: list[str] = []

    for encoding in _CANDIDATE_ENCODINGS:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", pd.errors.ParserWarning)
                df = pd.read_csv(
                    path,
                    on_bad_lines="warn",
                    nrows=nrows,
                    encoding=encoding,
                    low_memory=False,
                    dtype={"ZIP code": "string", "Complaint ID": "string"},
                )
        except UnicodeDecodeError as exc:
            failures.append(f"{encoding}: {exc}")
            continue

        skipped = sum(1 for w in caught if issubclass(w.category, pd.errors.ParserWarning))
        if skipped:
            LOGGER.warning(
                "Skipped %s malformed line(s) while reading %s. Row counts will be "
                "lower than the file's line count.",
                skipped,
                path,
            )
        LOGGER.info("Read %s rows from %s using %s.", len(df), path, encoding)
        return df

    raise ValueError(
        f"Unable to decode {path} with any of {list(_CANDIDATE_ENCODINGS)}. "
        + " | ".join(failures)
    )


def validate_columns(df: pd.DataFrame, required: Iterable[str] | None = None) -> None:
    """Validate that required CFPB columns exist."""
    required_cols = list(required or RAW_REQUIRED_COLUMNS)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + f". Available columns: {list(df.columns)}"
        )


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to CSV, creating parent directories as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def to_shareable_frame(df: pd.DataFrame, text_column: str = "clean_text") -> pd.DataFrame:
    """Drop narrative and location columns so the frame is safe to write out.

    Narrative length features are computed first, so downstream error analysis
    can still bucket by length without retaining the text itself. Use this for
    anything written under ``reports/``.
    """
    shareable = df.copy()

    if text_column in shareable.columns:
        text = shareable[text_column].fillna("").astype(str)
        if "text_char_len" not in shareable.columns:
            shareable["text_char_len"] = text.str.len()
        if "text_word_len" not in shareable.columns:
            shareable["text_word_len"] = text.str.split().map(len)

    dropped = [col for col in SENSITIVE_COLUMNS if col in shareable.columns]
    if dropped:
        LOGGER.info("Dropping %s from shareable output: %s", len(dropped), dropped)
    return shareable.drop(columns=dropped)


def dataset_profile(df: pd.DataFrame) -> dict:
    """Create a compact profile of a complaint DataFrame."""
    profile: dict = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_by_column": df.isna().sum().astype(int).to_dict(),
    }

    for col in ["Product", "Issue", "Company response to consumer", "Timely response?"]:
        if col in df.columns:
            profile[f"top_{col}"] = df[col].fillna("<MISSING>").value_counts().head(20).to_dict()

    if "Consumer complaint narrative" in df.columns:
        narrative = df["Consumer complaint narrative"].fillna("").astype(str)
        profile["non_empty_narratives"] = int(narrative.str.strip().ne("").sum())
        profile["narrative_char_len"] = {
            "mean": float(narrative.str.len().mean()),
            "median": float(narrative.str.len().median()),
            "p95": float(narrative.str.len().quantile(0.95)),
        }

    return profile
