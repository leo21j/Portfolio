"""Error-analysis utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Narrative word-count buckets used to check whether accuracy degrades on long
# complaints, which is where truncation would bite first.
LENGTH_BINS = [0, 50, 100, 200, 400, 10_000]
LENGTH_BIN_LABELS = ["0-50", "51-100", "101-200", "201-400", "401+"]


def _find_prediction_column(df: pd.DataFrame) -> str:
    """Locate the column holding predicted labels."""
    candidates = [col for col in df.columns if col.endswith("_prediction")]
    if not candidates:
        raise ValueError(
            "No prediction column found. Expected a column ending with '_prediction'."
        )
    if len(candidates) > 1:
        LOGGER.warning("Multiple prediction columns found (%s); using the first.", candidates)
    return candidates[0]


def _find_label_column(df: pd.DataFrame, prediction_column: str) -> str:
    """Infer the gold label column that matches a prediction column."""
    task = prediction_column.replace("_prediction", "")
    label_col = f"{task}_label"
    if label_col in df.columns:
        return label_col
    raise ValueError(
        f"Could not find the gold label column '{label_col}' for predictions "
        f"in '{prediction_column}'."
    )


def _word_lengths(df: pd.DataFrame) -> pd.Series | None:
    """Return narrative word counts, computing them from text if necessary.

    Predictions written by the pipeline carry ``text_word_len`` but not the
    narrative itself, so the precomputed column is preferred.
    """
    if "text_word_len" in df.columns:
        return df["text_word_len"]
    if "clean_text" in df.columns:
        return df["clean_text"].fillna("").astype(str).str.split().map(len)
    return None


def run_error_analysis(
    predictions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Create error-analysis CSV outputs from a predictions file.

    Writes ``all_errors.csv``, ``most_confused_pairs.csv``, and, when the input
    carries the relevant columns, ``accuracy_by_product.csv`` and
    ``accuracy_by_length_bucket.csv``.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_path, low_memory=False)
    pred_col = _find_prediction_column(df)
    label_col = _find_label_column(df, pred_col)

    df["is_correct"] = df[label_col].astype(str).eq(df[pred_col].astype(str))
    errors = df[~df["is_correct"]].copy()
    LOGGER.info("Found %s errors across %s rows.", len(errors), len(df))

    errors.to_csv(output_path / "all_errors.csv", index=False)

    confused_pairs = (
        errors.groupby([label_col, pred_col])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    confused_pairs.to_csv(output_path / "most_confused_pairs.csv", index=False)

    by_product = pd.DataFrame()
    if "Product" in df.columns:
        by_product = (
            df.groupby("Product")["is_correct"]
            .agg(["count", "mean"])
            .rename(columns={"mean": "accuracy"})
            .sort_values(["accuracy", "count"], ascending=[True, False])
            .reset_index()
        )
        by_product.to_csv(output_path / "accuracy_by_product.csv", index=False)

    by_length = pd.DataFrame()
    word_lengths = _word_lengths(df)
    if word_lengths is not None:
        working = df.copy()
        working["length_bucket"] = pd.cut(
            word_lengths,
            bins=LENGTH_BINS,
            labels=LENGTH_BIN_LABELS,
            include_lowest=True,
        )
        by_length = (
            working.groupby("length_bucket", observed=False)["is_correct"]
            .agg(["count", "mean"])
            .rename(columns={"mean": "accuracy"})
            .reset_index()
        )
        by_length.to_csv(output_path / "accuracy_by_length_bucket.csv", index=False)

    return {
        "errors": errors,
        "confused_pairs": confused_pairs,
        "by_product": by_product,
        "by_length": by_length,
    }
