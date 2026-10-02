"""Train/validation/test split helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from complaint_triage.data.validate import validate_label_file
from complaint_triage.io import write_csv


def _safe_stratify_labels(df: pd.DataFrame) -> pd.Series | None:
    """Return a safe stratification label or None when strata are too sparse."""
    combined = df["triage_label"].astype(str) + "__" + df["escalation_label"].astype(str)
    if combined.value_counts().min() >= 2:
        return combined

    if df["triage_label"].value_counts().min() >= 2:
        return df["triage_label"]

    return None


def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic train/validation/test splits."""
    validate_label_file(df)

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    stratify = _safe_stratify_labels(df)
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    # val_size is a fraction of the full dataset. Convert it to a fraction of train_val.
    relative_val_size = val_size / (1.0 - test_size)
    stratify_train_val = _safe_stratify_labels(train_val)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=stratify_train_val,
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save split CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_csv(train, output_path / "train.csv")
    write_csv(val, output_path / "val.csv")
    write_csv(test, output_path / "test.csv")


def split_label_file(
    input_path: str | Path,
    output_dir: str | Path,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load labels, split, and save train/validation/test files."""
    df = pd.read_csv(input_path, low_memory=False)
    train, val, test = make_splits(df, test_size=test_size, val_size=val_size, seed=seed)
    save_splits(train, val, test, output_dir)
    return train, val, test
