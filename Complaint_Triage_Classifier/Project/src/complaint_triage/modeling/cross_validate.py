"""Cross-validation for the classical baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from complaint_triage.constants import TASK_TO_LABEL_COLUMN
from complaint_triage.modeling.baseline import make_baseline_pipeline


def run_baseline_cross_validation(
    task: str,
    input_path: str | Path,
    output_path: str | Path,
    folds: int = 3,
    text_column: str = "clean_text",
    seed: int = 42,
) -> dict:
    """Run Stratified K-fold CV for the baseline model."""
    df = pd.read_csv(input_path, low_memory=False)
    label_column = TASK_TO_LABEL_COLUMN[task]
    X = df[text_column].fillna("").astype(str)
    y = df[label_column].astype(str)

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    model = make_baseline_pipeline()
    y_pred = cross_val_predict(model, X, y, cv=splitter, n_jobs=None)

    result = {
        "task": task,
        "folds": folds,
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
