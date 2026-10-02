"""Metric helpers for classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return core metrics for Trainer."""
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }


def make_classification_report(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Build a sklearn classification report dictionary."""
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )


def confusion_matrix_df(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    """Return confusion matrix as a labeled DataFrame."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        matrix,
        index=[f"true_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels],
    )


def binary_curve_metrics(y_true_ids: np.ndarray, positive_scores: np.ndarray) -> dict[str, float]:
    """Compute PR-AUC and ROC-AUC for binary escalation."""
    metrics: dict[str, float] = {}
    if len(np.unique(y_true_ids)) < 2:
        metrics["average_precision"] = float("nan")
        metrics["roc_auc"] = float("nan")
        return metrics

    metrics["average_precision"] = float(average_precision_score(y_true_ids, positive_scores))
    metrics["roc_auc"] = float(roc_auc_score(y_true_ids, positive_scores))
    return metrics


def threshold_sweep(
    y_true_ids: np.ndarray,
    positive_scores: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate binary metrics across thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    rows = []
    for threshold in thresholds:
        y_pred = (positive_scores >= threshold).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true_ids,
            y_pred,
            labels=[1],
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "precision_escalate": float(p),
                "recall_escalate": float(r),
                "f1_escalate": float(f1),
                "predicted_escalate_rate": float(y_pred.mean()),
            }
        )
    return pd.DataFrame(rows)
