"""Plotting utilities for reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve


def plot_confusion_matrix(
    y_true: np.ndarray | list[str],
    y_pred: np.ndarray | list[str],
    labels: list[str],
    output_path: str | Path,
) -> None:
    """Save a confusion matrix image."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        xticks_rotation=45,
        ax=ax,
        values_format="d",
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_training_curves(trainer_state_json: str | Path, output_path: str | Path) -> None:
    """Plot training/evaluation loss and metric curves from Trainer state."""
    state_path = Path(trainer_state_json)
    if not state_path.exists():
        return

    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    logs = state.get("log_history", [])
    if not logs:
        return

    df = pd.DataFrame(logs)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if "step" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    if "loss" in df.columns:
        train_df = df.dropna(subset=["loss"])
        ax.plot(train_df["step"], train_df["loss"], marker="o", label="train_loss")
    if "eval_loss" in df.columns:
        eval_df = df.dropna(subset=["eval_loss"])
        ax.plot(eval_df["step"], eval_df["eval_loss"], marker="o", label="eval_loss")
    if "eval_f1_macro" in df.columns:
        eval_df = df.dropna(subset=["eval_f1_macro"])
        ax.plot(eval_df["step"], eval_df["eval_f1_macro"], marker="o", label="eval_f1_macro")

    ax.set_xlabel("Training step")
    ax.set_title("Training and Evaluation Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_pr_curve(
    y_true_ids: np.ndarray,
    positive_scores: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save a precision-recall curve for binary escalation."""
    if len(np.unique(y_true_ids)) < 2:
        return

    precision, recall, _ = precision_recall_curve(y_true_ids, positive_scores)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_roc_curve(
    y_true_ids: np.ndarray,
    positive_scores: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save an ROC curve for binary escalation."""
    if len(np.unique(y_true_ids)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true_ids, positive_scores)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
