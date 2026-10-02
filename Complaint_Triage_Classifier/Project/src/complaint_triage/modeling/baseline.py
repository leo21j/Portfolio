"""TF-IDF + logistic regression baseline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from complaint_triage.analysis.plots import plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from complaint_triage.constants import TASK_TO_LABEL_COLUMN, TASK_TO_LABELS
from complaint_triage.io import to_shareable_frame
from complaint_triage.modeling.metrics import (
    binary_curve_metrics,
    confusion_matrix_df,
    threshold_sweep,
)

LOGGER = logging.getLogger(__name__)


def make_baseline_pipeline() -> Pipeline:
    """Build the TF-IDF + logistic regression pipeline.

    Word unigrams and bigrams are used because escalation cues are often short
    phrases ("legal action", "not mine") that unigrams alone would split.
    Class weights are balanced so the rare escalate class is not ignored.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=100_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    n_jobs=-1,
                    solver="saga",
                    penalty="l2",
                ),
            ),
        ]
    )


def train_baseline(
    task: str,
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    text_column: str = "clean_text",
    fit_on_train_val: bool = False,
    keep_narratives: bool = False,
) -> dict:
    """Train the baseline and evaluate it on the test split.

    Args:
        task: Either ``triage`` or ``escalation``.
        train_path: CSV holding the training split.
        val_path: CSV holding the validation split. Only read when
            ``fit_on_train_val`` is set.
        test_path: CSV holding the held-out test split.
        output_dir: Destination for the model, metrics, and plots.
        text_column: Column holding the model input text.
        fit_on_train_val: Fit on train+validation instead of train only.
            Defaults to False so the baseline sees exactly the same rows as the
            transformers, which keeps the headline comparison fair. The
            transformers spend the validation split on early stopping, so the
            baseline has no equivalent use for it.
        keep_narratives: Keep narrative and location columns in
            ``predictions.csv``. Off by default so the file is safe to share.

    Returns:
        The sklearn classification report as a dictionary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    if fit_on_train_val:
        val_df = pd.read_csv(val_path, low_memory=False)
        fit_df = pd.concat([train_df, val_df], ignore_index=True)
        LOGGER.info(
            "Fitting on train+validation (%s rows). This gives the baseline more "
            "data than the transformers received.",
            len(fit_df),
        )
    else:
        fit_df = train_df
        LOGGER.info("Fitting on the training split only (%s rows).", len(fit_df))

    labels = TASK_TO_LABELS[task]
    label_column = TASK_TO_LABEL_COLUMN[task]

    model = make_baseline_pipeline()
    model.fit(fit_df[text_column].fillna("").astype(str), fit_df[label_column].astype(str))

    y_true = test_df[label_column].astype(str).to_numpy()
    test_text = test_df[text_column].fillna("").astype(str)
    y_pred = model.predict(test_text)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_path / "classification_report.csv")
    with (output_path / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cm_df = confusion_matrix_df(y_true, y_pred, labels=labels)
    cm_df.to_csv(output_path / "confusion_matrix.csv")
    plot_confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        output_path=output_path / "confusion_matrix.png",
    )

    pred_df = test_df.copy()
    pred_df[f"{task}_prediction"] = y_pred
    pred_df[f"{task}_correct"] = pred_df[label_column].astype(str).eq(pred_df[f"{task}_prediction"])

    classifier = model.named_steps["classifier"]
    probabilities = model.predict_proba(test_text)
    class_order = list(classifier.classes_)
    for idx, label in enumerate(class_order):
        pred_df[f"prob_{label}"] = probabilities[:, idx]

    if task == "escalation" and "escalate" in class_order:
        positive_scores = probabilities[:, class_order.index("escalate")]
        label2id = {label: idx for idx, label in enumerate(labels)}
        true_ids = np.array([label2id[label] for label in y_true])

        curve_metrics = binary_curve_metrics(true_ids, positive_scores)
        with (output_path / "binary_curve_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(curve_metrics, f, indent=2)
        threshold_sweep(true_ids, positive_scores).to_csv(
            output_path / "threshold_sweep.csv",
            index=False,
        )
        plot_pr_curve(true_ids, positive_scores, output_path / "pr_curve.png")
        plot_roc_curve(true_ids, positive_scores, output_path / "roc_curve.png")

    if not keep_narratives:
        pred_df = to_shareable_frame(pred_df, text_column=text_column)
    pred_df.to_csv(output_path / "predictions.csv", index=False)

    joblib.dump(model, output_path / "model.joblib")
    return report
