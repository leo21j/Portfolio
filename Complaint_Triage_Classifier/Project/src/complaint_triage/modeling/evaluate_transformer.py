"""Evaluation utilities for trained transformer classifiers."""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from complaint_triage.analysis.plots import plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from complaint_triage.constants import TASK_TO_LABEL_COLUMN, TASK_TO_LABELS
from complaint_triage.io import to_shareable_frame
from complaint_triage.modeling.datasets import dataframe_to_dataset
from complaint_triage.modeling.metrics import (
    binary_curve_metrics,
    confusion_matrix_df,
    make_classification_report,
    threshold_sweep,
)

LOGGER = logging.getLogger(__name__)

# Used only when a model directory predates train_config.json.
FALLBACK_MAX_LENGTH = 256


def _trainer_tokenizer_kwargs(tokenizer) -> dict:
    """Pass the tokenizer/processor using the argument supported by this Transformers version."""
    signature = inspect.signature(Trainer.__init__)
    parameters = signature.parameters

    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in parameters:
        return {"tokenizer": tokenizer}
    return {}


def _load_label_maps(model_dir: str | Path, task: str) -> tuple[dict[str, int], dict[int, str]]:
    """Load label maps from model output or defaults."""
    path = Path(model_dir) / "label_maps.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        label2id = {str(k): int(v) for k, v in data["label2id"].items()}
        id2label = {int(k): str(v) for k, v in data["id2label"].items()}
        return label2id, id2label

    labels = TASK_TO_LABELS[task]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def _resolve_max_length(model_dir: Path, override: int | None) -> int:
    """Pick the sequence length to tokenize with at evaluation time.

    Evaluating at a different length than training silently truncates inputs
    and depresses the reported metrics, so the length used during training is
    saved to ``train_config.json`` and reused here by default.
    """
    if override is not None:
        return override

    config_path = model_dir / "train_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            saved = json.load(f)
        max_length = saved.get("max_length")
        if max_length:
            return int(max_length)

    LOGGER.warning(
        "No train_config.json in %s, so the training sequence length is unknown. "
        "Falling back to %s. Pass --max-length if the model was trained with a "
        "different value.",
        model_dir,
        FALLBACK_MAX_LENGTH,
    )
    return FALLBACK_MAX_LENGTH


def _load_model(model_path: Path) -> AutoModelForSequenceClassification:
    """Load a saved model, preferring the eager attention backend.

    The eager backend is requested because the fused SDPA kernels have produced
    device-side index errors on some driver and Transformers combinations
    during batched evaluation.
    """
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            model_path,
            attn_implementation="eager",
        )
    except TypeError:
        # Older Transformers versions do not accept attn_implementation.
        return AutoModelForSequenceClassification.from_pretrained(model_path)


def _check_embedding_size(model, tokenizer) -> None:
    """Fail loudly if the tokenizer and the model's embedding table disagree.

    Resizing the embedding table here would randomly initialize the new rows
    and quietly corrupt a trained model, so a mismatch is raised instead. Note
    that an embedding table padded to a multiple of 8 or 64 is larger than the
    vocabulary by design, which is why only the undersized case is an error.
    """
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if embedding_size < len(tokenizer):
        raise ValueError(
            f"Tokenizer has {len(tokenizer)} tokens but the model's embedding table "
            f"holds only {embedding_size} rows. The tokenizer in this directory does "
            "not match the saved weights; re-save the model and tokenizer together."
        )


def evaluate_transformer(
    task: str,
    model_dir: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    text_column: str = "clean_text",
    batch_size: int = 32,
    max_length: int | None = None,
    keep_narratives: bool = False,
) -> pd.DataFrame:
    """Evaluate a saved transformer model on a test CSV.

    Args:
        task: Either ``triage`` or ``escalation``.
        model_dir: Directory written by ``train_transformer``.
        test_path: CSV holding the held-out test split.
        output_dir: Destination for metrics, plots, and predictions.
        text_column: Column holding the model input text.
        batch_size: Per-device evaluation batch size.
        max_length: Override the training sequence length. Defaults to the
            value saved alongside the model.
        keep_narratives: Keep narrative and location columns in
            ``predictions.csv``. Off by default so the file is safe to share.

    Returns:
        The per-row prediction frame that was written to ``predictions.csv``.
    """
    model_path = Path(model_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(test_path, low_memory=False)

    labels = TASK_TO_LABELS[task]
    label_column = TASK_TO_LABEL_COLUMN[task]
    label2id, id2label = _load_label_maps(model_path, task)

    dataset = dataframe_to_dataset(test_df, task=task, text_column=text_column)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = _load_model(model_path)
    _check_embedding_size(model, tokenizer)

    resolved_max_length = _resolve_max_length(model_path, max_length)
    LOGGER.info("Tokenizing evaluation data at max_length=%s", resolved_max_length)

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=resolved_max_length,
        )

    tokenized = dataset.map(tokenize, batched=True)

    eval_args = TrainingArguments(
        output_dir=str(output_path),
        per_device_eval_batch_size=batch_size,
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        **_trainer_tokenizer_kwargs(tokenizer),
    )

    predictions = trainer.predict(tokenized)
    probabilities = softmax(predictions.predictions, axis=1)
    pred_ids = np.argmax(probabilities, axis=1)

    y_true_labels = test_df[label_column].astype(str).to_numpy()
    y_pred_labels = np.array([id2label[int(idx)] for idx in pred_ids])

    report = make_classification_report(y_true_labels, y_pred_labels, labels=labels)
    with (output_path / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    pd.DataFrame(report).transpose().to_csv(output_path / "classification_report.csv")

    cm_df = confusion_matrix_df(y_true_labels, y_pred_labels, labels=labels)
    cm_df.to_csv(output_path / "confusion_matrix.csv")

    plot_confusion_matrix(
        y_true_labels,
        y_pred_labels,
        labels=labels,
        output_path=output_path / "confusion_matrix.png",
    )

    pred_df = test_df.copy()
    pred_df[f"{task}_prediction"] = y_pred_labels
    pred_df[f"{task}_correct"] = pred_df[label_column].astype(str).eq(
        pred_df[f"{task}_prediction"]
    )

    for idx, label in enumerate(labels):
        pred_df[f"prob_{label}"] = probabilities[:, idx]

    if not keep_narratives:
        pred_df = to_shareable_frame(pred_df, text_column=text_column)
    pred_df.to_csv(output_path / "predictions.csv", index=False)

    if task == "escalation":
        positive_id = label2id["escalate"]
        true_ids = np.array([label2id[label] for label in y_true_labels])
        positive_scores = probabilities[:, positive_id]

        curve_metrics = binary_curve_metrics(true_ids, positive_scores)
        with (output_path / "binary_curve_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(curve_metrics, f, indent=2)

        thresholds = threshold_sweep(true_ids, positive_scores)
        thresholds.to_csv(output_path / "threshold_sweep.csv", index=False)

        plot_pr_curve(true_ids, positive_scores, output_path / "pr_curve.png")
        plot_roc_curve(true_ids, positive_scores, output_path / "roc_curve.png")

    return pred_df
