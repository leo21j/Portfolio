"""Transformer fine-tuning for triage and escalation."""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from complaint_triage.analysis.plots import plot_training_curves
from complaint_triage.constants import TASK_TO_LABEL_COLUMN, TASK_TO_LABELS
from complaint_triage.modeling.datasets import (
    dataframe_to_dataset,
    get_label_maps,
    tokenize_dataset,
)
from complaint_triage.modeling.metrics import compute_basic_metrics
from complaint_triage.utils.config import load_yaml
from complaint_triage.utils.device import get_torch_device_summary
from complaint_triage.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


class WeightedLossTrainer(Trainer):
    """Trainer with optional class-weighted cross entropy."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _trainer_tokenizer_kwargs(tokenizer) -> dict[str, Any]:
    """Pass the tokenizer/processor using the argument supported by this Transformers version."""
    signature = inspect.signature(Trainer.__init__)
    parameters = signature.parameters

    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in parameters:
        return {"tokenizer": tokenizer}
    return {}


def _training_args(output_dir: str | Path, training_cfg: dict[str, Any]) -> TrainingArguments:
    """Build TrainingArguments while supporting minor Transformers API changes."""
    signature = inspect.signature(TrainingArguments.__init__)
    parameters = signature.parameters

    force_cpu = bool(training_cfg.get("force_cpu", False))
    cuda_enabled = torch.cuda.is_available() and not force_cpu

    args: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": float(training_cfg.get("learning_rate", 3e-5)),
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 16)),
        "per_device_eval_batch_size": int(training_cfg.get("per_device_eval_batch_size", 32)),
        "num_train_epochs": float(training_cfg.get("num_train_epochs", 4)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.01)),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.06)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1)),
        "logging_steps": int(training_cfg.get("logging_steps", 50)),
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "load_best_model_at_end": True,
        "metric_for_best_model": training_cfg.get("metric_for_best_model", "f1_macro"),
        "greater_is_better": bool(training_cfg.get("greater_is_better", True)),
        "report_to": "none",
        "seed": int(training_cfg.get("seed", 42)),
    }

    # Transformers versions differ on eval_strategy vs evaluation_strategy.
    if "eval_strategy" in parameters:
        args["eval_strategy"] = "epoch"
    else:
        args["evaluation_strategy"] = "epoch"

    if "save_strategy" in parameters:
        args["save_strategy"] = "epoch"

    # Newer Transformers uses use_cpu; older versions use no_cuda.
    if "use_cpu" in parameters:
        args["use_cpu"] = force_cpu
    elif "no_cuda" in parameters:
        args["no_cuda"] = force_cpu

    if "fp16" in parameters:
        args["fp16"] = bool(training_cfg.get("fp16", False)) and cuda_enabled

    if "dataloader_pin_memory" in parameters:
        args["dataloader_pin_memory"] = cuda_enabled

    if "dataloader_num_workers" in parameters:
        args["dataloader_num_workers"] = int(training_cfg.get("dataloader_num_workers", 0))

    if "gradient_checkpointing" in parameters:
        args["gradient_checkpointing"] = bool(training_cfg.get("gradient_checkpointing", False))

    return TrainingArguments(**args)


def _compute_class_weights(labels: np.ndarray, num_labels: int) -> torch.Tensor:
    """Compute balanced class weights."""
    classes = np.arange(num_labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)


def train_transformer(
    task: str,
    config_path: str | Path,
    train_path: str | Path,
    val_path: str | Path,
    output_dir: str | Path,
    text_column: str = "clean_text",
    seed: int = 42,
) -> dict[str, float]:
    """Fine-tune a transformer model for one task."""
    set_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = load_yaml(config_path)
    model_cfg = config.get("model", {})
    # Copied so the loaded config is not mutated by the seed override.
    training_cfg = dict(config.get("training", {}))
    training_cfg["seed"] = seed

    device_summary = get_torch_device_summary(force_cpu=bool(training_cfg.get("force_cpu", False)))
    LOGGER.info("PyTorch device summary: %s", json.dumps(device_summary, indent=2))
    if device_summary["selected_device"] == "cpu":
        LOGGER.warning(
            "Training is running on CPU. If you expected GPU training, install a CUDA-enabled "
            "PyTorch build and verify with: python scripts/10_check_gpu.py"
        )

    model_name = model_cfg.get("name_or_path", "distilbert-base-uncased")
    max_length = int(model_cfg.get("max_length", 256))

    label_maps = get_label_maps(task)
    labels = TASK_TO_LABELS[task]
    label_column = TASK_TO_LABEL_COLUMN[task]

    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False)

    train_dataset = dataframe_to_dataset(train_df, task=task, text_column=text_column)
    val_dataset = dataframe_to_dataset(val_df, task=task, text_column=text_column)

    tokenized_train, tokenizer = tokenize_dataset(train_dataset, model_name, max_length=max_length)
    tokenized_val, _ = tokenize_dataset(val_dataset, model_name, max_length=max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={int(k): v for k, v in label_maps.id2label.items()},
        label2id=label_maps.label2id,
    )

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        return compute_basic_metrics(np.asarray(y_true), np.asarray(y_pred))

    class_weights = None
    if bool(training_cfg.get("use_class_weights", True)):
        y_train = train_df[label_column].map(label_maps.label2id).astype(int).to_numpy()
        class_weights = _compute_class_weights(y_train, num_labels=len(labels))

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=int(training_cfg.get("early_stopping_patience", 2))
        )
    ]

    trainer = WeightedLossTrainer(
        model=model,
        args=_training_args(output_path, training_cfg),
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        class_weights=class_weights,
        **_trainer_tokenizer_kwargs(tokenizer),
    )

    LOGGER.info("Hugging Face Trainer device: %s", trainer.args.device)
    trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    with (output_path / "label_maps.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": task,
                "labels": labels,
                "label2id": label_maps.label2id,
                "id2label": {str(k): v for k, v in label_maps.id2label.items()},
            },
            f,
            indent=2,
        )

    # Saved so evaluation tokenizes at the same sequence length it trained on.
    # Evaluating at a shorter length silently truncates inputs and depresses
    # every reported metric.
    with (output_path / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": task,
                "model_name_or_path": model_name,
                "max_length": max_length,
                "seed": seed,
                "text_column": text_column,
                "training": training_cfg,
            },
            f,
            indent=2,
        )

    with (output_path / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    plot_training_curves(output_path / "trainer_state.json", output_path / "training_curves.png")
    return {k: float(v) for k, v in metrics.items()}
