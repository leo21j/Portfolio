"""Dataset conversion and tokenization helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from complaint_triage.constants import TASK_TO_LABEL_COLUMN, TASK_TO_LABELS
from complaint_triage.data.validate import validate_task


@dataclass(frozen=True)
class LabelMaps:
    label2id: dict[str, int]
    id2label: dict[int, str]


def get_label_maps(task: str) -> LabelMaps:
    """Return label maps for a task."""
    validate_task(task)
    labels = TASK_TO_LABELS[task]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return LabelMaps(label2id=label2id, id2label=id2label)


def dataframe_to_dataset(
    df: pd.DataFrame,
    task: str,
    text_column: str = "clean_text",
) -> Dataset:
    """Convert a pandas DataFrame to a Hugging Face Dataset."""
    validate_task(task)
    label_column = TASK_TO_LABEL_COLUMN[task]
    maps = get_label_maps(task)

    missing = [col for col in [text_column, label_column] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for task '{task}': {missing}")

    model_df = df[[text_column, label_column]].copy()
    model_df[text_column] = model_df[text_column].fillna("").astype(str)
    model_df["labels"] = model_df[label_column].map(maps.label2id)

    if model_df["labels"].isna().any():
        bad = sorted(model_df.loc[model_df["labels"].isna(), label_column].unique())
        raise ValueError(f"Unknown labels for task '{task}': {bad}")

    model_df["labels"] = model_df["labels"].astype(int)
    model_df = model_df.rename(columns={text_column: "text"})
    return Dataset.from_pandas(model_df[["text", "labels"]], preserve_index=False)


def tokenize_dataset(
    dataset: Dataset,
    tokenizer_name: str,
    max_length: int,
) -> tuple[Dataset, AutoTokenizer]:
    """Tokenize a Hugging Face Dataset, truncating at ``max_length``."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True)
    return tokenized, tokenizer
