"""Inference helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from complaint_triage.preprocessing import normalize_text


def predict_texts(model_dir: str | Path, texts: list[str]) -> list[dict]:
    """Predict labels and probabilities for one or more texts."""
    model_path = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    label_maps_path = model_path / "label_maps.json"
    if label_maps_path.exists():
        with label_maps_path.open("r", encoding="utf-8") as f:
            maps = json.load(f)
        id2label = {int(k): v for k, v in maps["id2label"].items()}
    else:
        id2label = model.config.id2label

    cleaned = [normalize_text(text) for text in texts]
    encoded = tokenizer(cleaned, padding=True, truncation=True, return_tensors="pt")

    outputs = model(**encoded)
    probs = softmax(outputs.logits.detach().numpy(), axis=1)
    pred_ids = np.argmax(probs, axis=1)

    results = []
    for text, pred_id, row_probs in zip(cleaned, pred_ids, probs, strict=True):
        probabilities = {
            id2label[int(idx)]: float(prob)
            for idx, prob in enumerate(row_probs)
        }
        results.append(
            {
                "text": text,
                "prediction": id2label[int(pred_id)],
                "confidence": float(row_probs[pred_id]),
                "probabilities": probabilities,
            }
        )

    return results
