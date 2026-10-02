"""Label schema loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from complaint_triage.utils.config import load_yaml


@dataclass(frozen=True)
class LabelSchema:
    """Structured view of the YAML label schema."""

    triage_labels: dict[str, dict[str, Any]]
    escalation: dict[str, Any]
    weak_labeling: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> LabelSchema:
        data = load_yaml(path)
        return cls(
            triage_labels=data.get("triage_labels", {}),
            escalation=data.get("escalation", {}),
            weak_labeling=data.get("weak_labeling", {}),
        )

    @property
    def triage_label_ids(self) -> list[str]:
        return list(self.triage_labels.keys())

    def display_name(self, label_id: str) -> str:
        return self.triage_labels.get(label_id, {}).get("display_name", label_id)
