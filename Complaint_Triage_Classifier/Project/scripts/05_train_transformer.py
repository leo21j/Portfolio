"""Fine-tune a transformer classifier for one task."""

from __future__ import annotations

import argparse
import json

from complaint_triage.modeling.train_transformer import train_transformer
from complaint_triage.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer classifier.")
    parser.add_argument("--task", choices=["triage", "escalation"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--val", default="data/processed/val.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-column", default="clean_text")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_transformer(
        task=args.task,
        config_path=args.config,
        train_path=args.train,
        val_path=args.val,
        output_dir=args.output_dir,
        text_column=args.text_column,
        seed=args.seed,
    )
    logger.info("Wrote transformer model to %s", args.output_dir)
    logger.info("Eval metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
