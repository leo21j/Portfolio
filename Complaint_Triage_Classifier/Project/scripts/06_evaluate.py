"""Evaluate a trained transformer model on a held-out test set."""

from __future__ import annotations

import argparse

from complaint_triage.modeling.evaluate_transformer import evaluate_transformer
from complaint_triage.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained transformer model.")
    parser.add_argument("--task", choices=["triage", "escalation"], required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-column", default="clean_text")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override the max sequence length. Defaults to the value saved at training time.",
    )
    parser.add_argument(
        "--keep-narratives",
        action="store_true",
        help=(
            "Keep narrative and location columns in predictions.csv. Off by default "
            "so the file is safe to share."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = evaluate_transformer(
        task=args.task,
        model_dir=args.model_dir,
        test_path=args.test,
        output_dir=args.output_dir,
        text_column=args.text_column,
        batch_size=args.batch_size,
        max_length=args.max_length,
        keep_narratives=args.keep_narratives,
    )
    logger.info("Wrote evaluation outputs to %s", args.output_dir)
    logger.info("Evaluated rows: %s", len(predictions))


if __name__ == "__main__":
    main()
