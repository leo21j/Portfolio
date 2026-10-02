"""Create deterministic train/validation/test splits from a labeled file."""

from __future__ import annotations

import argparse

from complaint_triage.data.split import split_label_file
from complaint_triage.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/validation/test splits.")
    parser.add_argument("--input", default="data/labeled/verified_labels.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, val, test = split_label_file(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    logger.info("Wrote splits to %s", args.output_dir)
    logger.info("Train=%s | Val=%s | Test=%s", len(train), len(val), len(test))


if __name__ == "__main__":
    main()
