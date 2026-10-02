"""Run stratified k-fold cross-validation for the baseline model."""

from __future__ import annotations

import argparse
import json

from complaint_triage.modeling.cross_validate import run_baseline_cross_validation
from complaint_triage.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline cross-validation.")
    parser.add_argument("--task", choices=["triage", "escalation"], required=True)
    parser.add_argument("--input", default="data/labeled/verified_labels.csv")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-column", default="clean_text")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_baseline_cross_validation(
        task=args.task,
        input_path=args.input,
        output_path=args.output,
        folds=args.folds,
        text_column=args.text_column,
        seed=args.seed,
    )
    logger.info("Cross-validation result: %s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
