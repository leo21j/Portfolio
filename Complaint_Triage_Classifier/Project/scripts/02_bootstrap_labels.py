"""Apply the weak-label rules and build the manual review queue."""

from __future__ import annotations

import argparse

import pandas as pd

from complaint_triage.io import write_csv
from complaint_triage.labeling.escalation_rules import apply_weak_escalation_labels
from complaint_triage.labeling.manual_review import build_manual_review_queue
from complaint_triage.labeling.schema import LabelSchema
from complaint_triage.labeling.weak_labeler import apply_weak_triage_labels
from complaint_triage.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create weak labels and manual review queue.")
    parser.add_argument("--input", default="data/interim/complaints_clean.csv")
    parser.add_argument("--schema", default="configs/label_schema.yaml")
    parser.add_argument("--output", default="data/labeled/bootstrap_labels.csv")
    parser.add_argument("--review-output", default="data/labeled/manual_review_queue.csv")
    parser.add_argument("--sample-size", type=int, default=8000)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = LabelSchema.from_yaml(args.schema)
    df = pd.read_csv(args.input, low_memory=False)

    labeled = apply_weak_triage_labels(df, schema)
    labeled = apply_weak_escalation_labels(labeled, schema)

    # The training label columns start out as the weak labels. Correct them in
    # the review queue and save the result as verified_labels.csv.
    write_csv(labeled, args.output)

    threshold = (
        args.confidence_threshold
        if args.confidence_threshold is not None
        else float(schema.weak_labeling.get("review_confidence_threshold", 0.72))
    )
    review = build_manual_review_queue(
        labeled,
        confidence_threshold=threshold,
        sample_size=args.sample_size,
        max_rows_per_label=int(schema.weak_labeling.get("max_review_rows_per_label", 1500)),
        random_seed=args.seed,
    )
    write_csv(review, args.review_output)

    logger.info("Wrote weak labels to %s", args.output)
    logger.info("Wrote manual review queue to %s", args.review_output)
    logger.info("Triage distribution: %s", labeled["triage_label"].value_counts().to_dict())
    logger.info("Escalation distribution: %s", labeled["escalation_label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
