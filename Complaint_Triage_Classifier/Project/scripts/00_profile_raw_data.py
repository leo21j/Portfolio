"""Profile the raw CFPB complaints CSV before any cleaning."""

from __future__ import annotations

import argparse

from complaint_triage.io import dataset_profile, read_complaints_csv
from complaint_triage.utils.config import save_json
from complaint_triage.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the raw CFPB complaints CSV.")
    parser.add_argument("--input", default="data/raw/complaints.csv")
    parser.add_argument("--output", default="reports/metrics/raw_profile.json")
    parser.add_argument("--nrows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_complaints_csv(args.input, nrows=args.nrows)
    profile = dataset_profile(df)
    save_json(profile, args.output)
    logger.info("Wrote raw data profile to %s", args.output)
    logger.info("Rows: %s | Columns: %s", profile["rows"], len(profile["columns"]))


if __name__ == "__main__":
    main()
