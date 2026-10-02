"""Run inference on one complaint narrative and print the result as JSON."""

from __future__ import annotations

import argparse
import json

from complaint_triage.modeling.predict import predict_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one complaint narrative.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = predict_texts(args.model_dir, [args.text])
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
