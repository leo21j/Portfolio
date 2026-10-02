#!/bin/bash
#SBATCH --job-name=complaint-triage-distilbert
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

module purge || true

source .venv/bin/activate

python scripts/05_train_transformer.py \
  --task triage \
  --config configs/distilbert.yaml \
  --train data/processed/train.csv \
  --val data/processed/val.csv \
  --output-dir models/distilbert/triage
