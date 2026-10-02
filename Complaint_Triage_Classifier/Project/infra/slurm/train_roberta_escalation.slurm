#!/bin/bash
#SBATCH --job-name=complaint-escalation-roberta
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

module purge || true

source .venv/bin/activate

python scripts/05_train_transformer.py \
  --task escalation \
  --config configs/roberta.yaml \
  --train data/processed/train.csv \
  --val data/processed/val.csv \
  --output-dir models/roberta/escalation
