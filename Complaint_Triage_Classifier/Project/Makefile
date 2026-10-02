RAW ?= data/raw/complaints.csv
LABELS ?= data/labeled/verified_labels.csv

.PHONY: help setup test lint check sample smoke profile prepare labels splits \
	baseline-triage baseline-escalation \
	train-distilbert-triage train-distilbert-escalation \
	train-roberta-triage train-roberta-escalation \
	evaluate-distilbert-triage evaluate-distilbert-escalation \
	evaluate-roberta-triage evaluate-roberta-escalation \
	error-analysis-distilbert-triage error-analysis-distilbert-escalation \
	crossval-triage crossval-escalation clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-36s\033[0m %s\n", $$1, $$2}'

# --- Setup and checks -------------------------------------------------------

setup:  ## Install the package with dev dependencies
	pip install -e ".[dev]"

test:  ## Run the test suite
	pytest

lint:  ## Run ruff
	ruff check src scripts tests

check: lint test  ## Lint and test

# --- Synthetic sample -------------------------------------------------------

sample:  ## Regenerate the synthetic sample dataset
	python scripts/make_sample_data.py

smoke:  ## Run the whole pipeline on the synthetic sample (plumbing check only)
	python scripts/01_prepare_data.py --input data/raw/complaints_sample.csv \
		--output data/interim/complaints_clean.csv \
		--profile-output reports/metrics/clean_profile.json
	python scripts/02_bootstrap_labels.py --input data/interim/complaints_clean.csv \
		--output data/labeled/bootstrap_labels.csv \
		--review-output data/labeled/manual_review_queue.csv --sample-size 100
	python scripts/03_make_splits.py --input data/labeled/bootstrap_labels.csv \
		--output-dir data/processed
	python scripts/04_train_baseline.py --task triage --output-dir models/baseline/triage

# --- Data pipeline ----------------------------------------------------------

profile:  ## Profile the raw CFPB CSV
	python scripts/00_profile_raw_data.py --input $(RAW) \
		--output reports/metrics/raw_profile.json

prepare:  ## Clean and filter narratives
	python scripts/01_prepare_data.py --input $(RAW) \
		--output data/interim/complaints_clean.csv \
		--profile-output reports/metrics/clean_profile.json

labels:  ## Create weak labels and the manual review queue
	python scripts/02_bootstrap_labels.py --input data/interim/complaints_clean.csv \
		--schema configs/label_schema.yaml \
		--output data/labeled/bootstrap_labels.csv \
		--review-output data/labeled/manual_review_queue.csv --sample-size 8000

splits:  ## Create train/validation/test splits from $(LABELS)
	python scripts/03_make_splits.py --input $(LABELS) --output-dir data/processed

# --- Baseline ---------------------------------------------------------------

baseline-triage:  ## Train the baseline for triage
	python scripts/04_train_baseline.py --task triage --output-dir models/baseline/triage

baseline-escalation:  ## Train the baseline for escalation
	python scripts/04_train_baseline.py --task escalation --output-dir models/baseline/escalation

# --- Transformer training ---------------------------------------------------

train-distilbert-triage:  ## Fine-tune DistilBERT for triage
	python scripts/05_train_transformer.py --task triage --config configs/distilbert.yaml \
		--output-dir models/distilbert/triage

train-distilbert-escalation:  ## Fine-tune DistilBERT for escalation
	python scripts/05_train_transformer.py --task escalation --config configs/distilbert.yaml \
		--output-dir models/distilbert/escalation

train-roberta-triage:  ## Fine-tune RoBERTa for triage
	python scripts/05_train_transformer.py --task triage --config configs/roberta.yaml \
		--output-dir models/roberta/triage

train-roberta-escalation:  ## Fine-tune RoBERTa for escalation
	python scripts/05_train_transformer.py --task escalation --config configs/roberta.yaml \
		--output-dir models/roberta/escalation

# --- Evaluation -------------------------------------------------------------

evaluate-distilbert-triage:  ## Evaluate DistilBERT on the triage test split
	python scripts/06_evaluate.py --task triage --model-dir models/distilbert/triage \
		--output-dir reports/metrics/distilbert_triage

evaluate-distilbert-escalation:  ## Evaluate DistilBERT on the escalation test split
	python scripts/06_evaluate.py --task escalation --model-dir models/distilbert/escalation \
		--output-dir reports/metrics/distilbert_escalation

evaluate-roberta-triage:  ## Evaluate RoBERTa on the triage test split
	python scripts/06_evaluate.py --task triage --model-dir models/roberta/triage \
		--output-dir reports/metrics/roberta_triage

evaluate-roberta-escalation:  ## Evaluate RoBERTa on the escalation test split
	python scripts/06_evaluate.py --task escalation --model-dir models/roberta/escalation \
		--output-dir reports/metrics/roberta_escalation

# --- Analysis ---------------------------------------------------------------

error-analysis-distilbert-triage:  ## Error analysis for DistilBERT triage
	python scripts/08_error_analysis.py \
		--predictions reports/metrics/distilbert_triage/predictions.csv \
		--output-dir reports/error_analysis/distilbert_triage

error-analysis-distilbert-escalation:  ## Error analysis for DistilBERT escalation
	python scripts/08_error_analysis.py \
		--predictions reports/metrics/distilbert_escalation/predictions.csv \
		--output-dir reports/error_analysis/distilbert_escalation

crossval-triage:  ## 3-fold cross-validation for the triage baseline
	python scripts/09_cross_validate.py --task triage --input $(LABELS) --folds 3 \
		--output reports/metrics/cv_triage_baseline.json

crossval-escalation:  ## 3-fold cross-validation for the escalation baseline
	python scripts/09_cross_validate.py --task escalation --input $(LABELS) --folds 3 \
		--output reports/metrics/cv_escalation_baseline.json

# --- Housekeeping -----------------------------------------------------------

clean:  ## Remove generated data, models, reports, and caches
	rm -rf models reports/metrics/* reports/figures/* reports/error_analysis/*
	rm -rf data/interim/* data/processed/* data/labeled/*.csv
	rm -rf .pytest_cache .ruff_cache src/*.egg-info *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	git checkout -- reports data 2>/dev/null || true
