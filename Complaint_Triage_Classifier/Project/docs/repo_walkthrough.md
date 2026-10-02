# Repository Walkthrough

## Pipeline scripts

Run in order. Every script takes `--help`; paths default to the conventional
locations, so the flags below are only needed to override them.

| Script | Purpose | Key inputs → outputs |
|---|---|---|
| `00_profile_raw_data.py` | Inspect the raw CFPB CSV before cleaning | `complaints.csv` → `reports/metrics/raw_profile.json` |
| `01_prepare_data.py` | Filter to usable narratives, normalize text, add length features | `complaints.csv` → `data/interim/complaints_clean.csv` |
| `02_bootstrap_labels.py` | Apply weak-label rules, build the prioritized review queue | clean CSV → `bootstrap_labels.csv`, `manual_review_queue.csv` |
| `03_make_splits.py` | Stratified train/validation/test split | `verified_labels.csv` → `data/processed/{train,val,test}.csv` |
| `04_train_baseline.py` | TF-IDF + logistic regression | splits → `models/baseline/<task>/` |
| `05_train_transformer.py` | Fine-tune DistilBERT or RoBERTa | splits + config → `models/<model>/<task>/` |
| `06_evaluate.py` | Evaluate a checkpoint on the test split | model dir → `reports/metrics/<run>/` |
| `07_predict.py` | Single-narrative inference, prints JSON | model dir + `--text` → stdout |
| `08_error_analysis.py` | Confusion pairs, accuracy by product and length | `predictions.csv` → `reports/error_analysis/<run>/` |
| `09_cross_validate.py` | Stratified k-fold for the baseline | labeled CSV → metrics JSON |
| `10_check_gpu.py` | Report the PyTorch device and run a GPU smoke test | → stdout |
| `make_sample_data.py` | Regenerate the synthetic fixture | → `data/raw/complaints_sample.csv` |

`make help` wraps the common invocations.

## Full commands

### Profile and clean

```bash
python scripts/00_profile_raw_data.py \
  --input data/raw/complaints.csv \
  --output reports/metrics/raw_profile.json

python scripts/01_prepare_data.py \
  --input data/raw/complaints.csv \
  --output data/interim/complaints_clean.csv \
  --profile-output reports/metrics/clean_profile.json
```

Step 1 validates the required columns, keeps only non-empty narratives,
deduplicates on `Complaint ID`, builds `clean_text`, and adds character and word
length features.

### Weak labels and review

```bash
python scripts/02_bootstrap_labels.py \
  --input data/interim/complaints_clean.csv \
  --schema configs/label_schema.yaml \
  --output data/labeled/bootstrap_labels.csv \
  --review-output data/labeled/manual_review_queue.csv \
  --sample-size 8000
```

Review `manual_review_queue.csv` by hand, correcting `triage_label` and
`escalation_label`, then save the result as `data/labeled/verified_labels.csv`.
See [`annotation_guidelines.md`](annotation_guidelines.md) for the category
definitions and required columns.

### Splits

```bash
python scripts/03_make_splits.py \
  --input data/labeled/verified_labels.csv \
  --output-dir data/processed \
  --test-size 0.15 --val-size 0.15 --seed 42
```

Stratification uses the triage and escalation labels jointly, falling back to
triage alone, then to no stratification, when a stratum has fewer than two rows.

### Baseline

```bash
python scripts/04_train_baseline.py --task triage \
  --output-dir models/baseline/triage

python scripts/04_train_baseline.py --task escalation \
  --output-dir models/baseline/escalation
```

Fits on the training split only, matching what the transformers see. Add
`--fit-on-train-val` to fit on both.

### Transformers

```bash
python scripts/05_train_transformer.py --task triage \
  --config configs/distilbert.yaml --output-dir models/distilbert/triage

python scripts/05_train_transformer.py --task triage \
  --config configs/roberta.yaml --output-dir models/roberta/triage
```

Swap `--task escalation` for the binary task. Each run writes the checkpoint,
`label_maps.json`, `train_config.json` (including the sequence length, which
evaluation reads back), `eval_metrics.json`, and training curves.

### Evaluation and analysis

```bash
python scripts/06_evaluate.py --task triage \
  --model-dir models/distilbert/triage \
  --output-dir reports/metrics/distilbert_triage

python scripts/08_error_analysis.py \
  --predictions reports/metrics/distilbert_triage/predictions.csv \
  --output-dir reports/error_analysis/distilbert_triage
```

Evaluation writes `classification_report.{json,csv}`, `confusion_matrix.{csv,png}`,
and `predictions.csv`. For the escalation task it also writes
`threshold_sweep.csv`, `binary_curve_metrics.json`, `pr_curve.png`, and
`roc_curve.png`.

`predictions.csv` has narratives, ZIP codes, state, and company removed; pass
`--keep-narratives` to retain them for local inspection only.
