# Customer Complaint Triage and Escalation Classifier

Text classification for consumer financial complaints. Given a free-text
complaint narrative, the system predicts which operational team should handle it
and whether it needs urgent specialist review.

Built on the public [CFPB Consumer Complaint
Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).


## 1. The problem

Organizations that handle complaints at volume have to route each one to the
right team, and spot the few that carry real urgency: suspected fraud, legal
exposure, or a consumer at risk of losing a home. Done by hand this is slow and
inconsistent, and the cost of missing an urgent case is much higher than the
cost of misrouting a routine one.

That asymmetry drives the whole design. Escalation is treated as a
cost-sensitive problem where recall on the urgent class matters more than
precision, rather than as a generic binary classifier scored on accuracy.

## 2. Two prediction tasks

**Task A — triage routing (6 classes).** Which queue should handle this?

| Label | Meaning |
|---|---|
| `fraud_identity_risk` | Fraud, scam, or identity risk |
| `billing_payment_dispute` | Billing or payment dispute |
| `account_access_management` | Account access, login, or account management |
| `documentation_processing_verification` | Documentation, processing, or verification issue |
| `customer_service_failure` | Customer service failure |
| `legal_compliance_regulatory` | Legal, compliance, or regulatory concern |

**Task B — escalation (binary).** Does this need urgent specialist review?

| Label | Meaning |
|---|---|
| `escalate` | Urgent human review |
| `do_not_escalate` | Normal triage queue |

These six categories are not CFPB's product taxonomy. CFPB labels complaints by
*financial product* ("Mortgage", "Credit card"), which says nothing about who
should act on it — a mortgage complaint might be a fraud case, a billing dispute,
or a legal threat. The categories above are cut by **operational action needed**
instead, because that is what a routing model has to predict to be useful.
Full definitions and tie-break rules are in
[`docs/annotation_guidelines.md`](docs/annotation_guidelines.md).

## 3. Approach

### Labels: rule-based bootstrap, then human review

CFPB publishes no triage or escalation labels, so they had to be created. Hand
labeling 10,000 narratives from scratch is not realistic, so the project uses
weak supervision as a first pass:

1. **Rule-based scoring.** A hand-curated schema
   ([`configs/label_schema.yaml`](configs/label_schema.yaml), ~340 lines) scores
   every category against narrative keywords plus CFPB metadata. Exact matches on
   the `Issue` field are weighted highest, since that taxonomy is already close to
   the triage categories; loose keyword hits are weighted lowest.
2. **Confidence, not just a label.** Each row gets a confidence score derived
   from the winning category's margin over the runner-up, so a narrative that
   matches two categories equally stays low-confidence instead of being
   silently assigned to one. Confidence is capped below 1.0 — a keyword rule
   should never express certainty.
3. **A review queue that prioritizes.** The queue
   ([`manual_review.py`](src/complaint_triage/labeling/manual_review.py))
   deliberately oversamples low-confidence rows and everything flagged for
   escalation, then balances across categories so no single frequent class
   fills it.

The rules exist to make human review tractable, not to replace it. Weak labels
are a starting point; `verified_labels.csv` is the training target.

### Models

| Model | Role |
|---|---|
| TF-IDF + logistic regression | Fast baseline. Word unigrams and bigrams, because escalation cues are short phrases ("legal action", "not mine") that unigrams would split. |
| DistilBERT | Efficient transformer, 256-token window |
| RoBERTa-base | Stronger transformer, 384-token window |

Both transformers train with class-weighted cross-entropy and early stopping on
validation macro F1.

### Evaluation

- **Triage:** macro F1 as the headline metric, so the rare categories count as
  much as the common ones, plus per-class breakdowns and a confusion matrix.
- **Escalation:** recall on `escalate` as the headline metric, plus PR-AUC and a
  [threshold sweep](src/complaint_triage/modeling/metrics.py) across 19
  operating points. A deployment would pick its threshold from that sweep based
  on how much reviewer capacity it has, not from the default 0.5.
- **Error analysis:** most-confused category pairs, accuracy by CFPB product,
  and accuracy by narrative length — the last one checks whether truncation is
  costing anything on long complaints.

The baseline and the transformers are fitted on the same training split by
default, so the comparison is not quietly skewed by data volume. (Fitting the
baseline on train+validation is available behind `--fit-on-train-val`, since the
transformers spend validation on early stopping and the baseline has no
equivalent use for it.)

## 4. Quickstart

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

A 120-row synthetic sample is committed, so the pipeline runs immediately
without downloading anything:

```bash
python scripts/01_prepare_data.py --input data/raw/complaints_sample.csv \
  --output data/interim/complaints_clean.csv \
  --profile-output reports/metrics/clean_profile.json
python scripts/02_bootstrap_labels.py --input data/interim/complaints_clean.csv \
  --output data/labeled/bootstrap_labels.csv \
  --review-output data/labeled/manual_review_queue.csv --sample-size 100
python scripts/03_make_splits.py --input data/labeled/bootstrap_labels.csv \
  --output-dir data/processed
python scripts/04_train_baseline.py --task triage --output-dir models/baseline/triage
```

The sample is for verifying that the plumbing works. It contains 30 distinct
hand-written narratives repeated across 120 rows, so a model will trivially
memorize it — **do not read any metric from it as a result.** See
[`scripts/make_sample_data.py`](scripts/make_sample_data.py) for how it is
generated.

Run the checks:

```bash
pytest
ruff check src scripts tests
```

## 5. Running on the real dataset

Download the CSV export from the [CFPB Consumer Complaint
Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
and save it to `data/raw/complaints.csv`. That path is gitignored; see
[Data handling](#8-data-handling) for why.

Only rows with a non-empty `Consumer complaint narrative` are usable, which is a
minority of the export — narratives are published only where the consumer
consented. Expected columns are documented in
[`docs/data_dictionary.md`](docs/data_dictionary.md).

| Step | Command | Output |
|---|---|---|
| 0. Profile | `make profile` | Column stats, narrative coverage |
| 1. Clean | `make prepare` | `data/interim/complaints_clean.csv` |
| 2. Weak labels | `make labels` | Weak labels + manual review queue |
| 3. **Review by hand** | — | Save corrections as `data/labeled/verified_labels.csv` |
| 4. Split | `make splits` | Stratified train/val/test in `data/processed/` |
| 5. Baseline | `make baseline-triage`, `make baseline-escalation` | Model, metrics, plots |
| 6. Fine-tune | `make train-distilbert-triage`, etc. | Transformer checkpoints |
| 7. Evaluate | `make evaluate-distilbert-triage`, etc. | Reports, confusion matrix, curves |
| 8. Error analysis | `make error-analysis-distilbert-triage` | Confusion pairs, accuracy breakdowns |

`make help` lists every target. Each script also takes explicit CLI arguments if
you want to override paths — see `python scripts/<name>.py --help`, or
[`docs/repo_walkthrough.md`](docs/repo_walkthrough.md) for the full commands.

Single-narrative inference:

```bash
python scripts/07_predict.py --model-dir models/distilbert/triage \
  --text "Someone opened a credit card in my name and I never applied for it."
```

```json
{
  "prediction": "fraud_identity_risk",
  "confidence": 0.93,
  "probabilities": { "fraud_identity_risk": 0.93, "...": "..." }
}
```

*(Output shape shown for illustration; the values depend on your trained model.)*

### GPU

Transformer training uses Hugging Face `Trainer`, which moves to CUDA
automatically when a CUDA-enabled PyTorch build is installed. Verify first:

```bash
python scripts/10_check_gpu.py
```

If it reports `cuda_available: false` on a machine with an NVIDIA GPU, the
environment has a CPU-only PyTorch build — the default on Windows from PyPI.
[`docs/gpu_setup.md`](docs/gpu_setup.md) has the fix.
[`infra/slurm/`](infra/slurm) holds the batch scripts used for training on an
HPC cluster; they assume a `.venv` in the project root.

## 6. Repository structure

```text
.
├── configs/
│   ├── distilbert.yaml           # DistilBERT hyperparameters
│   ├── roberta.yaml              # RoBERTa hyperparameters
│   └── label_schema.yaml         # Weak-label rules and category definitions
├── data/                         # All contents gitignored except the sample
│   ├── raw/complaints_sample.csv # Synthetic, committed, smoke-test only
│   ├── interim/                  # Cleaned narratives
│   ├── labeled/                  # Weak labels, review queue, verified labels
│   └── processed/                # Train/val/test splits
├── docs/
│   ├── problem_statement.md      # Business and ML framing
│   ├── annotation_guidelines.md  # Category definitions and tie-break rules
│   ├── data_dictionary.md        # Column reference
│   ├── experiment_plan.md        # Models, metrics, hyperparameter grid
│   ├── model_card.md             # Intended use and limitations
│   ├── gpu_setup.md              # CUDA setup for Windows
│   └── repo_walkthrough.md       # Script-by-script reference
├── infra/slurm/                  # HPC batch scripts
├── reports/                      # Generated metrics and figures (gitignored)
├── scripts/                      # 00-10, the numbered pipeline + make_sample_data
├── src/complaint_triage/
│   ├── data/                     # Loading, validation, splitting
│   ├── labeling/                 # Weak-label rules, review queue
│   ├── modeling/                 # Baseline, transformers, metrics, inference
│   ├── analysis/                 # Error analysis and plots
│   └── utils/                    # Config, logging, seeding, device checks
└── tests/
```

## 7. Built with

**ML/NLP:** PyTorch, Hugging Face Transformers + Datasets, scikit-learn
**Data:** pandas, NumPy, SciPy
**Plotting:** matplotlib
**Tooling:** pytest, ruff, Docker, GitHub Actions, SLURM

Design choices worth noting: configuration lives in YAML rather than in code, so
the label schema can be changed without touching Python; the training and
evaluation code inspects `TrainingArguments` signatures at runtime to tolerate
renamed Transformers arguments across versions; and the sequence length used
during training is saved with the checkpoint so evaluation cannot silently
tokenize at a different length.

## 8. Data handling

CFPB narratives are published with consumer consent, but the export still
carries state, ZIP code, exact dates, and company alongside the text. The repo
treats that as sensitive:

- **No real complaint data is committed.** `data/` is gitignored except for the
  synthetic sample. The only CSV in version control is machine-generated.
- **Generated reports are scrubbed.** `predictions.csv` and the error-analysis
  tables drop narratives, ZIP codes, state, and company before being written,
  keeping only `Complaint ID`, labels, probabilities, and derived length
  features. Error analysis still works, because length features survive the
  scrub. Override with `--keep-narratives` for local inspection only.
- **Narratives are not reproduced in documentation.** Every example in the docs
  and in the synthetic sample was written for this project.
- `reports/` and `models/` are gitignored in full.


## 10. Limitations

- **Label quality is the binding constraint.** Weak labels inherit whatever the
  keyword rules get wrong. Metrics computed against them measure rule agreement,
  not correctness — which is why they are not published here.
- **The categories are a judgment call.** Six operational buckets is a design
  decision, not a fact about the data. Some narratives genuinely span two, and
  the tie-break rules in the annotation guidelines resolve those by convention.
- **CFPB data is not representative.** Only consumers who complained to a federal
  regulator *and* consented to publication appear. Skewed toward certain
  products, companies, and demographics.
- **Escalation is the riskiest output.** A false negative means an urgent
  complaint sits in a normal queue. This should run as decision support with a
  human in the loop, never as an autonomous router.
- **Truncation.** At 256 and 384 tokens, long narratives lose their tails. The
  length-bucket error analysis exists to measure that cost.

## 11. Possible next steps

- Human-verify a few thousand rows and retrain to get defensible metrics
- Active learning: route the model's low-confidence predictions to review,
  rather than the rules' low-confidence ones
- Calibrate escalation probabilities and choose a threshold from reviewer capacity
- Per-class threshold tuning instead of a single argmax for triage
- Test whether CFPB metadata as model features beats text alone

## References

- [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [CFPB field reference](https://cfpb.github.io/api/ccdb/fields.html)
- [CFPB data-use and narrative publication policy](https://www.consumerfinance.gov/complaint/data-use/)
- [Hugging Face sequence classification guide](https://huggingface.co/docs/transformers/en/tasks/sequence_classification)
