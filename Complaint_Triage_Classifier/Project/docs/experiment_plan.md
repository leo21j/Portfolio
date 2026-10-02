# Experiment Plan

## Research question

Can a transformer-based text classifier accurately route consumer financial complaint narratives into operational triage categories and identify complaints requiring urgent human review?

## Models

### Baseline

- TF-IDF vectorizer
- Logistic Regression
- Class weights balanced

### Transformer models

- `distilbert-base-uncased`
- `FacebookAI/roberta-base`

## Training setup

- 70/15/15 train/validation/test split, stratified on the triage and escalation
  labels jointly when class counts permit
- Macro F1 as the primary triage metric, so rare categories count as much as
  common ones
- Recall on `escalate` as the primary escalation metric, since a false negative
  is more expensive than a false positive
- Early stopping on validation macro F1, patience 2
- Class-weighted cross-entropy for imbalance
- Sequence length 256 for DistilBERT and 384 for RoBERTa. The length used in
  training is written to `train_config.json` and read back at evaluation time,
  so the two cannot drift apart.

The baseline is fitted on the training split only, matching what the
transformers see. The transformers consume the validation split for early
stopping; the baseline has no equivalent use for it, so giving it train+validation
would make the comparison unequal on data volume.

## Hyperparameter candidates

| Parameter | Candidate values |
|---|---|
| Learning rate | 2e-5, 3e-5, 5e-5 |
| Batch size | 16, 32 |
| Epochs | 3, 4, 5 |
| Max sequence length | 256, 384 |

## Metrics

### Triage

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Weighted F1
- Per-class precision/recall/F1
- Confusion matrix

### Escalation

- Precision for `escalate`
- Recall for `escalate`
- F1 for `escalate`
- Macro F1
- PR-AUC
- ROC-AUC
- Threshold analysis

## Cross-validation

Use 3-fold cross-validation on the labeled dataset for the baseline model. This gives a quick stability check before expensive transformer training.

## Error analysis

Analyze:

- most confused class pairs,
- worst recall classes,
- likely majority-class bias,
- false negatives for `escalate`,
- performance by narrative length bucket,
- performance by CFPB product category,
- whether some labels should be merged or split.
