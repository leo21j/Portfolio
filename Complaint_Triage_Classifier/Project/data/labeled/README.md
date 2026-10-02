# Labeled data

All CSVs in this directory are gitignored. They contain complaint narratives and
must not be committed.

| File | Produced by | Contents |
|---|---|---|
| `bootstrap_labels.csv` | `scripts/02_bootstrap_labels.py` | Every clean row with weak labels, confidence scores, and the rules that fired |
| `manual_review_queue.csv` | `scripts/02_bootstrap_labels.py` | Prioritized subset for human review |
| `verified_labels.csv` | You, by hand | Final training labels |

## Review workflow

`bootstrap_labels.csv` sets `triage_label` and `escalation_label` to the weak
rule output so the pipeline can run immediately. Those values are a starting
point, not ground truth.

`manual_review_queue.csv` is the file to actually work through. It oversamples
low-confidence rows and everything the rules flagged for escalation, then
balances across categories. Correct `triage_label` and `escalation_label`, use
the `review_notes` column for uncertain calls, and save the result as
`verified_labels.csv`.

`verified_labels.csv` must contain:

```text
Complaint ID
clean_text
triage_label
escalation_label
```

Extra columns are allowed and ignored. See
[`docs/annotation_guidelines.md`](../../docs/annotation_guidelines.md) for the
category definitions and tie-break rules.
