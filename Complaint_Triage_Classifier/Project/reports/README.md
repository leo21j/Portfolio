# Reports

Generated metrics, plots, predictions, and error-analysis files are written
here. The contents are gitignored; only the directory placeholders are tracked.

Files produced by a full run:

| Path | Contents |
|---|---|
| `metrics/<run>/classification_report.{json,csv}` | Per-class and macro precision, recall, F1 |
| `metrics/<run>/confusion_matrix.{csv,png}` | Confusion matrix |
| `metrics/<run>/predictions.csv` | Per-row predictions and class probabilities |
| `metrics/<run>/threshold_sweep.csv` | Escalation metrics across 19 thresholds |
| `metrics/<run>/binary_curve_metrics.json` | Escalation PR-AUC and ROC-AUC |
| `metrics/<run>/{pr,roc}_curve.png` | Escalation curves |
| `error_analysis/<run>/most_confused_pairs.csv` | Which categories get mixed up |
| `error_analysis/<run>/accuracy_by_product.csv` | Accuracy by CFPB product |
| `error_analysis/<run>/accuracy_by_length_bucket.csv` | Accuracy by narrative length |
| `error_analysis/<run>/all_errors.csv` | Every misclassified row |

`predictions.csv` and `all_errors.csv` have narratives, ZIP codes, state, and
company stripped before writing, so they can be inspected and shared without
exposing complaint text. Derived length features are kept so the length-bucket
analysis still works. Pass `--keep-narratives` to retain the text for local
inspection, and do not commit the result.
