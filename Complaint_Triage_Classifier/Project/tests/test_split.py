import pandas as pd

from complaint_triage.constants import TRIAGE_LABELS
from complaint_triage.data.split import make_splits


def test_make_splits_preserves_rows():
    rows = []
    for idx in range(60):
        rows.append(
            {
                "Complaint ID": str(idx),
                "clean_text": f"Example complaint text {idx} with enough words.",
                "triage_label": TRIAGE_LABELS[idx % len(TRIAGE_LABELS)],
                "escalation_label": "escalate" if idx % 3 == 0 else "do_not_escalate",
            }
        )
    df = pd.DataFrame(rows)
    train, val, test = make_splits(df, test_size=0.2, val_size=0.2, seed=123)
    assert len(train) + len(val) + len(test) == len(df)
    assert set(train["Complaint ID"]).isdisjoint(set(val["Complaint ID"]))
    assert set(train["Complaint ID"]).isdisjoint(set(test["Complaint ID"]))
