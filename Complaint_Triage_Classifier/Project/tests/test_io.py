import pandas as pd

from complaint_triage.constants import SENSITIVE_COLUMNS
from complaint_triage.io import to_shareable_frame, validate_columns


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Complaint ID": ["1", "2"],
            "Product": ["Credit card", "Mortgage"],
            "State": ["CA", "TX"],
            "ZIP code": ["94025", "750XX"],
            "Company": ["EXAMPLE BANK", "EXAMPLE SERVICER"],
            "raw_text": ["Someone opened an account.", "I was charged twice."],
            "clean_text": ["Someone opened an account.", "I was charged twice."],
            "triage_label": ["fraud_identity_risk", "billing_payment_dispute"],
            "triage_prediction": ["fraud_identity_risk", "customer_service_failure"],
        }
    )


def test_to_shareable_frame_drops_every_sensitive_column():
    out = to_shareable_frame(_prediction_frame())
    for column in SENSITIVE_COLUMNS:
        assert column not in out.columns


def test_to_shareable_frame_keeps_labels_and_identifiers():
    out = to_shareable_frame(_prediction_frame())
    for column in ["Complaint ID", "Product", "triage_label", "triage_prediction"]:
        assert column in out.columns


def test_to_shareable_frame_preserves_length_features_for_error_analysis():
    out = to_shareable_frame(_prediction_frame())
    assert out.loc[0, "text_word_len"] == 4
    assert out.loc[0, "text_char_len"] == len("Someone opened an account.")


def test_to_shareable_frame_does_not_mutate_the_input():
    df = _prediction_frame()
    to_shareable_frame(df)
    assert "clean_text" in df.columns


def test_validate_columns_reports_every_missing_column():
    df = pd.DataFrame({"Complaint ID": ["1"]})
    try:
        validate_columns(df, required=["Complaint ID", "Product", "Issue"])
    except ValueError as exc:
        assert "Product" in str(exc)
        assert "Issue" in str(exc)
    else:
        raise AssertionError("validate_columns should have raised")
