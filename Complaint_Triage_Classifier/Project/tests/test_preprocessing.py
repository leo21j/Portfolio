import pandas as pd

from complaint_triage.preprocessing import normalize_text, prepare_complaints

CREDIT_CARD_SUB_PRODUCT = "General-purpose credit card or charge card"


def _raw_frame() -> pd.DataFrame:
    """Two CFPB-shaped rows: one with a usable narrative, one without."""
    return pd.DataFrame(
        {
            "Date received": ["2024-01-01", "2024-01-02"],
            "Product": ["Credit card", "Credit card"],
            "Sub-product": [CREDIT_CARD_SUB_PRODUCT, CREDIT_CARD_SUB_PRODUCT],
            "Issue": ["Billing disputes", "Billing disputes"],
            "Sub-issue": ["", ""],
            "Consumer complaint narrative": [
                "This is a long enough complaint narrative about a billing issue.",
                "",
            ],
            "Company public response": ["", ""],
            "Company": ["A", "B"],
            "State": ["CA", "CA"],
            "ZIP code": ["900XX", "900XX"],
            "Tags": ["", ""],
            "Consumer consent provided?": ["Consent provided", "Consent provided"],
            "Submitted via": ["Web", "Web"],
            "Date sent to company": ["2024-01-02", "2024-01-03"],
            "Company response to consumer": [
                "Closed with explanation",
                "Closed with explanation",
            ],
            "Timely response?": ["Yes", "Yes"],
            "Consumer disputed?": ["", ""],
            "Complaint ID": ["1", "2"],
        }
    )


def test_normalize_text_collapses_whitespace_and_redactions():
    result = normalize_text("Hello\n\nXXXX    world!!!")
    assert result == "Hello [REDACTED] world!!"


def test_normalize_text_handles_nulls():
    assert normalize_text(None) == ""


def test_prepare_complaints_filters_empty_narratives():
    out = prepare_complaints(_raw_frame(), min_text_chars=10)
    assert len(out) == 1
    assert out.loc[0, "Complaint ID"] == "1"
    assert "clean_text" in out.columns


def test_prepare_complaints_adds_length_features():
    out = prepare_complaints(_raw_frame(), min_text_chars=10)
    assert out.loc[0, "text_word_len"] == 11
    assert out.loc[0, "text_char_len"] == len(out.loc[0, "clean_text"])


def test_prepare_complaints_drops_duplicate_complaint_ids():
    df = pd.concat([_raw_frame(), _raw_frame()], ignore_index=True)
    out = prepare_complaints(df, min_text_chars=10)
    assert len(out) == 1
