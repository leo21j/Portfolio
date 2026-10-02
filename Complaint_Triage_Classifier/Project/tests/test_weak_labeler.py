from pathlib import Path

import pandas as pd

from complaint_triage.labeling.escalation_rules import weak_escalation_label
from complaint_triage.labeling.schema import LabelSchema
from complaint_triage.labeling.text_rules import contains_phrase
from complaint_triage.labeling.weak_labeler import (
    FALLBACK_TRIAGE_LABEL,
    NO_MATCH_REASON,
    confidence_from_scores,
    weak_triage_label,
)

ROOT = Path(__file__).resolve().parents[1]


def make_schema() -> LabelSchema:
    return LabelSchema.from_yaml(ROOT / "configs" / "label_schema.yaml")


def test_weak_labeler_detects_fraud():
    row = pd.Series(
        {
            "clean_text": "Someone opened a fake account in my name and this is identity theft.",
            "Product": "Credit reporting",
            "Sub-product": "Credit reporting",
            "Issue": "Fraud or scam",
            "Sub-issue": "",
            "Company public response": "",
            "Company response to consumer": "",
            "Tags": "",
        }
    )
    label, confidence, reason = weak_triage_label(row, make_schema())
    assert label == "fraud_identity_risk"
    assert confidence > 0.5
    assert "fraud_identity_risk" in reason


def test_weak_labeler_falls_back_when_no_rule_fires():
    row = pd.Series({"clean_text": "zzz qqq", "Product": "", "Issue": ""})
    label, confidence, reason = weak_triage_label(row, make_schema())
    assert label == FALLBACK_TRIAGE_LABEL
    assert confidence == 0.0
    assert reason == NO_MATCH_REASON


def test_confidence_rises_with_margin():
    tight = confidence_from_scores({"a": 4.0, "b": 3.5})
    clear = confidence_from_scores({"a": 4.0, "b": 0.0})
    assert clear > tight


def test_confidence_is_capped_below_certainty():
    assert confidence_from_scores({"a": 1000.0, "b": 0.0}) <= 0.98


def test_escalation_detects_legal_threat():
    row = pd.Series(
        {
            "clean_text": "The collector threatened a lawsuit and legal action.",
            "Issue": "Took or threatened to take negative or legal action",
            "Sub-issue": "",
            "Tags": "",
            "Company response to consumer": "",
            "Company public response": "",
            "Timely response?": "Yes",
        }
    )
    label, score, reason = weak_escalation_label(row, make_schema())
    assert label == "escalate"
    assert score >= 2
    assert "legal action" in reason.lower() or "issue:" in reason.lower()


def test_escalation_stays_low_for_routine_complaint():
    row = pd.Series(
        {
            "clean_text": "I would like a copy of my monthly statement by mail.",
            "Issue": "Fees or interest",
            "Sub-issue": "",
            "Tags": "",
            "Company response to consumer": "Closed with explanation",
            "Company public response": "",
            "Timely response?": "Yes",
        }
    )
    label, score, _ = weak_escalation_label(row, make_schema())
    assert label == "do_not_escalate"
    assert score < 2


def test_contains_phrase_matches_at_word_start():
    assert contains_phrase("they threatened to sue me", "sue")
    assert contains_phrase("i submitted documents twice", "document")


def test_contains_phrase_allows_inflections():
    # A keyword list should catch ordinary word endings.
    assert contains_phrase("they charged me twice", "charge")
    assert contains_phrase("the fees were wrong", "fee")


def test_contains_phrase_does_not_match_mid_word():
    # "issue" appears in most narratives; it must not trigger the "sue" keyword.
    assert not contains_phrase("this is the issue i reported", "sue")
    assert not contains_phrase("i bought a cup of coffee", "fee")


def test_contains_phrase_handles_multi_word_and_empty():
    assert contains_phrase("i cannot access my account online", "cannot access")
    assert not contains_phrase("anything", "")
