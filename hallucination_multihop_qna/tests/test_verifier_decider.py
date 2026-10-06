"""Tests for the post-generation verifier and decider.

Both run in "overlap" mode here, the mode used for the v39 run, so no model
download is needed.
"""

import math

import pytest

from pipeline.data_loader import Passage
from pipeline.decider import Decider
from pipeline.reranker import RerankResult
from pipeline.verifier import VerificationResult, Verifier


def _result(title, sentences, score=2.0, rank=0):
    return RerankResult(
        passage=Passage(title=title, sentences=list(sentences)),
        score=score,
        rank=rank,
        retrieval_rank=rank,
        retrieval_score=0.5,
        supporting_sentences=list(sentences),
        supporting_sentence_indices=list(range(len(sentences))),
    )


EVIDENCE = [
    _result("Shirley Temple", ["Shirley Temple served as Chief of Protocol of the United States."]),
    _result("Kiss and Tell (1945 film)", ["Kiss and Tell stars Shirley Temple as Corliss Archer."], rank=1),
]


# --- Verifier -----------------------------------------------------------------


def test_answer_found_in_evidence_is_supported():
    v = Verifier().verify("Chief of Protocol", EVIDENCE)
    assert v.is_supported
    # 0.75 * claim-token precision (1.0 here) + 0.25 * overlap F1, which the
    # longer evidence sentence keeps below 1.
    assert 0.75 < v.support_score < 1.0
    assert v.metadata["mode"] == "overlap"


def test_answer_absent_from_evidence_is_not_supported():
    v = Verifier().verify("Secretary of Agriculture", EVIDENCE)
    assert not v.is_supported
    assert v.unsupported_claims == ["Secretary of Agriculture"]


def test_empty_answer_and_missing_evidence_are_unsupported():
    assert Verifier().verify("", EVIDENCE).metadata["reason"] == "empty_answer"
    assert Verifier().verify("yes", []).metadata["reason"] == "no_evidence"


def test_yes_no_scored_on_evidence_quality_not_overlap():
    # "yes" shares no tokens with any evidence, so lexical overlap would score
    # it zero; the yes/no path scores the evidence behind it instead.
    v = Verifier().verify("yes", EVIDENCE, supporting_facts=[["Shirley Temple", 0], ["Kiss and Tell (1945 film)", 0]])
    assert v.is_supported
    assert v.support_score >= 0.75


def test_supporting_fact_coverage_counts_cited_facts_present_in_evidence():
    v = Verifier().verify("Chief of Protocol", EVIDENCE, supporting_facts=[["Shirley Temple", 0], ["Not Retrieved", 3]])
    assert v.metadata["supporting_fact_coverage"] == 0.5


# --- Decider ------------------------------------------------------------------


def _verification(supported, score):
    return VerificationResult(support_score=score, is_supported=supported)


def test_confidence_blends_verifier_and_reranker():
    decision, _ = Decider().decide("Chief of Protocol", _verification(True, 0.8), EVIDENCE)
    expected = 0.7 * 0.8 + 0.3 * (1 / (1 + math.exp(-2.0)))
    assert decision.confidence == pytest.approx(expected, abs=1e-4)
    assert decision.answer == "Chief of Protocol"


def test_unsupported_answer_is_kept_with_halved_confidence_by_default():
    supported, _ = Decider().decide("x", _verification(True, 0.2), EVIDENCE)
    unsupported, _ = Decider().decide("x", _verification(False, 0.2), EVIDENCE)
    assert unsupported.answer == "x"
    assert unsupported.confidence == pytest.approx(supported.confidence / 2, abs=1e-4)


def test_abstain_mode_replaces_unsupported_answers():
    decision, _ = Decider(abstain_on_unsupported=True).decide("x", _verification(False, 0.1), EVIDENCE)
    assert decision.answer == "Insufficient evidence"


def test_duplicate_supporting_facts_are_removed_in_order():
    _, attempt = Decider().decide(
        "x",
        _verification(True, 0.9),
        EVIDENCE,
        supporting_facts=[["Shirley Temple", 0], ["Shirley Temple", "0"], ["Kiss and Tell (1945 film)", 0]],
    )
    assert attempt["supporting_facts"] == [["Shirley Temple", 0], ["Kiss and Tell (1945 film)", 0]]


def test_retry_replaces_answer_when_it_becomes_supported():
    def retry():
        return {"answer": "Chief of Protocol", "supporting_facts": [["Shirley Temple", 0]]}

    decision, _ = Decider().decide(
        "Secretary of Agriculture",
        Verifier().verify("Secretary of Agriculture", EVIDENCE),
        EVIDENCE,
        verifier=Verifier(),
        retry_fn=retry,
    )
    assert decision.answer == "Chief of Protocol"
    assert decision.verifier_supported
