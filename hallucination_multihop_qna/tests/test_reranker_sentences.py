"""Tests for supporting-sentence selection.

Sentence selection is what produces the HotpotQA supporting-fact predictions,
so a silent off-by-one here shows up directly as a lower SP-EM score.

The Reranker is built with __new__ and a stub cross-encoder so these run
without downloading the model.
"""

import numpy as np
import pytest

from pipeline.data_loader import Passage
from pipeline.reranker import Reranker, RerankResult


class _StubCrossEncoder:
    """Returns descending logits; the first sentence scores highest."""

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        return np.array([1.0 - 0.6 * i for i in range(len(pairs))], dtype=np.float32)


def _reranker(sentence_passage_limit=3, threshold=0.4, max_sents=5):
    r = Reranker.__new__(Reranker)
    r.sentence_score_threshold = threshold
    r.max_sentences_per_passage = max_sents
    r.sentence_passage_limit = sentence_passage_limit
    r.title_overlap_boost = 0.05
    r.batch_size = 32
    r._cross_encoder = _StubCrossEncoder()
    return r


def _result(title, sentences, rank):
    passage = Passage(title=title, sentences=list(sentences))
    return RerankResult(
        passage=passage,
        score=1.0 - rank * 0.1,
        rank=rank,
        retrieval_rank=rank,
        retrieval_score=0.5,
    )


def test_selects_sentences_without_raising_when_results_exceed_the_limit():
    """Regression: ranges are built for the first N passages only.

    Zipping them against the full result list must not raise, and must not
    mis-align a passage with another passage's sentence range.
    """
    results = [_result(f"T{i}", [f"Sentence {i}.{j}" for j in range(4)], i) for i in range(5)]
    _reranker(sentence_passage_limit=3)._select_supporting_sentences("q", results)

    # Only the first three passages get sentences.
    assert all(r.supporting_sentences for r in results[:3])
    assert all(not r.supporting_sentences for r in results[3:])


def test_selected_indices_point_into_the_right_passage():
    results = [_result(f"T{i}", [f"S{i}a", f"S{i}b", f"S{i}c"], i) for i in range(3)]
    _reranker(sentence_passage_limit=3)._select_supporting_sentences("q", results)

    for r in results:
        for sent, idx in zip(r.supporting_sentences, r.supporting_sentence_indices, strict=True):
            assert r.passage.sentences[idx] == sent, "index does not resolve to its own sentence"


def test_blank_sentences_do_not_shift_the_index_mapping():
    # Empty sentences are skipped when embedding, so the mapping back to the
    # original passage index has to account for the gap.
    results = [_result("T", ["First.", "   ", "Third.", "", "Fifth."], 0)]
    _reranker(sentence_passage_limit=1)._select_supporting_sentences("q", results)

    r = results[0]
    assert r.supporting_sentences, "expected at least one sentence"
    for sent, idx in zip(r.supporting_sentences, r.supporting_sentence_indices, strict=True):
        assert r.passage.sentences[idx] == sent
        assert sent.strip(), "a blank sentence should never be selected"


def test_indices_are_returned_in_passage_order():
    results = [_result("T", [f"S{j}" for j in range(5)], 0)]
    _reranker(sentence_passage_limit=1, threshold=-1.0)._select_supporting_sentences("q", results)
    idxs = results[0].supporting_sentence_indices
    assert idxs == sorted(idxs), "sentences should read in chronological order in the prompt"


def test_passage_with_no_sentences_is_handled():
    results = [_result("Empty", [], 0), _result("T", ["Only one."], 1)]
    _reranker(sentence_passage_limit=2)._select_supporting_sentences("q", results)
    assert results[0].supporting_sentences == []


def test_all_passages_empty_clears_every_result():
    results = [_result("A", [], 0), _result("B", ["  "], 1)]
    _reranker(sentence_passage_limit=2)._select_supporting_sentences("q", results)
    assert all(r.supporting_sentences == [] for r in results)


def test_max_sentences_per_passage_is_respected():
    results = [_result("T", [f"S{j}" for j in range(10)], 0)]
    # threshold=-1 so every sentence clears it; the cap must still apply.
    _reranker(sentence_passage_limit=1, threshold=-1.0, max_sents=3)._select_supporting_sentences("q", results)
    assert len(results[0].supporting_sentences) <= 3


def test_sentence_scores_are_squashed_to_unit_interval():
    scores = _reranker()._score_sentences("q", ["a", "b", "c", "d", "e"])
    assert np.all((scores > 0) & (scores < 1))
    assert list(scores) == sorted(scores, reverse=True)


def test_top_passages_get_a_two_sentence_minimum():
    # A threshold no sentence can clear: only the guaranteed minimum is kept.
    results = [_result(f"T{i}", [f"S{i}.{j}" for j in range(4)], i) for i in range(3)]
    _reranker(sentence_passage_limit=3, threshold=2.0)._select_supporting_sentences("q", results)
    assert [len(r.supporting_sentences) for r in results] == [2, 2, 1]


def test_empty_candidate_list_is_a_no_op():
    _reranker()._select_supporting_sentences("q", [])


if __name__ == "__main__":
    pytest.main([__file__])
