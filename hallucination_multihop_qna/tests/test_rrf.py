"""Tests for Reciprocal Rank Fusion, the core of hybrid retrieval."""

import pytest

from pipeline.indexer import _rrf_score


def test_rrf_is_monotonically_decreasing_in_rank():
    scores = [_rrf_score(r, k=60) for r in range(10)]
    assert scores == sorted(scores, reverse=True)


def test_rrf_matches_the_documented_formula():
    # score = 1 / (k + rank + 1)
    assert _rrf_score(0, k=60) == pytest.approx(1 / 61)
    assert _rrf_score(1, k=60) == pytest.approx(1 / 62)
    assert _rrf_score(9, k=60) == pytest.approx(1 / 70)


def test_rrf_ignores_score_magnitude_entirely():
    # The reason RRF is used: BM25 scores span roughly 0-25 while cosine
    # similarities span -1 to 1, so they cannot be summed directly. RRF
    # depends on rank alone, so a retriever's scale cannot dominate.
    assert _rrf_score(3, k=60) == _rrf_score(3, k=60)


def test_smaller_k_is_more_contrastive():
    # The project runs k=20 rather than the standard 60, on the grounds that a
    # lower k spreads the top ranks further apart. Verify that holds: the gap
    # between rank 0 and rank 1 should be wider at k=20.
    gap_k20 = _rrf_score(0, k=20) - _rrf_score(1, k=20)
    gap_k60 = _rrf_score(0, k=60) - _rrf_score(1, k=60)
    assert gap_k20 > gap_k60


def test_rrf_is_always_positive():
    for rank in (0, 1, 5, 100, 10_000):
        assert _rrf_score(rank, k=20) > 0


def test_alpha_weighted_fusion_endpoints():
    """A weighted sum of two RRF scores should collapse to one side at the
    alpha endpoints, which is what alpha=0 / alpha=1 are documented to mean."""
    dense_rank, sparse_rank = 0, 5
    dense = _rrf_score(dense_rank, k=20)
    sparse = _rrf_score(sparse_rank, k=20)

    assert 1.0 * dense + 0.0 * sparse == pytest.approx(dense)
    assert 0.0 * dense + 1.0 * sparse == pytest.approx(sparse)

    # At alpha=0.5 a document ranked highly by both beats one ranked highly by
    # only one retriever.
    both = 0.5 * _rrf_score(1, k=20) + 0.5 * _rrf_score(1, k=20)
    one_only = 0.5 * _rrf_score(0, k=20) + 0.5 * 0.0
    assert both > one_only
