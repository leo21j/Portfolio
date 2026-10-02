import pytest

from services.job_matcher.src.scoring import combine_scores, normalize_scores, rank


def chunk(chunk_id, text="text"):
    return {"chunk_id": chunk_id, "text": text}


# --- normalize_scores ---


def test_normalize_scores_maps_into_unit_range():
    out = normalize_scores([0.1, 5.0, 12.0, 300.0])
    assert all(0.0 < value < 1.0 for value in out)


def test_normalize_scores_preserves_ordering():
    raw = [3.0, 1.0, 9.0, 5.0]
    out = normalize_scores(raw)
    # Ranking by normalized score must match ranking by raw score.
    assert sorted(range(len(raw)), key=lambda i: -raw[i]) == \
           sorted(range(len(out)), key=lambda i: -out[i])


def test_normalize_scores_handles_identical_scores():
    # No ranking information to recover, so everything collapses to the midpoint
    # rather than dividing by a zero standard deviation.
    assert normalize_scores([4.0, 4.0, 4.0]) == [0.5, 0.5, 0.5]


def test_normalize_scores_handles_empty_and_single():
    assert normalize_scores([]) == []
    assert normalize_scores([7.0]) == [0.5]


def test_normalize_scores_is_scale_invariant():
    # Standardizing first means a BM25 list in the hundreds and a cosine list
    # in [0, 1] land on the same scale, which is the whole point.
    assert normalize_scores([1.0, 2.0, 3.0]) == pytest.approx(
        normalize_scores([100.0, 200.0, 300.0])
    )


# --- combine_scores ---


def test_alpha_one_is_vector_only():
    vector = [(chunk("a"), 9.0)]
    bm25 = [(chunk("b"), 9.0)]
    combined = combine_scores(vector, bm25, [0.9], [0.9], alpha=1.0)
    assert combined["a"]["score"] == pytest.approx(0.9)
    assert combined["b"]["score"] == pytest.approx(0.0)


def test_alpha_zero_is_bm25_only():
    vector = [(chunk("a"), 9.0)]
    bm25 = [(chunk("b"), 9.0)]
    combined = combine_scores(vector, bm25, [0.9], [0.9], alpha=0.0)
    assert combined["a"]["score"] == pytest.approx(0.0)
    assert combined["b"]["score"] == pytest.approx(0.9)


def test_agreement_between_retrievers_outranks_a_single_strong_hit():
    # "b" is found by both retrievers at a middling score; "a" only by the
    # vector side at a high score. Consensus should win.
    vector = [(chunk("a"), 1.0), (chunk("b"), 0.5)]
    bm25 = [(chunk("b"), 0.5)]
    combined = combine_scores(vector, bm25, [0.9, 0.6], [0.6], alpha=0.5)
    assert combined["b"]["score"] > combined["a"]["score"]


def test_scores_from_both_retrievers_accumulate():
    vector = [(chunk("a"), 1.0)]
    bm25 = [(chunk("a"), 1.0)]
    combined = combine_scores(vector, bm25, [0.8], [0.4], alpha=0.5)
    assert combined["a"]["score"] == pytest.approx(0.5 * 0.8 + 0.5 * 0.4)


def test_chunk_found_by_only_one_retriever_is_still_returned():
    combined = combine_scores([(chunk("a"), 1.0)], [], [0.7], [], alpha=0.5)
    assert set(combined) == {"a"}


def test_mismatched_score_length_raises_rather_than_truncating():
    with pytest.raises(ValueError):
        combine_scores([(chunk("a"), 1.0), (chunk("b"), 1.0)], [], [0.5], [], alpha=0.5)


def test_alpha_outside_unit_range_is_rejected():
    with pytest.raises(ValueError):
        combine_scores([], [], [], [], alpha=1.5)


# --- rank ---


def test_rank_orders_by_descending_score():
    combined = {
        "a": {"chunk": chunk("a"), "score": 0.2},
        "b": {"chunk": chunk("b"), "score": 0.9},
        "c": {"chunk": chunk("c"), "score": 0.5},
    }
    assert [c["chunk_id"] for c, _ in rank(combined)] == ["b", "c", "a"]


def test_rank_of_empty_is_empty():
    assert rank({}) == []
