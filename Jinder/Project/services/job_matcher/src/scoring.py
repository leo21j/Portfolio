"""Score normalization and fusion for hybrid retrieval.

Kept free of torch, faiss, and sentence-transformers imports so the fusion
logic can be tested without the model stack.
"""

from __future__ import annotations

import math

# Result rows are (chunk, score) pairs as returned by each retriever.
Result = tuple[dict, float]


def normalize_scores(scores: list[float]) -> list[float]:
    """Map scores onto 0-1 via z-score then logistic squashing.

    BM25 returns unbounded term-frequency sums while the vector index returns
    cosine similarities in a narrow band, so the two cannot be blended on their
    raw scales. Standardizing each list before squashing makes the weighted sum
    in :func:`combine_scores` meaningful.

    A z-score is used rather than min-max because min-max forces the worst
    result in every batch to exactly 0, which throws away the distinction
    between "weakest of a strong set" and "genuinely irrelevant".

    When every score is identical the standard deviation is zero and there is
    no ranking information to recover, so all results collapse to 0.5.
    """
    if not scores:
        return []

    mean = sum(scores) / len(scores)
    variance = sum((value - mean) ** 2 for value in scores) / len(scores)
    std = variance ** 0.5

    if std == 0:
        return [0.5] * len(scores)

    return [1 / (1 + math.exp(-((value - mean) / std))) for value in scores]


def combine_scores(
    vector_results: list[Result],
    bm25_results: list[Result],
    vector_scores: list[float],
    bm25_scores: list[float],
    alpha: float,
) -> dict[str, dict]:
    """Blend normalized vector and BM25 scores into one ranking.

    ``alpha`` weights the dense retriever: 1.0 is vector-only, 0.0 is BM25-only.
    A chunk returned by both retrievers accumulates both contributions, which is
    what lets agreement between the two outrank a strong hit from either one
    alone. A chunk found by only one retriever keeps just that side's share, so
    it is implicitly penalized against consensus results.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    combined: dict[str, dict] = {}

    for (chunk, _), score in zip(vector_results, vector_scores, strict=True):
        combined[chunk["chunk_id"]] = {"chunk": chunk, "score": alpha * score}

    for (chunk, _), score in zip(bm25_results, bm25_scores, strict=True):
        chunk_id = chunk["chunk_id"]
        weighted = (1 - alpha) * score
        if chunk_id in combined:
            combined[chunk_id]["score"] += weighted
        else:
            combined[chunk_id] = {"chunk": chunk, "score": weighted}

    return combined


def rank(combined: dict[str, dict]) -> list[Result]:
    """Flatten the fused map into a list ordered by descending score."""
    ordered = sorted(combined.values(), key=lambda item: item["score"], reverse=True)
    return [(item["chunk"], item["score"]) for item in ordered]
