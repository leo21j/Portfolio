"""Tests for embedding normalization.

Retrieval correctness depends on unit vectors: the index is a
``faiss.IndexFlatIP`` (inner product) and the reranker compares dot products
against a cosine threshold. Both are only meaningful on normalized input.
"""

import numpy as np

from pipeline.embedder import OllamaEmbedder


def test_normalize_produces_unit_vectors():
    vecs = np.array([[3.0, 4.0], [1.0, 0.0], [5.0, 12.0]], dtype=np.float32)
    out = OllamaEmbedder._l2_normalize(vecs)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_normalize_preserves_direction():
    vecs = np.array([[3.0, 4.0]], dtype=np.float32)
    out = OllamaEmbedder._l2_normalize(vecs)
    assert np.allclose(out, [[0.6, 0.8]])


def test_normalize_is_idempotent():
    # Ollama's embedding models already return unit vectors, so normalizing
    # must be a no-op on already-normalized input.
    vecs = np.array([[0.6, 0.8], [1.0, 0.0]], dtype=np.float32)
    once = OllamaEmbedder._l2_normalize(vecs)
    twice = OllamaEmbedder._l2_normalize(once)
    assert np.allclose(once, twice)


def test_zero_vector_does_not_divide_by_zero():
    vecs = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    out = OllamaEmbedder._l2_normalize(vecs)
    assert np.all(np.isfinite(out))
    assert np.allclose(out[0], [0.0, 0.0])


def test_inner_product_equals_cosine_after_normalizing():
    a = np.array([[2.0, 0.0]], dtype=np.float32)
    b = np.array([[1.0, 1.0]], dtype=np.float32)
    na = OllamaEmbedder._l2_normalize(a)[0]
    nb = OllamaEmbedder._l2_normalize(b)[0]
    cosine = float(np.dot(a[0], b[0]) / (np.linalg.norm(a[0]) * np.linalg.norm(b[0])))
    assert float(np.dot(na, nb)) == cosine
