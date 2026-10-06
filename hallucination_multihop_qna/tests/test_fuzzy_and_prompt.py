"""Tests for fuzzy title matching (hop 2) and prompt construction."""

from types import SimpleNamespace

import pytest

from pipeline.data_loader import Passage
from pipeline.indexer import HybridRetriever
from pipeline.prompt_builder import PromptBuilder

# --- Fuzzy title matching -------------------------------------------------------


def _retriever(titles, threshold=70):
    # Built with __new__ so no embedding model or Ollama server is needed.
    r = HybridRetriever.__new__(HybridRetriever)
    r._passages = [Passage(title=t, sentences=[f"About {t}."]) for t in titles]
    r._fuzzy_title_threshold = threshold
    r._build_title_index()
    return r


def test_fuzzy_title_match_survives_surface_variation():
    r = _retriever(["Oh My God (Guns N' Roses song)", "End of Days (film)", "Shirley Temple"])
    hits = r._retrieve_by_title_fuzzy(["End of Days"])
    assert [h.passage.title for h in hits] == ["End of Days (film)"]
    assert hits[0].hop == 1


def test_fuzzy_title_match_respects_threshold():
    r = _retriever(["Shirley Temple"], threshold=95)
    assert r._retrieve_by_title_fuzzy(["Shirley Templeton Jones"]) == []


def test_each_title_is_returned_once_across_entities():
    r = _retriever(["Shirley Temple"])
    hits = r._retrieve_by_title_fuzzy(["Shirley Temple", "Temple Shirley"])
    assert len(hits) == 1


# --- Prompt builder ---------------------------------------------------------------


def _builder(max_chars=9000):
    cfg = SimpleNamespace(max_evidence_chars=max_chars)
    pb = PromptBuilder.__new__(PromptBuilder)
    pb.cfg = cfg
    pb.prompts = None
    return pb


def test_trimming_keeps_instructions_intact():
    pb = _builder(max_chars=300)
    instructions = "QUESTION: who?\nAVAILABLE FACTS:\nFact 0: x\nRespond with JSON."
    trimmed = pb._fit_evidence("Evidence:\n" + "e" * 1000, instructions)
    assert len(trimmed) + len(instructions) + 2 <= 300
    assert trimmed.endswith("[... truncated ...]")


def test_evidence_that_fits_is_untouched():
    pb = _builder()
    assert pb._fit_evidence("Evidence: short", "QUESTION: q") == "Evidence: short"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Were Scott Derrickson and Ed Wood of the same nationality?", True),
        ("Is the university located in the state capital?", True),
        ("Which writer was from England?", False),
        ("Who is older, Annie Morton or Terry Richardson?", False),
    ],
)
def test_yes_no_detection_is_syntactic(question, expected):
    assert _builder()._is_yesno_question(question) is expected


# --- Index cache consistency ------------------------------------------------------


def _write_cache(path, n_vectors, n_passages):
    import json

    import faiss
    import numpy as np

    from pipeline.indexer import BM25Retriever

    index = faiss.IndexFlatIP(4)
    index.add(np.eye(4, dtype=np.float32)[np.arange(n_vectors) % 4])
    faiss.write_index(index, str(path / "faiss.index"))
    (path / "dense_meta.json").write_text(json.dumps({"dim": 4, "model_name": "stub"}))
    passages = [{"title": f"T{i}", "sentences": [f"S{i}"]} for i in range(n_passages)]
    (path / "passages.json").write_text(json.dumps(passages))
    (path / "config.json").write_text(json.dumps({"alpha": 0.7, "alpha_bridge": 0.5, "rrf_k": 20}))
    bm25 = BM25Retriever()
    bm25.index([f"T{i} S{i}" for i in range(n_passages)])
    bm25.save(path)


def _loadable_retriever():
    from pipeline.indexer import BM25Retriever, DenseRetriever

    r = HybridRetriever.__new__(HybridRetriever)
    r.dense = DenseRetriever.__new__(DenseRetriever)
    r.bm25 = BM25Retriever()
    r.alpha, r.alpha_bridge, r.alpha_comparison, r.rrf_k = 0.7, 0.6, 0.5, 20
    return r


def test_cache_with_missing_vectors_is_rejected(tmp_path):
    _write_cache(tmp_path, n_vectors=3, n_passages=5)
    with pytest.raises(ValueError, match="inconsistent"):
        _loadable_retriever().load(tmp_path)


def test_cache_does_not_override_configured_fusion_weights(tmp_path):
    _write_cache(tmp_path, n_vectors=5, n_passages=5)
    r = _loadable_retriever()
    r.load(tmp_path)
    assert r.alpha_bridge == 0.6  # cache says 0.5; the config value wins
    assert len(r._title_list) == 5
