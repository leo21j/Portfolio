import os
import re

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dataclasses import dataclass, field

import numpy as np

from pipeline.data_loader import Passage
from pipeline.indexer import RetrievalResult
from scripts.config import get_best_device
from scripts.logger import get_logger

log = get_logger("reranker")


@dataclass
class RerankResult:
    passage: Passage
    score: float
    rank: int
    retrieval_rank: int
    retrieval_score: float
    supporting_sentences: list[str] = field(default_factory=list)
    supporting_sentence_indices: list[int] = field(default_factory=list)
    sentence_scores: list[float] = field(default_factory=list)
    hop: int = 0


class Reranker:
    """Cross-encoder reranking plus supporting-sentence selection.

    One cross-encoder does both jobs: it rescores (query, passage) pairs to
    pick the top passages, then scores (query, sentence) pairs inside those
    passages to pick the evidence sentences shown to the generator.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
        sentence_score_threshold: float = 0.25,
        max_sentences_per_passage: int = 5,
        batch_size: int = 32,
        sentence_passage_limit: int = 3,
        title_overlap_boost: float = 0.05,
    ):
        if device == "auto":
            device = get_best_device()
        self.model_name = model_name
        self.device = device
        self.sentence_score_threshold = sentence_score_threshold
        self.max_sentences_per_passage = max_sentences_per_passage
        self.batch_size = batch_size
        self.sentence_passage_limit = sentence_passage_limit
        self.title_overlap_boost = title_overlap_boost

        # Imported here rather than at module scope: sentence-transformers pulls
        # in the full transformers stack, and callers that only need the
        # RerankResult dataclass should not pay that import cost.
        from sentence_transformers import CrossEncoder

        log.info(f"Loading cross-encoder: {model_name}")
        self._cross_encoder = CrossEncoder(model_name, device=device)
        log.success("Reranker ready.")

    @classmethod
    def from_config(cls, cfg=None) -> "Reranker":
        from scripts.config import load_config

        if cfg is None or isinstance(cfg, str):
            cfg = load_config(cfg)
        r = cfg.reranker
        return cls(
            model_name=r.model_name,
            device=r.device,
            sentence_score_threshold=r.sentence_score_threshold,
            max_sentences_per_passage=r.max_sentences_per_passage,
            batch_size=r.batch_size,
            sentence_passage_limit=r.sentence_passage_limit,
            title_overlap_boost=r.title_overlap_boost,
        )

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
        select_sentences: bool = True,
    ) -> list[RerankResult]:
        if not candidates:
            return []

        pairs = [(query, c.passage.title_text) for c in candidates]

        log.step(f"Cross-encoder scoring {len(pairs)} passages...")
        scores: np.ndarray = self._cross_encoder.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        results = [
            RerankResult(
                passage=cand.passage,
                score=float(score),
                rank=0,
                retrieval_rank=cand.rank,
                retrieval_score=cand.score,
                hop=cand.hop,
            )
            for cand, score in zip(candidates, scores, strict=True)
        ]

        results.sort(key=lambda r: r.score, reverse=True)

        if top_k is not None:
            results = results[:top_k]

        for i, r in enumerate(results):
            r.rank = i

        if select_sentences:
            self._select_supporting_sentences(query, results)

        return results

    def _normalize_tokens(self, text: str) -> set:
        return {re.sub(r"[^a-z0-9]+", "", t.lower()) for t in text.split() if re.sub(r"[^a-z0-9]+", "", t.lower())}

    def _score_sentences(self, query: str, sentences: list[str]) -> np.ndarray:
        """Cross-encoder relevance of each sentence to the query, squashed to [0, 1].

        Raw cross-encoder outputs are unbounded logits; the sigmoid makes the
        fixed `sentence_score_threshold` meaningful.
        """
        pairs = [(query, sent) for sent in sentences]
        log.step(f"Cross-encoder scoring {len(pairs)} sentences...")
        logits = self._cross_encoder.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))

    def _select_supporting_sentences(self, query: str, results: list[RerankResult]) -> None:
        """Pick the evidence sentences for the top `sentence_passage_limit` passages.

        Every passage gets a guaranteed minimum (2 for the top two passages, 1
        below that) so a bridge fact is not lost for scoring just under the
        threshold; further sentences are added while they clear
        `sentence_score_threshold`, up to `max_sentences_per_passage`.
        """
        active_results = results[: self.sentence_passage_limit]
        passage_sentence_ranges: list[tuple[int, int]] = []
        all_sentences: list[str] = []
        # Blank sentences are skipped, so keep a mapping back to each
        # sentence's original index in its passage.
        sentence_to_orig_idx_mapping: list[list[int]] = []

        for r in active_results:
            start = len(all_sentences)
            orig_indices = []
            for i, s in enumerate(r.passage.sentences):
                s_strip = s.strip()
                if s_strip:
                    all_sentences.append(s_strip)
                    orig_indices.append(i)
            passage_sentence_ranges.append((start, start + len(orig_indices)))
            sentence_to_orig_idx_mapping.append(orig_indices)

        if not all_sentences:
            for r in results:
                r.supporting_sentence_indices = []
                r.supporting_sentences = []
                r.sentence_scores = []
            return

        sim_scores = self._score_sentences(query, all_sentences)

        query_words = self._normalize_tokens(query)
        # Zip over active_results, not results: the ranges were only built for
        # the first sentence_passage_limit passages, and the remaining results
        # have their sentence fields cleared explicitly at the end.
        for r_idx, (r, (start, end)) in enumerate(zip(active_results, passage_sentence_ranges, strict=True)):
            if end <= start:
                continue

            sents = all_sentences[start:end]
            scores = sim_scores[start:end].tolist()
            orig_indices = sentence_to_orig_idx_mapping[r_idx]

            # Title-match boost: if passage title words overlap with query, boost all sentence scores
            title_words = self._normalize_tokens(r.passage.title)
            if title_words & query_words:
                scores = [s + self.title_overlap_boost for s in scores]

            paired = sorted(zip(scores, sents, orig_indices, strict=True), reverse=True)

            selected_sents = []
            selected_scores = []
            selected_indices = []

            # Rank-aware guaranteed minimum: top passages get more sentences
            min_guarantee = 2 if r.rank <= 1 else 1

            for i, (score, sent, orig_idx) in enumerate(paired):
                if i < min_guarantee or (
                    score >= self.sentence_score_threshold and len(selected_sents) < self.max_sentences_per_passage
                ):
                    selected_sents.append(sent)
                    selected_scores.append(round(float(score), 4))
                    selected_indices.append(orig_idx)
                else:
                    break  # Sorted descending — all remaining are below threshold

            # Sort the selected sentences chronologically (by orig_idx) so the
            # prompt reads naturally.
            chronological = sorted(zip(selected_indices, selected_sents, selected_scores, strict=True))

            if chronological:
                indices, sentences, sentence_scores = zip(*chronological, strict=True)
                r.supporting_sentence_indices = list(indices)
                r.supporting_sentences = list(sentences)
                r.sentence_scores = list(sentence_scores)
            else:
                r.supporting_sentence_indices = []
                r.supporting_sentences = []
                r.sentence_scores = []

        # Clear sentence fields for the lower-ranked passages we skipped
        for r in results[self.sentence_passage_limit :]:
            r.supporting_sentence_indices = []
            r.supporting_sentences = []
            r.sentence_scores = []

    def __repr__(self) -> str:
        return (
            f"Reranker(model={self.model_name}, "
            f"threshold={self.sentence_score_threshold}, "
            f"max_sents={self.max_sentences_per_passage}, "
            f"device={self.device})"
        )
