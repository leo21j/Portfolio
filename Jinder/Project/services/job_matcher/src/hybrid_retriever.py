import logging

import numpy as np
import torch

from services.job_matcher.src.bm25_retriever import BM25Retriever
from services.job_matcher.src.graph_retriever import GraphRetriever
from services.job_matcher.src.scoring import combine_scores, normalize_scores, rank
from services.job_matcher.src.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, alpha: float = 0.5, use_rerank: bool = False, use_graph: bool = True):
        self.alpha = alpha
        self.use_rerank = use_rerank
        self.use_graph = use_graph
        self.vector_retriever = VectorRetriever()
        self.bm25_retriever = BM25Retriever()
        self.graph_retriever = GraphRetriever() if use_graph else None
        self.reranker = self._init_reranker() if use_rerank else None

        logger.info("HybridRetriever initialized with PyTorch")
        logger.info(f"Search balance: {alpha:.1f} vector + {1 - alpha:.1f} BM25")
        if use_graph:
            logger.info("Graph-based retrieval enabled")

    def _init_reranker(self):
        """Initialize cross-encoder reranker"""
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Cross-encoder reranking enabled")
            return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except ImportError:
            logger.warning("Cross-encoder not available, disabling reranking")
            self.use_rerank = False
            return None

    def build_index(self, chunks: list[dict], index_file: str | None = None):
        """Build both indices with PyTorch optimization"""
        logger.info("Building hybrid retrieval index...")

        self.vector_retriever.build_index(chunks, index_file=index_file)
        self.bm25_retriever.build_index(chunks)

        if self.graph_retriever and self.vector_retriever.index:
            # Get embeddings for graph building
            texts = [chunk['text'] for chunk in chunks]
            with torch.no_grad():
                embeddings = self.vector_retriever.model.encode(
                    texts,
                    convert_to_numpy=True,
                    device=self.vector_retriever.device,
                    show_progress_bar=True
                )
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_normalized = embeddings / norms

            self.graph_retriever.build_graph(chunks, embeddings_normalized)

        device_info = self.vector_retriever.get_device_info()
        logger.info("Hybrid retriever ready!")
        logger.info(f"Device: {device_info['device']}")
        logger.info(f"PyTorch: {device_info['pytorch_version']}")
        if device_info['cuda_available']:
            logger.info(f"CUDA GPUs: {device_info['cuda_device_count']}")

    def search(
        self,
        query: str,
        k: int = 10,
        use_rerank: bool | None = None,
        use_graph: bool | None = None,
    ) -> list[tuple[dict, float]]:
        """Hybrid search with optional cross-encoder reranking and graph expansion"""
        should_rerank = use_rerank if use_rerank is not None else self.use_rerank
        should_use_graph = use_graph if use_graph is not None else self.use_graph
        search_k = k * 3 if should_rerank else k * 2

        # Get hybrid results
        hybrid_results = self._get_hybrid_results(query, search_k)

        # Apply reranking if needed
        if should_rerank and self.reranker:
            hybrid_results = self._rerank_results(query, hybrid_results, k * 2)

        # Apply graph expansion if enabled
        if should_use_graph and self.graph_retriever:
            hybrid_results = self.graph_retriever.expand_results(
                hybrid_results,
                max_hops=1,
                max_total=k * 2
            )

        return hybrid_results[:k]

    def _get_hybrid_results(self, query: str, search_k: int) -> list[tuple[dict, float]]:
        """Get combined results from vector and BM25 search"""
        vector_results = self.vector_retriever.search(query, search_k)
        bm25_results = self.bm25_retriever.search(query, search_k)

        combined = combine_scores(
            vector_results,
            bm25_results,
            normalize_scores([score for _, score in vector_results]),
            normalize_scores([score for _, score in bm25_results]),
            alpha=self.alpha,
        )
        return rank(combined)

    def _rerank_results(
            self,
            query: str,
            results: list[tuple[dict, float]],
            k: int
    ) -> list[tuple[dict, float]]:
        """Apply cross-encoder reranking to results"""
        if not results:
            return results

        try:
            candidates = results[:min(len(results), k * 2)]
            # Truncate each document to keep cross-encoder memory bounded.
            pairs = [[query, chunk['text'][:512]] for chunk, _ in candidates]

            raw_scores = self.reranker.predict(pairs, show_progress_bar=False)
            scores = normalize_scores(list(raw_scores))

            reranked = [
                (candidates[i][0], float(scores[i]))
                for i in range(len(candidates))
            ]
            reranked.sort(key=lambda x: x[1], reverse=True)

            return reranked[:k]

        except Exception as e:
            logger.warning(f"Reranking failed, falling back to hybrid results: {e}")
            return results[:k]

    def get_system_info(self):
        vector_info = self.vector_retriever.get_device_info()

        return {
            **vector_info,
            "hybrid_alpha": self.alpha,
            "total_chunks": (
                len(self.vector_retriever.chunks) if self.vector_retriever.chunks else 0
            ),
        }
