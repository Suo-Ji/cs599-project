"""Hybrid retrieval engine (Layer 2) — the integration point.

Combines dense (cosine) retrieval, sparse (BM25) retrieval, RRF fusion, and
cross-encoder reranking into a single :class:`HybridRetriever`. Pipeline:

    query
      |-- dense  (top dense_top_k)   --|
      |-- sparse (top sparse_top_k)   --|--> RRF fuse (k=60) --> top rerank_candidates
                                                      |
                                          cross-encoder rerank --> final top_n

Dense and sparse ranks are fused by RRF; the fused order is reranked by the
cross-encoder; the top ``final_top_n`` chunks feed the LLM context window.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..common.config import AppConfig, get_config
from ..common.logging_setup import get_logger
from ..common.schemas import DocumentChunk
from ..ingestion.vectorstore import VectorIndex
from .bm25_retriever import BM25Retriever
from .reranker import CrossEncoderReranker
from .rrf import fuse_scores

_logger: logging.Logger = get_logger("retrieval.hybrid")


class HybridRetriever:
    """Dense + sparse retrieval fused by RRF, then cross-encoder reranked."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        vector_index: Optional[VectorIndex] = None,
        bm25: Optional[BM25Retriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self._config = config or get_config()
        self._vector_index = vector_index
        self._bm25 = bm25 or BM25Retriever()
        self._reranker = reranker

        # Cached thresholds / sizes from config.
        self._dense_top_k = self._config.retrieval.dense_top_k
        self._sparse_top_k = self._config.retrieval.sparse_top_k
        self._rrf_k = self._config.retrieval.rrf_k
        self._rerank_candidates = self._config.retrieval.rerank_candidates
        self._final_top_n = self._config.retrieval.final_top_n

    # ------------------------------------------------------------------
    # Construction helper
    # ------------------------------------------------------------------

    @classmethod
    def for_corpus(
        cls,
        chunks: list[DocumentChunk],
        config: Optional[AppConfig] = None,
        vector_index: Optional[VectorIndex] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> "HybridRetriever":
        """Build a retriever over an in-memory chunk corpus.

        ``chunks`` populate the BM25 index directly; if a ``vector_index`` is
        supplied it is used for dense retrieval (it should already contain the
        same chunks). When no reranker is given, reranking is disabled and the
        RRF-fused order is returned as the final ranking.
        """
        cfg = config or get_config()
        bm25 = BM25Retriever()
        bm25.add_chunks(chunks)
        return cls(config=cfg, vector_index=vector_index, bm25=bm25, reranker=reranker)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Populate the BM25 index from ``chunks`` (dense index set at init)."""
        self._bm25.add_chunks(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        where: Optional[dict[str, Any]] = None,
        final_top_n: Optional[int] = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Run the full hybrid pipeline; return ``(chunk, score)`` pairs.

        The returned score is the cross-encoder rerank score when a reranker is
        configured, otherwise the RRF fused score.
        """
        fused = self._fuse(query, where=where)
        if not fused:
            _logger.info("Hybrid retrieval returned no candidates for query.")
            return []

        candidates = self._select_candidates(fused)
        top_n = final_top_n or self._final_top_n

        if self._reranker is not None:
            try:
                candidate_chunks = [chunk for chunk, _ in candidates]
                reranked = self._reranker.rerank(query, candidate_chunks, top_n=top_n)
                _logger.info("Retrieved %d candidates, reranked to %d.", len(candidates), len(reranked))
                return reranked
            except Exception as exc:  # noqa: BLE001 - never crash the graph
                _logger.error("Reranking failed; falling back to RRF order: %s", exc)

        # No reranker (or rerank failed): return RRF-fused order.
        return [(chunk, score) for chunk, score in candidates[:top_n]]

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _fuse(
        self, query: str, where: Optional[dict[str, Any]] = None
    ) -> list[tuple[DocumentChunk, float]]:
        """Run dense + sparse, fuse by RRF, return ``(chunk, rrf_score)`` pairs."""
        dense_results = self._dense_search(query, where=where)
        sparse_results = self._sparse_search(query)

        dense_ids = [c.id for c, _ in dense_results]
        sparse_ids = [c.id for c, _ in sparse_results]

        fused_ids = fuse_scores(
            [dense_ids, sparse_ids], k=self._rrf_k, top_n=self._rerank_candidates
        )

        if not fused_ids:
            return []

        # Rehydrate fused ids to chunks via the union of both result sets.
        by_id: dict[str, DocumentChunk] = {
            c.id: c for c, _ in (dense_results + sparse_results)
        }
        fused: list[tuple[DocumentChunk, float]] = []
        for doc_id, score in fused_ids:
            chunk = by_id.get(doc_id)
            if chunk is not None:
                fused.append((chunk, score))
        _logger.info(
            "Fusion: dense=%d sparse=%d -> %d unique candidates.",
            len(dense_ids), len(sparse_ids), len(fused),
        )
        return fused

    def _dense_search(
        self, query: str, where: Optional[dict[str, Any]] = None
    ) -> list[tuple[DocumentChunk, float]]:
        """Cosine dense retrieval via the vector index."""
        if self._vector_index is None:
            _logger.debug("No vector index configured; dense retrieval empty.")
            return []
        try:
            return self._vector_index.query(query, top_k=self._dense_top_k, where=where)
        except Exception as exc:  # noqa: BLE001 - isolate dense failures
            _logger.error("Dense retrieval failed: %s", exc, exc_info=True)
            return []

    def _sparse_search(self, query: str) -> list[tuple[DocumentChunk, float]]:
        """BM25 sparse retrieval."""
        try:
            return self._bm25.query(query, top_k=self._sparse_top_k)
        except Exception as exc:  # noqa: BLE001 - isolate sparse failures
            _logger.error("Sparse retrieval failed: %s", exc, exc_info=True)
            return []

    @staticmethod
    def _select_candidates(
        fused: list[tuple[DocumentChunk, float]]
    ) -> list[tuple[DocumentChunk, float]]:
        """Pass-through hook kept explicit so candidate selection is visible."""
        return fused
