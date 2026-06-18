"""BM25 sparse retriever (Layer 2).

Tokenizes a corpus of document chunks and serves exact keyword matching via the
Okapi BM25 ranking function. Complements dense retrieval: technical abbreviations
(BNN, MLP) and specific metric values that dense embeddings smooth over are
matched literally here.

Tokenization is domain-aware:
  * case-folded
  * punctuation stripped
  * alphanumerics preserved (so "PICP", "MPIW", "0.41" survive)
  * a light stopword filter avoids over-weighting grammar words
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from ..common.logging_setup import get_logger
from ..common.schemas import DocumentChunk

_logger: logging.Logger = get_logger("retrieval.bm25")

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:\.[0-9]+)?")
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
        "is", "are", "was", "were", "be", "been", "being", "by", "with", "from",
        "as", "into", "this", "that", "these", "those", "it", "its", "we", "our",
        "they", "their", "which", "such", "than", "then", "so", "if", "not", "no",
    }
)


def tokenize(text: str) -> list[str]:
    """Tokenize text into normalized terms suitable for BM25 indexing."""
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


class BM25Retriever:
    """In-memory BM25 index over document chunks.

    The index is rebuilt on :meth:`add_chunks`; BM25 is a closed-form ranker so
    there is no incremental training. Designed to be (re)built cheaply at
    ingestion time or at agent warm-up from a chunk list.
    """

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._tokenized: list[list[str]] = []
        self._bm25: Optional["BM25Okapi"] = None  # type: ignore[name-defined]

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Index a list of chunks. Replaces any existing index."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("rank_bm25 is required for sparse retrieval.") from exc

        self._chunks = list(chunks)
        self._tokenized = [tokenize(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized)
        _logger.info("BM25 index built over %d chunks.", len(self._chunks))
        return len(self._chunks)

    @property
    def size(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(self, query_text: str, top_k: int = 20) -> list[tuple[DocumentChunk, float]]:
        """Return ``(chunk, bm25_score)`` pairs ranked by descending relevance.

        BM25 scores are unbounded positive floats (not normalized). They are used
        only for ranking within this method; RRF consumes ranks, not raw scores.
        """
        if self._bm25 is None or not self._chunks:
            _logger.warning("BM25 query on empty index; returning no results.")
            return []

        query_tokens = tokenize(query_text)
        if not query_tokens:
            return []

        try:
            scores = self._bm25.get_scores(query_tokens)
        except Exception as exc:  # noqa: BLE001 - localize scoring failures
            _logger.error("BM25 scoring failed: %s", exc, exc_info=True)
            return []

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [(self._chunks[i], float(scores[i])) for i in ranked_indices]

    def ranked_ids(self, query_text: str, top_k: int = 20) -> list[str]:
        """Return just the chunk ids in BM25 rank order (for RRF consumption)."""
        return [c.id for c, _ in self.query(query_text, top_k=top_k)]
