"""Cross-encoder reranker (Layer 2).

Wraps a sentence-transformers cross-encoder (default ``BAAI/bge-reranker-large``)
to rescore query-passage pairs. Unlike bi-encoders used for retrieval, the
cross-encoder jointly attends over the query and passage, yielding more accurate
relevance scores. The model loads lazily so importing this module is free.

Reranking takes the RRF-fused candidate set (top 20) and returns the top N
(default 5) most relevant chunks to the LLM context window.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..common.logging_setup import get_logger
from ..common.schemas import DocumentChunk

_logger: logging.Logger = get_logger("retrieval.reranker")


class CrossEncoderReranker:
    """Lazy sentence-transformers cross-encoder reranker."""

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking."
            ) from exc
        _logger.info("Loading reranker model %s (device=%s)", self._model_name, self._device)
        self._model = CrossEncoder(self._model_name, device=self._device)

    def rerank(
        self,
        query: str,
        candidates: list[DocumentChunk],
        top_n: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """Return the top-N ``(chunk, score)`` pairs for ``query``.

        The cross-encoder produces calibrated relevance logits; higher is more
        relevant. Candidates with no text are skipped. If the model is missing or
        scoring fails, the input order is preserved (graceful degradation) and
        the failure is logged rather than raised.
        """
        if not candidates:
            return []

        valid = [c for c in candidates if c.content.strip()]
        if not valid:
            return []

        try:
            self._ensure_loaded()
            pairs = [(query, c.content) for c in valid]
            scores = self._model.predict(pairs, convert_to_numpy=True)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            _logger.warning("Reranking unavailable (%s); preserving candidate order.", exc)
            return [(c, 0.0) for c in valid[:top_n]]

        ranked = sorted(
            zip(valid, [float(s) for s in scores]),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_n]
        return ranked

    def score_pair(self, query: str, passage: str) -> float:
        """Score a single query-passage pair (debug / evaluation helper)."""
        self._ensure_loaded()
        return float(self._model.predict([(query, passage)])[0])
