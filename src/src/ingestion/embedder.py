"""Embedding model wrapper (Layer 1).

Wraps sentence-transformers to serve dense vectors for the configured model
(default ``BAAI/bge-large-zh-v1.5``). The model is loaded lazily on first use so
importing this module never triggers a network download. All calls are wrapped so
a model error is reported and never crashes the caller unexpectedly.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from ..common.logging_setup import get_logger

_logger: logging.Logger = get_logger("ingestion.embedder")


class Embedder:
    """Lazy sentence-transformers embedder producing normalized dense vectors."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._normalize = normalize_embeddings
        self._model: Any = None
        self._dim: Optional[int] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        """Embedding dimension, populated after the first encode call."""
        if self._dim is None:
            raise RuntimeError("Embedding dimension unknown until the model is loaded.")
        return self._dim

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "sentence-transformers is required to generate embeddings."
            ) from exc
        _logger.info("Loading embedding model %s (device=%s)", self._model_name, self._device)
        self._model = SentenceTransformer(self._model_name, device=self._device)

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Embed a batch of texts and return a (N, dim) float32 numpy array."""
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        self._ensure_loaded()
        try:
            vectors = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=self._normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
        except Exception as exc:  # noqa: BLE001 - surface but localize
            _logger.error("Embedding encode failed: %s", exc, exc_info=True)
            raise
        vectors = np.asarray(vectors, dtype=np.float32)
        self._dim = int(vectors.shape[1])
        return vectors

    def encode_one(self, text: str) -> np.ndarray:
        """Embed a single text and return a 1-D float32 numpy array."""
        return self.encode([text])[0]
