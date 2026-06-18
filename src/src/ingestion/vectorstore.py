"""Chroma vector-store wrapper with metadata scalar filtering (Layer 1).

Stores document chunks as embeddings plus their flat metadata so retrieval can
filter by ``source_file``, ``section_title``, ``page_number`` and keywords. Uses
an external embedder (this wrapper never calls a default embedding function) so
the same model is used at index and query time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from ..common.logging_setup import get_logger
from ..common.schemas import DocumentChunk
from .embedder import Embedder

_logger: logging.Logger = get_logger("ingestion.vectorstore")


class VectorIndex:
    """Persistent Chroma index over :class:`DocumentChunk` objects."""

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str,
        embedder: Embedder,
        distance_metric: str = "cosine",
    ) -> None:
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._embedder = embedder
        self._distance_metric = distance_metric
        self._client: Any = None
        self._collection: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("chromadb is required for the vector store.") from exc
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._distance_metric},
        )

    def open(self) -> "VectorIndex":
        self._ensure_client()
        return self

    def close(self) -> None:
        self._collection = None
        self._client = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[DocumentChunk], batch_size: int = 64) -> int:
        """Embed and upsert chunks by their stable id. Returns the count stored."""
        if not chunks:
            return 0
        self._ensure_client()
        stored = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [c.content for c in batch]
            try:
                embeddings = self._embedder.encode(texts).tolist()
            except Exception as exc:  # noqa: BLE001 - keep index run alive
                _logger.error("Embedding failed for batch %d-%d: %s", start, start + len(batch), exc)
                continue
            self._collection.upsert(
                ids=[c.id for c in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[c.to_filterable_metadata() for c in batch],
            )
            stored += len(batch)
        _logger.info("Upserted %d/%d chunks into '%s'", stored, len(chunks), self._collection_name)
        return stored

    def count(self) -> int:
        self._ensure_client()
        try:
            return int(self._collection.count())
        except Exception as exc:  # noqa: BLE001
            _logger.warning("count() failed: %s", exc)
            return 0

    def reset(self) -> None:
        """Delete the collection (used by re-indexing workflows)."""
        self._ensure_client()
        try:
            self._client.delete_collection(self._collection_name)
        except Exception as exc:  # noqa: BLE001
            _logger.debug("delete_collection (ignored): %s", exc)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._distance_metric},
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 20,
        where: Optional[dict[str, Any]] = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Return ``(chunk, similarity_score)`` pairs ranked by cosine similarity.

        ``similarity_score`` is in [0, 1] (1 - normalized distance). ``where`` is a
        Chroma metadata filter enabling scalar filtering on injected metadata.
        """
        self._ensure_client()
        try:
            query_vector = self._embedder.encode_one(query_text).tolist()
        except Exception as exc:  # noqa: BLE001
            _logger.error("Query embedding failed: %s", exc)
            return []

        try:
            result = self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            _logger.error("Chroma query failed: %s", exc)
            return []

        return self._results_to_chunks(result)

    @staticmethod
    def _results_to_chunks(result: Any) -> list[tuple[DocumentChunk, float]]:
        """Project a Chroma query result onto ``(DocumentChunk, score)`` pairs."""
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        pairs: list[tuple[DocumentChunk, float]] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            meta = meta or {}
            keywords_csv = meta.pop("keywords", "") if isinstance(meta, dict) else ""
            chunk = DocumentChunk(
                id=meta.pop("id", ""),
                content=doc,
                source_file=meta.pop("source_file", "unknown"),
                section_title=meta.pop("section_title", "unknown"),
                page_number=int(meta.pop("page_number", 0) or 0),
                page_range=(int(meta.pop("page_number", 0) or 0),) * 2,
                keywords=[k for k in str(keywords_csv).split(",") if k],
                chunk_index=int(meta.pop("chunk_index", 0) or 0),
                metadata=meta,
            )
            # Chroma cosine space returns distance in [0, 2]; similarity = 1 - distance.
            score = max(0.0, 1.0 - float(dist))
            pairs.append((chunk, score))
        return pairs
