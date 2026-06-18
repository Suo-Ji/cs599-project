"""Ingestion pipeline orchestration (Layer 1).

Wires the parser, semantic splitter, embedder, and vector index together so a
single call can turn a directory of source documents into a populated vector
store. Designed to degrade gracefully: parsing/splitting succeed even when the
heavy embedding model is unavailable (the indexing step is then skipped with an
explicit log line rather than failing silently).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..common.config import AppConfig, get_config
from ..common.logging_setup import get_logger
from ..common.schemas import DocumentChunk
from .embedder import Embedder
from .parser import ParsedDocument, parse_document
from .splitter import SemanticSplitter
from .vectorstore import VectorIndex

_logger: logging.Logger = get_logger("ingestion.pipeline")


@dataclass
class IngestionStats:
    """Summary of a pipeline run."""

    documents: int = 0
    chunks: int = 0
    indexed: int = 0
    skipped_files: list[str] = field(default_factory=list)
    indexed_successfully: bool = True
    chunks_by_source: dict[str, int] = field(default_factory=dict)
    chunks_by_section: dict[str, int] = field(default_factory=dict)

    @property
    def sections(self) -> list[str]:
        return sorted(self.chunks_by_section, key=self.chunks_by_section.get, reverse=True)  # type: ignore[arg-type]


class IngestionPipeline:
    """Parse -> split -> (optionally) embed & index source documents."""

    SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".pdf", ".txt", ".md"})

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        embedder: Optional[Embedder] = None,
        vector_index: Optional[VectorIndex] = None,
    ) -> None:
        self._config = config or get_config()
        self._splitter = SemanticSplitter(self._config)
        self._embedder = embedder
        self._vector_index = vector_index

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def with_default_vectorstore(cls, config: Optional[AppConfig] = None) -> "IngestionPipeline":
        """Build a pipeline backed by a real Chroma index + configured embedder."""
        cfg = config or get_config()
        embedder = Embedder(model_name=cfg.ingestion.embedding_model)
        index = VectorIndex(
            persist_dir=cfg.vectorstore_dir,
            collection_name=cfg.paths.collection_name,
            embedder=embedder,
            distance_metric=cfg.vectorstore.distance_metric,
        ).open()
        return cls(config=cfg, embedder=embedder, vector_index=index)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, source_dir: Optional[Path] = None, reset_index: bool = False) -> IngestionStats:
        """Ingest all supported files in ``source_dir`` (default: configured pdf_dir)."""
        directory = source_dir or self._config.pdf_dir
        if not directory.exists():
            _logger.warning("Source directory does not exist: %s", directory)
            return IngestionStats()

        if reset_index and self._vector_index is not None:
            self._vector_index.reset()
            _logger.info("Vector index reset before ingestion.")

        files = [
            p for p in sorted(directory.iterdir())
            if p.suffix.lower() in self.SUPPORTED_SUFFIXES
        ]
        if not files:
            _logger.warning("No supported source files found in %s", directory)
            return IngestionStats()

        all_chunks: list[DocumentChunk] = []
        stats = IngestionStats()

        for path in files:
            try:
                document = parse_document(path)
                chunks = self._splitter.split_document(document)
            except Exception as exc:  # noqa: BLE001 - keep run going across files
                _logger.error("Failed to process %s: %s", path.name, exc, exc_info=True)
                stats.skipped_files.append(path.name)
                continue

            all_chunks.extend(chunks)
            stats.documents += 1
            stats.chunks += len(chunks)
            stats.chunks_by_source[path.name] = len(chunks)
            for chunk in chunks:
                stats.chunks_by_section[chunk.section_title] = (
                    stats.chunks_by_section.get(chunk.section_title, 0) + 1
                )

        if self._vector_index is not None and self._embedder is not None:
            try:
                stats.indexed = self._vector_index.add_chunks(all_chunks)
            except Exception as exc:  # noqa: BLE001 - indexing is non-fatal to splitting
                _logger.error("Indexing failed: %s", exc, exc_info=True)
                stats.indexed_successfully = False
        else:
            stats.indexed_successfully = False
            _logger.info("No vector index configured; skipping embedding step (chunks still produced).")

        _logger.info(
            "Ingestion complete: %d documents, %d chunks, %d indexed.",
            stats.documents, stats.chunks, stats.indexed,
        )
        return stats

    # ------------------------------------------------------------------
    # Accessors for tests / validation
    # ------------------------------------------------------------------

    def parse_and_split(self, source_path: Path) -> tuple[ParsedDocument, list[DocumentChunk]]:
        """Parse and split a single file without indexing. Used by validation."""
        document = parse_document(source_path)
        chunks = self._splitter.split_document(document)
        return document, chunks
