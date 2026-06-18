"""Semantic chunker for parsed academic documents (Layer 1).

Operates on :class:`~src.ingestion.parser.ParsedDocument` objects and emits
:class:`~src.common.schemas.DocumentChunk` instances sized to a target token
window (500-800 by default) with ~10% overlap between adjacent chunks.

Design constraints enforced here:
  * No hard character truncation. Chunks are assembled from whole logical units
    (paragraphs, atomic equations/tables). A paragraph larger than the window is
    split at sentence boundaries as a last resort.
  * Equations and experimental tables are atomic: they never straddle a chunk
    boundary, so formulas and metric tables stay intact within a single chunk.
  * Overlap is added at the block boundary nearest the overlap target, so no
    atomic block is cut to produce overlap.
"""

from __future__ import annotations

import logging
import re
from hashlib import md5
from typing import Protocol

from ..common.config import AppConfig
from ..common.logging_setup import get_logger
from ..common.schemas import DocumentChunk
from .keyword_extractor import extract_keywords
from .parser import ParsedDocument, ParsedSection, TextBlock
from .tokenizer_utils import TokenCounter, count_tokens, get_token_counter, sanitize_text

_logger: logging.Logger = get_logger("ingestion.splitter")

# Sentence splitter tolerant of common abbreviations (best-effort, deterministic).
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class _ContentUnit:
    """A chunking building block derived from a parsed block.

    Atomic units (equations, tables) carry ``is_atomic=True`` and are never split
    further. Paragraphs may be split into sentence units when oversized.
    """

    __slots__ = ("text", "tokens", "page", "is_atomic")

    def __init__(self, text: str, tokens: int, page: int, is_atomic: bool) -> None:
        self.text = text
        self.tokens = tokens
        self.page = page
        self.is_atomic = is_atomic

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"_ContentUnit(tokens={self.tokens}, atomic={self.is_atomic})"


class Tokenizable(Protocol):
    def __call__(self, text: str) -> int: ...


class SemanticSplitter:
    """Splits a parsed document into metadata-rich, semantically grouped chunks."""

    def __init__(
        self,
        config: AppConfig,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._config = config
        self._counter: TokenCounter = token_counter or get_token_counter()
        self._min_tokens = config.ingestion.min_chunk_tokens
        self._min_target = config.ingestion.chunk_min_tokens
        self._max_target = config.ingestion.chunk_max_tokens
        self._overlap_target = max(1, int(config.ingestion.overlap_ratio * config.ingestion.chunk_max_tokens))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def split_document(self, document: ParsedDocument) -> list[DocumentChunk]:
        """Split every section of ``document`` into DocumentChunk objects."""
        source_stem = _stem(document.source_file)
        chunks: list[DocumentChunk] = []
        global_index = 0

        for section in document.sections:
            section_units = self._section_to_units(section)
            if not section_units:
                continue
            section_chunks = self._pack_units(
                units=section_units,
                source_stem=source_stem,
                section_title=section.title,
                start_index=global_index,
            )
            for chunk in section_chunks:
                chunk.chunk_index = global_index
                global_index += 1
            chunks.extend(section_chunks)

        _logger.info(
            "Split %s into %d chunks (target %d-%d tokens, %d%% overlap)",
            document.source_file, len(chunks),
            self._min_target, self._max_target, int(self._config.ingestion.overlap_ratio * 100),
        )
        return chunks

    # ------------------------------------------------------------------
    # Unit construction
    # ------------------------------------------------------------------

    def _section_to_units(self, section: ParsedSection) -> list[_ContentUnit]:
        """Flatten a section's blocks into ordered chunking units."""
        units: list[_ContentUnit] = []
        for block in section.blocks:
            units.extend(self._block_to_units(block))
        return units

    def _block_to_units(self, block: TextBlock) -> list[_ContentUnit]:
        """Convert one parsed block into one or more chunking units.

        Atomic blocks (equations, tables) become a single unit even if they exceed
        the window (a warning is logged). Oversized paragraphs are split into
        sentence units; small paragraphs stay whole.
        """
        token_count = count_tokens(block.text, self._counter)

        if block.is_atomic:
            if token_count > self._max_target:
                _logger.warning(
                    "Atomic %s block (%d tokens) exceeds window %d; kept intact.",
                    block.kind, token_count, self._max_target,
                )
            return [_ContentUnit(block.text, token_count, block.page_number, is_atomic=True)]

        if token_count <= self._max_target:
            return [_ContentUnit(block.text, token_count, block.page_number, is_atomic=False)]

        # Oversized paragraph: split at sentence boundaries, never character-truncate.
        return self._split_paragraph_by_sentence(block.text, block.page_number)

    def _split_paragraph_by_sentence(self, text: str, page: int) -> list[_ContentUnit]:
        sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        if not sentences:
            return []

        units: list[_ContentUnit] = []
        buffer: list[str] = []
        buffer_tokens = 0

        def flush() -> None:
            nonlocal buffer, buffer_tokens
            if buffer:
                joined = " ".join(buffer).strip()
                units.append(_ContentUnit(joined, buffer_tokens, page, is_atomic=False))
                buffer = []
                buffer_tokens = 0

        for sentence in sentences:
            sent_tokens = count_tokens(sentence, self._counter)
            # A single sentence larger than the window is still kept whole; the
            # contract forbids character truncation, so we do not split further.
            if buffer and buffer_tokens + sent_tokens > self._max_target:
                flush()
            buffer.append(sentence)
            buffer_tokens += sent_tokens

        flush()
        return units

    # ------------------------------------------------------------------
    # Packing (greedy window assembly with overlap)
    # ------------------------------------------------------------------

    def _pack_units(
        self,
        units: list[_ContentUnit],
        source_stem: str,
        section_title: str,
        start_index: int,
    ) -> list[DocumentChunk]:
        """Greedy-pack units into chunks within the target window.

        Algorithm:
          1. Accumulate units until adding the next unit would exceed ``_max_target``.
          2. If the accumulated set is non-empty, emit it as a chunk, then start a
             new chunk seeded with an overlap window (the trailing units of the
             emitted chunk whose cumulative tokens are <= ``_overlap_target``).
          3. Atomic units that individually exceed the window flush the buffer
             first, then occupy their own chunk.
        """
        chunks: list[DocumentChunk] = []
        local_index = 0
        buffer: list[_ContentUnit] = []
        buffer_tokens = 0
        unit_cursor = 0
        overlap_pending: list[_ContentUnit] = []

        while unit_cursor < len(units):
            unit = units[unit_cursor]

            # An atomic oversized unit must stand alone: flush buffer first.
            if unit.is_atomic and unit.tokens > self._max_target and buffer:
                chunks.append(
                    self._build_chunk(
                        buffer, source_stem, section_title, start_index + local_index
                    )
                )
                local_index += 1
                buffer = []
                buffer_tokens = 0

            if buffer and (buffer_tokens + unit.tokens > self._max_target) and buffer_tokens >= self._min_target:
                # Emit the current buffer; compute overlap for the next chunk.
                chunks.append(
                    self._build_chunk(
                        buffer, source_stem, section_title, start_index + local_index
                    )
                )
                local_index += 1
                overlap_pending = self._compute_overlap(buffer)
                buffer = list(overlap_pending)
                buffer_tokens = sum(u.tokens for u in buffer)
                overlap_pending = []

            buffer.append(unit)
            buffer_tokens += unit.tokens
            unit_cursor += 1

        if buffer:
            chunks.append(
                self._build_chunk(buffer, source_stem, section_title, start_index + local_index)
            )

        return chunks

    def _compute_overlap(self, emitted: list[_ContentUnit]) -> list[_ContentUnit]:
        """Return the trailing units of ``emitted`` to seed the next chunk's overlap.

        Rules:
          * Atomic units (equations, tables) are never duplicated as overlap —
            they belong to exactly one chunk, per the integrity contract.
          * Trailing non-atomic paragraphs are carried until their cumulative size
            meets the overlap budget. The most recent paragraph is always included
            even if it alone exceeds the budget, so overlap is non-empty whenever
            the emitted chunk ends in a paragraph.
        """
        overlap: list[_ContentUnit] = []
        budget = self._overlap_target
        for unit in reversed(emitted):
            if unit.is_atomic:
                break
            if unit.tokens >= self._max_target:
                # A window-sized unit carries no boundary context and would
                # duplicate the bulk of the previous chunk; skip it.
                break
            overlap.append(unit)
            if sum(u.tokens for u in overlap) >= budget:
                break
        overlap.reverse()
        return overlap

    # ------------------------------------------------------------------
    # Chunk construction
    # ------------------------------------------------------------------

    def _build_chunk(
        self,
        units: list[_ContentUnit],
        source_stem: str,
        section_title: str,
        index: int,
    ) -> DocumentChunk:
        content = sanitize_text("\n".join(u.text for u in units).strip())
        token_count = count_tokens(content, self._counter)
        pages = [u.page for u in units if u.page > 0]
        page_start = min(pages) if pages else 0
        page_end = max(pages) if pages else 0
        keywords = extract_keywords(content, self._config)

        return DocumentChunk(
            id=_chunk_id(source_stem, section_title, index, content),
            content=content,
            source_file=f"{source_stem}.pdf",
            section_title=section_title,
            page_number=page_start,
            page_range=(page_start, page_end),
            keywords=keywords,
            chunk_index=index,
            token_count=token_count,
            metadata={"num_units": len(units)},
        )


def _stem(filename: str) -> str:
    """Return the filename without its extension."""
    return filename.rsplit(".", 1)[0] if "." in filename else filename


def _chunk_id(source_stem: str, section_title: str, index: int, content: str) -> str:
    """Deterministic chunk id: <source>::<index>::<content_hash>."""
    digest = md5(content.encode("utf-8")).hexdigest()[:10]
    return f"{source_stem}::{index:04d}::{digest}"
