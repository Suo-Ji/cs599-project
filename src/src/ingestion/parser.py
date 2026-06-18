"""PDF / text parser for academic literature (Layer 1).

Produces a :class:`ParsedDocument` that preserves document structure:
sections (with heading level and page span) and atomic text blocks
(paragraphs, display equations, tables). Structure preservation is what lets the
downstream splitter keep formulas and experimental metric tables intact within a
single chunk.

PDF extraction uses ``pypdf``; plain-text inputs are accepted directly so the
pipeline can be exercised on fixtures without a PDF.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..common.logging_setup import get_logger
from .tokenizer_utils import sanitize_text

_logger: logging.Logger = get_logger("ingestion.parser")

# ------------------------------------------------------------------
# Heading detection heuristics
# ------------------------------------------------------------------

# "1 Introduction", "2.1 Related Work", "3. Methodology"
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{0,80})$")
# Bare ALL-CAPS short heading with no terminal period: "INTRODUCTION", "RESULTS"
_ALLCAPS_HEADING = re.compile(r"^[A-Z][A-Z0-9 \-:&/]{1,50}$")

# Recognized un-numbered section keywords (compared on the lowercased line).
_KEYWORD_HEADINGS: frozenset[str] = frozenset(
    {
        "abstract", "introduction", "background", "related work", "related works",
        "preliminaries", "methodology", "methods", "method", "approach",
        "model", "framework", "experiments", "experimental setup",
        "experimental results", "experiments and results", "results",
        "evaluation", "ablation study", "ablation studies", "analysis",
        "discussion", "conclusion", "conclusions", "future work",
        "references", "bibliography", "appendix", "appendices",
        "acknowledgments", "acknowledgements",
    }
)

# Display-math / table environment delimiters (LaTeX-flavored and markdown).
_MATH_ENVS = ("equation", "align", "gather", "eqnarray", "multline")
_TABLE_ENVS = ("table", "tabular", "longtable", "table*")


class TextBlock(BaseModel):
    """An atomic unit of parsed text.

    Atomic blocks (equations, tables) are never split across chunks downstream.
    """

    kind: Literal["paragraph", "equation", "table", "heading"]
    text: str
    page_number: int = Field(default=0, ge=0)

    @property
    def is_atomic(self) -> bool:
        """Equations and tables must remain within a single chunk."""
        return self.kind in {"equation", "table"}


class ParsedSection(BaseModel):
    """A logical section (heading + the blocks that follow until the next heading)."""

    title: str
    level: int = Field(default=1, ge=0)
    page_start: int = Field(default=0, ge=0)
    page_end: int = Field(default=0, ge=0)
    blocks: list[TextBlock] = Field(default_factory=list)

    def append_block(self, block: TextBlock) -> None:
        self.blocks.append(block)
        if block.page_number > 0:
            self.page_end = block.page_number if self.page_end == 0 else max(self.page_end, block.page_number)


class ParsedDocument(BaseModel):
    """Full parsed representation of one source file."""

    source_file: str
    page_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    sections: list[ParsedSection] = Field(default_factory=list)

    @property
    def block_count(self) -> int:
        return sum(len(s.blocks) for s in self.sections)


# ------------------------------------------------------------------
# Internal accumulator (mutable, never serialized)
# ------------------------------------------------------------------


@dataclass
class _SectionAccumulator:
    """In-progress section used while streaming lines; converted to ParsedSection."""

    title: str
    level: int
    page_start: int
    lines: list[tuple[str, int]] = field(default_factory=list)

    def to_parsed_section(self) -> ParsedSection:
        blocks = _segment_blocks(self.lines)
        section = ParsedSection(title=self.title, level=self.level, page_start=self.page_start)
        if blocks:
            section.page_start = self.page_start or blocks[0].page_number
            section.page_end = max(b.page_number for b in blocks)
            for block in blocks:
                section.append_block(block)
        else:
            section.page_end = self.page_start
        return section


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def parse_document(source_path: str | Path) -> ParsedDocument:
    """Parse a PDF or text file into a structured :class:`ParsedDocument`.

    Raises ``ValueError`` for unsupported extensions or unreadable files.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Source document not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages_text, raw_meta = _read_pdf(path)
    elif suffix in {".txt", ".md"}:
        pages_text, raw_meta = _read_text(path)
    else:
        raise ValueError(f"Unsupported source extension '{suffix}' for {path}")

    sections = _segment_into_sections(pages_text)
    sections = [s for s in sections if s.blocks]
    _logger.info(
        "Parsed %s: %d pages, %d sections, %d blocks",
        path.name, len(pages_text), len(sections), sum(len(s.blocks) for s in sections),
    )
    return ParsedDocument(
        source_file=path.name,
        page_count=len(pages_text),
        metadata=raw_meta,
        sections=sections,
    )


# ------------------------------------------------------------------
# Readers
# ------------------------------------------------------------------


def _read_pdf(path: Path) -> tuple[list[str], dict[str, Any]]:
    """Extract per-page text and document metadata via pypdf.

    Wrapped so an extraction failure on a single page logs a warning and yields
    an empty page instead of crashing the whole ingestion run.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("pypdf is required to parse PDF files.") from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    meta: dict[str, Any] = {}
    try:
        if reader.metadata is not None:
            meta = {
                k.lstrip("/"): str(v)
                for k, v in reader.metadata.items()
                if v is not None
            }
    except Exception as exc:  # noqa: BLE001 - metadata is non-critical
        _logger.warning("Metadata extraction failed for %s: %s", path.name, exc)

    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - per-page resilience
            _logger.warning("Text extraction failed on page %d of %s: %s", index, path.name, exc)
            text = ""
        pages.append(sanitize_text(text))
    return pages, meta


def _read_text(path: Path) -> tuple[list[str], dict[str, Any]]:
    """Read a plain-text fixture as a single page."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return [sanitize_text(text)], {"source_type": "text", "line_count": text.count("\n") + 1}


# ------------------------------------------------------------------
# Section segmentation
# ------------------------------------------------------------------


def _detect_heading(line: str) -> tuple[str, int] | None:
    """Return (title, level) if ``line`` is a heading, else None.

    ``level`` is derived from the numeric prefix depth (e.g. "2.1" -> 2); keyword
    and bare-capital headings default to level 1.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None

    # Trailing period is tolerated, but a sentence ending in a period is not a heading.
    candidate = stripped[:-1] if stripped.endswith(".") and not stripped.endswith("..") else stripped

    numbered = _NUMBERED_HEADING.match(candidate)
    if numbered:
        prefix, title = numbered.group(1), numbered.group(2).strip()
        level = prefix.count(".") + 1
        return title, level

    lowered = candidate.lower()
    if lowered in _KEYWORD_HEADINGS:
        return candidate, 1

    # Bare all-caps short lines, only if they contain a space or are short enough.
    if _ALLCAPS_HEADING.match(candidate) and (" " in candidate or len(candidate) <= 12):
        return candidate, 1

    return None


def _segment_into_sections(pages_text: list[str]) -> list[ParsedSection]:
    """Split per-page text into sections using heading heuristics."""
    accumulators: list[_SectionAccumulator] = [
        _SectionAccumulator(title="(preamble)", level=0, page_start=1)
    ]
    current = accumulators[0]

    for page_index, page_text in enumerate(pages_text, start=1):
        for raw_line in page_text.split("\n"):
            line = raw_line.rstrip()
            heading = _detect_heading(line)
            if heading is not None:
                title, level = heading
                current = _SectionAccumulator(title=title, level=level, page_start=page_index)
                accumulators.append(current)
            else:
                current.lines.append((line, page_index))

    return [acc.to_parsed_section() for acc in accumulators]


# ------------------------------------------------------------------
# Block segmentation within a section's accumulated lines
# ------------------------------------------------------------------


def _segment_blocks(ordered_lines: list[tuple[str, int]]) -> list[TextBlock]:
    """Classify ordered (line, page) pairs into atomic text blocks.

    Recognizes three atomic kinds beyond paragraphs:
      * display equations: ``$$ ... $$``, ``\\[ ... \\]``, ``\\begin{equation}``
      * tables: ``\\begin{table}``/``\\begin{tabular}`` blocks and consecutive
        markdown-style ``| ... |`` rows
    Inline ``$ ... $`` math stays inside its paragraph.
    """
    blocks: list[TextBlock] = []
    paragraph_buffer: list[str] = []
    paragraph_page: int = 0

    def flush_paragraph() -> None:
        nonlocal paragraph_buffer, paragraph_page
        text = " ".join(paragraph_buffer).strip()
        if text:
            blocks.append(TextBlock(kind="paragraph", text=text, page_number=paragraph_page))
        paragraph_buffer = []
        paragraph_page = 0

    i = 0
    n = len(ordered_lines)
    while i < n:
        line, page = ordered_lines[i]
        stripped = line.strip()

        if _is_display_math_open(stripped):
            flush_paragraph()
            block_lines, consumed = _consume_atomic_env(ordered_lines, i, mode="math")
            blocks.append(TextBlock(kind="equation", text="\n".join(block_lines).strip(), page_number=page))
            i += consumed
            continue

        if _is_table_env_open(stripped) or _is_markdown_table_row(stripped):
            flush_paragraph()
            block_lines, consumed = _consume_atomic_env(ordered_lines, i, mode="table")
            blocks.append(TextBlock(kind="table", text="\n".join(block_lines).strip(), page_number=page))
            i += consumed
            continue

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if paragraph_page == 0:
            paragraph_page = page
        paragraph_buffer.append(stripped)
        i += 1

    flush_paragraph()
    return blocks


def _is_display_math_open(line: str) -> bool:
    if line.startswith("$$") or line.startswith("\\["):
        return True
    return any(line.startswith(f"\\begin{{{env}}}") for env in _MATH_ENVS)


def _is_table_env_open(line: str) -> bool:
    return any(line.startswith(f"\\begin{{{env}}}") for env in _TABLE_ENVS)


def _is_markdown_table_row(line: str) -> bool:
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 2


def _is_display_math_close(line: str) -> bool:
    rstripped = line.rstrip()
    if rstripped.endswith("$$") or rstripped.endswith("\\]"):
        return True
    return any(line.startswith(f"\\end{{{env}}}") for env in _MATH_ENVS)


def _is_table_close(line: str) -> bool:
    return any(line.startswith(f"\\end{{{env}}}") for env in _TABLE_ENVS)


def _consume_atomic_env(
    ordered_lines: list[tuple[str, int]], start: int, mode: str
) -> tuple[list[str], int]:
    """Consume a contiguous atomic environment starting at ``start``.

    Returns (collected_lines, lines_consumed). Stops at the matching closing
    delimiter. For markdown tables, stops at the first non-table line. Falls back
    to a single line if no close is found so a malformed block still stays atomic
    rather than corrupting paragraphs.
    """
    collected: list[str] = []
    i = start
    n = len(ordered_lines)
    while i < n:
        line, _page = ordered_lines[i]
        collected.append(line)

        if mode == "math":
            if i == start and line.count("$$") >= 2:
                return collected, 1
            if i > start and _is_display_math_close(line):
                return collected, i - start + 1
        else:  # table
            if _is_table_env_open(ordered_lines[start][0]) and _is_table_close(line) and i > start:
                return collected, i - start + 1
            if i > start and _is_markdown_table_row(ordered_lines[start][0]):
                if not _is_markdown_table_row(line) and line.strip():
                    return collected[:-1], i - start
        i += 1

    return collected, i - start + 1
