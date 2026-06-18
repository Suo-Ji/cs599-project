"""Unit tests for the PDF/text parser (Phase 2).

Exercises heading detection, block classification, and the atomic preservation
of equations and tables during parsing. Uses a synthetic text fixture so no PDF
or external dependency is required.
"""

from __future__ import annotations

from src.ingestion.parser import (
    TextBlock,
    _detect_heading,
    parse_document,
)

_FIXTURE = (
    "Abstract\n\n"
    "This paper studies uncertainty estimation with EDL.\n\n"
    "1 Introduction\n\n"
    "We compare a BNN baseline against evidential models.\n\n"
    "2 Methodology\n\n"
    "2.1 Loss\n\n"
    "The evidential loss is defined below.\n\n"
    "$$\n\\mathcal{L} = x^2 + y^2\n$$\n\n"
    "3 Experiments\n\n"
    "| Model | RMSE |\n"
    "|-------|------|\n"
    "| EDL   | 0.41 |\n"
)


def test_heading_detection_levels() -> None:
    assert _detect_heading("1 Introduction") == ("Introduction", 1)
    assert _detect_heading("2.1 Loss") == ("Loss", 2)
    assert _detect_heading("Abstract") == ("Abstract", 1)
    assert _detect_heading("INTRODUCTION") == ("INTRODUCTION", 1)
    # A sentence is not a heading.
    assert _detect_heading("This is a full sentence about calibration.") is None
    assert _detect_heading("") is None


def test_parse_sections_and_blocks(tmp_path) -> None:
    path = tmp_path / "paper.txt"
    path.write_text(_FIXTURE, encoding="utf-8")

    document = parse_document(path)
    titles = [s.title for s in document.sections]
    assert "Abstract" in titles
    assert "Introduction" in titles
    assert "Loss" in titles  # subsection 2.1
    assert "Experiments" in titles
    assert document.page_count == 1


def test_equation_and_table_are_atomic_blocks(tmp_path) -> None:
    path = tmp_path / "paper.txt"
    path.write_text(_FIXTURE, encoding="utf-8")

    document = parse_document(path)
    all_blocks: list[TextBlock] = [b for s in document.sections for b in s.blocks]

    equations = [b for b in all_blocks if b.kind == "equation"]
    tables = [b for b in all_blocks if b.kind == "table"]
    assert len(equations) == 1, "exactly one display equation expected"
    assert len(tables) == 1, "exactly one metrics table expected"
    assert equations[0].is_atomic is True
    assert tables[0].is_atomic is True
    # The equation block must contain both delimiters (intact, unsplit).
    assert "$$" in equations[0].text
    assert "\\mathcal{L}" in equations[0].text
    # The table block must contain the RMSE row intact.
    assert "RMSE" in tables[0].text
