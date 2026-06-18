"""Phase 1 unit tests: validate Pydantic schemas construct and enforce constraints.

These tests run without external services (no DB, no network, no models) so the
Phase 1 baseline can be verified immediately after `pip install -r requirements.txt`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.common.schemas import (
    AgentState,
    DocumentChunk,
    DocumentGrade,
    EvaluationResult,
    HallucinationGrade,
    QueryAnalysis,
)


# --- DocumentChunk -------------------------------------------------------

def test_document_chunk_construction() -> None:
    chunk = DocumentChunk(
        id="paperA::0",
        content="Evidential deep learning estimates higher-order distributions.",
        source_file="edl_paper.pdf",
        section_title="Methodology",
        page_number=4,
        keywords=["EDL", "NLL"],
        chunk_index=0,
        token_count=64,
    )
    assert chunk.id == "paperA::0"
    assert chunk.page_range == (0, 0)
    flat = chunk.to_filterable_metadata()
    assert flat["source_file"] == "edl_paper.pdf"
    assert flat["keywords"] == "EDL,NLL"


def test_document_chunk_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        DocumentChunk(id="x", content="", source_file="a.pdf")


def test_document_chunk_rejects_negative_page() -> None:
    with pytest.raises(ValidationError):
        DocumentChunk(id="x", content="ok", source_file="a.pdf", page_number=-1)


# --- AgentState ----------------------------------------------------------

def test_agent_state_defaults_are_fresh_containers() -> None:
    state_a = AgentState(question="q1")
    state_b = AgentState(question="q2")
    state_a.retrieved_documents.append(
        DocumentChunk(id="c", content="t", source_file="a.pdf")
    )
    state_a.retry_count += 1
    # New instances must not share mutable defaults.
    assert state_b.retrieved_documents == []
    assert state_b.retry_count == 0


def test_agent_state_rejects_negative_retry() -> None:
    with pytest.raises(ValidationError):
        AgentState(question="q", retry_count=-1)


# --- Grading schemas (deterministic binary) ------------------------------

def test_document_grade_bounds() -> None:
    grade = DocumentGrade(score=0.9, decision="relevant")
    assert grade.decision == "relevant"
    with pytest.raises(ValidationError):
        DocumentGrade(score=1.5, decision="relevant")
    with pytest.raises(ValidationError):
        DocumentGrade(score=0.5, decision="maybe")  # type: ignore[arg-type]


def test_hallucination_grade_bounds() -> None:
    grade = HallucinationGrade(grounded_fraction=0.2, decision="not_grounded")
    assert grade.unsupported_claims == []
    with pytest.raises(ValidationError):
        HallucinationGrade(grounded_fraction=-0.1, decision="grounded")


def test_query_analysis_route_literal() -> None:
    analysis = QueryAnalysis(processed_query="calibration error definition")
    assert analysis.route == "default"
    with pytest.raises(ValidationError):
        QueryAnalysis(processed_query="x", route="bogus")  # type: ignore[arg-type]


# --- EvaluationResult ----------------------------------------------------

def test_evaluation_result_defaults() -> None:
    result = EvaluationResult(query="What is PICP?")
    assert result.passed is False
    assert result.faithfulness == 0.0
    assert 0.0 <= result.answer_relevancy <= 1.0
