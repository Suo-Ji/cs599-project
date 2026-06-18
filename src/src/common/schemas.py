"""Pydantic data schemas for the Agentic RAG system.

Defines the canonical data contracts used across all four architectural layers:
  * DocumentChunk      — ingestion output unit
  * AgentState         — LangGraph shared state (typed dict semantics, Pydantic-backed)
  * EvaluationResult   — single-query metric record from the evaluation pipeline

Plus small supporting schemas referenced by the agent nodes (query routing,
document grading, hallucination checking). All fields carry explicit types.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================
# Layer 1 — ingestion
# ============================================================

class DocumentChunk(BaseModel):
    """A single semantic chunk produced by the ingestion pipeline.

    Carries the raw text plus the structured metadata injected at ingestion time
    (source file, enclosing section, page range, extracted domain keywords).
    This object is what gets embedded, stored, retrieved, and graded downstream.
    """

    id: str = Field(..., description="Stable, deterministic chunk identifier.")
    content: str = Field(..., min_length=1, description="Chunk text content.")
    source_file: str = Field(..., description="Origin PDF file name.")
    section_title: str = Field(
        default="unknown",
        description="Logical section the chunk belongs to (e.g. 'Methodology').",
    )
    page_number: int = Field(
        default=0,
        ge=0,
        description="Starting page number of the chunk (1-indexed).",
    )
    page_range: tuple[int, int] = Field(
        default=(0, 0),
        description="(start_page, end_page); set when a chunk spans multiple pages.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Extracted domain keywords (metrics, model abbreviations).",
    )
    chunk_index: int = Field(
        default=0,
        ge=0,
        description="Position of this chunk within its source document.",
    )
    token_count: int = Field(
        default=0,
        ge=0,
        description="Token length of content, computed by the active tokenizer.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extension slot for unstructured metadata not covered above."
    )

    model_config = {"frozen": False}

    def to_filterable_metadata(self) -> dict[str, Any]:
        """Project the chunk onto the flat metadata dict Chroma indexes.

        Only flat, scalar-or-list values are returned; nested dicts are excluded
        so that Chroma scalar filters work correctly.
        """
        flat: dict[str, Any] = {
            "source_file": self.source_file,
            "section_title": self.section_title,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
        }
        if self.keywords:
            flat["keywords"] = ",".join(self.keywords)
        return flat


# ============================================================
# Layer 3 — LangGraph agent state
# ============================================================

class AgentState(BaseModel):
    """Shared state object threaded through the LangGraph control flow.

    Field semantics (matching the spec's typed-dict contract):
      * question             — original user query
      * processed_query      — rewritten/optimized query produced by the Router
      * retrieved_documents  — chunks surfaced by the retrieval engine
      * generation           — final or in-progress LLM answer
      * retry_count          — Query_Rewriter invocations so far (capped)
      * evaluation_logs      — append-only trace of grading/hallucination decisions

    Lists use ``default_factory`` so each graph run starts with a clean, mutable
    container. Merge semantics across parallel branches are handled in Phase 4
    via LangGraph reducers; this schema only defines the shape.
    """

    question: str = Field(default="", description="Original user query.")
    processed_query: str = Field(
        default="",
        description="Router-optimized query actually sent to the retrieval engine.",
    )
    query_route: str = Field(
        default="default",
        description="Classified intent label from the Query_Analyzer/Router node.",
    )
    retrieved_documents: list[DocumentChunk] = Field(
        default_factory=list,
        description="Chunks returned by Retrieve_&_Rerank.",
    )
    retrieval_scores: list[float] = Field(
        default_factory=list,
        description="Post-rerank relevance scores aligned with retrieved_documents.",
    )
    generation: str = Field(default="", description="LLM-generated answer.")
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of Query_Rewriter attempts executed so far.",
    )
    generation_attempts: int = Field(
        default=0,
        ge=0,
        description="Number of Generate_Answer attempts (bounds hallucination retries).",
    )
    grade_decision: Optional[str] = Field(
        default=None,
        description="Document_Grader output: 'relevant' | 'irrelevant'.",
    )
    hallucination_decision: Optional[str] = Field(
        default=None,
        description="Hallucination_Checker output: 'grounded' | 'not_grounded'.",
    )
    evaluation_logs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Append-only trace of deterministic node decisions.",
    )

    @field_validator("retry_count")
    @classmethod
    def _retry_within_limit(cls, value: int) -> int:
        if value < 0:
            raise ValueError("retry_count must be non-negative.")
        return value


# ============================================================
# Supporting schemas for agent nodes
# ============================================================

class QueryAnalysis(BaseModel):
    """Output of the Query_Analyzer/Router node."""

    route: Literal["default", "methodology", "experimental", "definition"] = Field(
        default="default",
        description="Classified query intent used to bias retrieval filters.",
    )
    processed_query: str = Field(
        ..., description="Rewritten query optimized for retrieval."
    )
    needs_definition: bool = Field(
        default=False,
        description="Whether the query asks for a term definition (affects chunking).",
    )


class DocumentGrade(BaseModel):
    """Output of the Document_Grader node — deterministic binary judgement."""

    score: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated relevance score in [0, 1]."
    )
    decision: Literal["relevant", "irrelevant"] = Field(
        ..., description="Binary pass/fail derived from score vs threshold."
    )
    rationale: str = Field(
        default="",
        description="Short objective justification (no evaluative adjectives).",
    )


class HallucinationGrade(BaseModel):
    """Output of the Hallucination_Checker node — deterministic binary judgement."""

    grounded_fraction: float = Field(
        ..., ge=0.0, le=1.0, description="Share of claims traceable to context."
    )
    decision: Literal["grounded", "not_grounded"] = Field(
        ..., description="Binary pass/fail derived from grounded_fraction vs threshold."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Verbatim claims not found in retrieved context.",
    )


# ============================================================
# Layer 4 — evaluation
# ============================================================

class EvaluationResult(BaseModel):
    """A single query's evaluation record produced by the evaluation pipeline."""

    query: str = Field(..., description="The academic query under evaluation.")
    answer: str = Field(default="", description="Generated answer under test.")
    ground_truth: str = Field(
        default="", description="Reference answer from the benchmark dataset."
    )
    contexts: list[str] = Field(
        default_factory=list,
        description="Retrieved context chunks used to produce the answer.",
    )
    faithfulness: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Share of answer grounded in contexts."
    )
    answer_relevancy: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How well the answer addresses the query."
    )
    context_recall: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Share of ground-truth facts present in retrieved contexts.",
    )
    context_precision: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Precision of retrieved chunks ranked against ground truth.",
    )
    passed: bool = Field(
        default=False, description="Whether all configured metric floors were met."
    )
    notes: str = Field(default="", description="Free-form evaluator annotations.")


class EvaluationReport(BaseModel):
    """Aggregate report over a full benchmark run."""

    results: list[EvaluationResult] = Field(default_factory=list)
    mean_faithfulness: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_answer_relevancy: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_context_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_context_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    total_queries: int = Field(default=0, ge=0)
