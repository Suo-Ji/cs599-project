r"""LangGraph agentic control flow (Layer 3).

State-driven graph with self-correction. Nodes:

    START
      -> Query_Analyzer/Router
      -> Retrieve_&_Rerank
      -> Document_Grader --- relevant -----------------> Generate_Answer
                     \                                             |
                      \-- irrelevant (< max retries) -> Query_Rewriter -> Retrieve
                       \-- irrelevant (>= max retries) -> Generate_Answer (forced)
                                                                    |
                                                            Hallucination_Checker
                                                            grounded -----> END
                                                            not_grounded ->
                                                                Generate_Answer (if < gen limit)
                                                                Retrieve     (otherwise)

Control-flow bounds prevent infinite loops: the rewrite loop is capped at
``max_rewrite_retries`` and the generate/hallucination loop is capped by a
separate generation-attempt limit. Once either limit is reached the graph
emits its best-effort answer and halts.

The graph uses a TypedDict state with accumulator reducers for the list/dict
fields, which is the idiom LangGraph supports for branch merges. The Pydantic
:class:`~src.common.schemas.AgentState` remains the canonical schema elsewhere.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from ..common.config import AppConfig, get_config
from ..common.logging_setup import get_logger
from ..common.schemas import (
    AgentState,
    DocumentChunk,
    DocumentGrade,
    HallucinationGrade,
    QueryAnalysis,
)
from ..retrieval.hybrid import HybridRetriever
from .llm_client import LLMClient, LLMError
from . import prompts

_logger: logging.Logger = get_logger("agent.graph")

# Default ceiling on generate/hallucination retries (independent of rewrite cap).
_GENERATION_ATTEMPT_LIMIT: int = 2


# ------------------------------------------------------------------
# Graph state (TypedDict with reducers)
# ------------------------------------------------------------------


def _overwrite(_old: Any, new: Any) -> Any:
    """Reducer that replaces the field with the latest value."""
    return new


def _append(old: Any, new: Any) -> list[Any]:
    """Reducer that concatenates lists (for the append-only evaluation trace)."""
    return list(old or []) + list(new or [])


class GraphState(TypedDict, total=False):
    """Mutable state threaded through the LangGraph control flow.

    ``Annotated[..., _overwrite]`` fields keep only the most recent value across
    a branch merge, matching the Pydantic AgentState semantics.
    """

    question: Annotated[str, _overwrite]
    processed_query: Annotated[str, _overwrite]
    query_route: Annotated[str, _overwrite]
    retrieved_documents: Annotated[list[DocumentChunk], _overwrite]
    retrieval_scores: Annotated[list[float], _overwrite]
    generation: Annotated[str, _overwrite]
    retry_count: Annotated[int, _overwrite]
    generation_attempts: Annotated[int, _overwrite]
    grade_decision: Annotated[Optional[str], _overwrite]
    hallucination_decision: Annotated[Optional[str], _overwrite]
    evaluation_logs: Annotated[list[dict[str, Any]], _append]


# ------------------------------------------------------------------
# Node container
# ------------------------------------------------------------------


class AgentNodes:
    """Holds shared dependencies and implements each graph node as a method.

    Each node returns a partial ``GraphState`` delta; LangGraph merges it.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        retriever: Optional[HybridRetriever] = None,
        llm: Optional[LLMClient] = None,
        generation_attempt_limit: int = _GENERATION_ATTEMPT_LIMIT,
    ) -> None:
        self._config = config or get_config()
        self._retriever = retriever
        self._llm = llm or LLMClient(self._config)
        self._gen_limit = generation_attempt_limit
        self._grade_threshold = self._config.retrieval.grade_score_threshold
        self._grounded_threshold = self._config.agent.grounded_claim_threshold
        self._max_rewrites = self._config.agent.max_rewrite_retries

    # ------------------------------------------------------------------
    # Node 1: Query_Analyzer / Router
    # ------------------------------------------------------------------

    def query_analyzer(self, state: GraphState) -> dict[str, Any]:
        question = state.get("question", "")
        user_prompt = (
            f"User question: {question}\n\n"
            "Classify the intent and produce an optimized retrieval query."
        )
        data = self._safe_json(
            prompts.QUERY_ANALYZER_SYSTEM, user_prompt,
            default={"route": "default", "processed_query": question},
        )
        analysis = QueryAnalysis(**_coerce(data, QueryAnalysis))

        self._log("query_analyzer", route=analysis.route, query=analysis.processed_query)
        return {
            "processed_query": analysis.processed_query or question,
            "query_route": analysis.route,
            "retrieved_documents": [],
            "evaluation_logs": [{"node": "query_analyzer", "route": analysis.route}],
        }

    # ------------------------------------------------------------------
    # Node 2: Retrieve & Rerank
    # ------------------------------------------------------------------

    def retrieve(self, state: GraphState) -> dict[str, Any]:
        query = state.get("processed_query") or state.get("question", "")
        if self._retriever is None:
            self._log("retrieve", status="no_retriever")
            return {"retrieved_documents": [], "retrieval_scores": []}
        try:
            results = self._retriever.retrieve(query)
            chunks = [c for c, _ in results]
            scores = [s for _, s in results]
        except Exception as exc:  # noqa: BLE001 - retrieval must not crash the graph
            _logger.error("Retrieval node failed: %s", exc, exc_info=True)
            chunks, scores = [], []
        self._log("retrieve", query=query, n=len(chunks))
        return {
            "retrieved_documents": chunks,
            "retrieval_scores": scores,
            "evaluation_logs": [{"node": "retrieve", "n_results": len(chunks)}],
        }

    # ------------------------------------------------------------------
    # Node 3: Document_Grader
    # ------------------------------------------------------------------

    def document_grader(self, state: GraphState) -> dict[str, Any]:
        question = state.get("question", "")
        docs = state.get("retrieved_documents", [])
        context = _format_context(docs)

        user_prompt = (
            f"User question: {question}\n\n"
            f"Retrieved context documents:\n{context}"
        )
        data = self._safe_json(prompts.DOCUMENT_GRADER_SYSTEM, user_prompt, default={
            "score": 0.0, "decision": "irrelevant", "rationale": "grader unavailable",
        })
        grade = DocumentGrade(**_coerce(data, DocumentGrade))

        decision = grade.decision
        if grade.decision == "relevant" and grade.score < self._grade_threshold:
            decision = "irrelevant"

        self._log("document_grader", score=grade.score, decision=decision)
        return {
            "grade_decision": decision,
            "evaluation_logs": [
                {"node": "document_grader", "score": grade.score, "decision": decision}
            ],
        }

    # ------------------------------------------------------------------
    # Node 4: Query_Rewriter
    # ------------------------------------------------------------------

    def query_rewriter(self, state: GraphState) -> dict[str, Any]:
        question = state.get("question", "")
        user_prompt = (
            f"Original user question: {question}\n"
            f"Previous query: {state.get('processed_query', '')}\n\n"
            "Rewrite the query to improve retrieval."
        )
        data = self._safe_json(
            prompts.QUERY_REWRITER_SYSTEM, user_prompt,
            default={"route": "default", "processed_query": question},
        )
        analysis = QueryAnalysis(**_coerce(data, QueryAnalysis))

        retry_count = int(state.get("retry_count", 0)) + 1
        self._log("query_rewriter", retry=retry_count, query=analysis.processed_query)
        return {
            "processed_query": analysis.processed_query or question,
            "query_route": analysis.route,
            "retry_count": retry_count,
            "retrieved_documents": [],
            "evaluation_logs": [
                {"node": "query_rewriter", "retry_count": retry_count}
            ],
        }

    # ------------------------------------------------------------------
    # Node 5: Generate_Answer
    # ------------------------------------------------------------------

    def generate_answer(self, state: GraphState) -> dict[str, Any]:
        question = state.get("question", "")
        docs = state.get("retrieved_documents", [])
        context = _format_context(docs)

        user_prompt = (
            f"User question: {question}\n\n"
            f"Context documents:\n{context}\n\nAnswer the question using only the context."
        )
        generation = self._safe_text(prompts.GENERATE_ANSWER_SYSTEM, user_prompt)
        attempts = int(state.get("generation_attempts", 0)) + 1
        self._log("generate_answer", attempt=attempts, length=len(generation))
        return {
            "generation": generation,
            "generation_attempts": attempts,
            "evaluation_logs": [
                {"node": "generate_answer", "attempt": attempts}
            ],
        }

    # ------------------------------------------------------------------
    # Node 6: Hallucination_Checker
    # ------------------------------------------------------------------

    def hallucination_checker(self, state: GraphState) -> dict[str, Any]:
        docs = state.get("retrieved_documents", [])
        context = _format_context(docs)
        generation = state.get("generation", "")

        user_prompt = (
            f"Generated answer:\n{generation}\n\n"
            f"Retrieved context documents:\n{context}"
        )
        data = self._safe_json(prompts.HALLUCINATION_CHECKER_SYSTEM, user_prompt, default={
            "grounded_fraction": 1.0, "decision": "grounded", "unsupported_claims": [],
        })
        grade = HallucinationGrade(**_coerce(data, HallucinationGrade))

        decision = grade.decision
        if grade.decision == "grounded" and grade.grounded_fraction < self._grounded_threshold:
            decision = "not_grounded"

        self._log("hallucination_checker", fraction=grade.grounded_fraction, decision=decision)
        return {
            "hallucination_decision": decision,
            "evaluation_logs": [
                {
                    "node": "hallucination_checker",
                    "grounded_fraction": grade.grounded_fraction,
                    "decision": decision,
                }
            ],
        }

    # ------------------------------------------------------------------
    # Safe LLM call wrappers (graph must survive token timeouts)
    # ------------------------------------------------------------------

    def _safe_json(self, system_prompt: str, user_prompt: str, default: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._llm.invoke_json(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001 - graph must survive any LLM failure
            _logger.error("LLM JSON call failed; using fallback default: %s", exc)
            return default

    def _safe_text(self, system_prompt: str, user_prompt: str) -> str:
        try:
            return self._llm.invoke(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001 - graph must survive any LLM failure
            _logger.error("LLM text call failed; using fallback answer: %s", exc)
            return "Answer unavailable: the language model did not respond."

    # ------------------------------------------------------------------
    # Routing functions (pure, used by conditional edges)
    # ------------------------------------------------------------------

    def route_after_grading(self, state: GraphState) -> str:
        """relevant -> generate ; irrelevant & under cap -> rewrite ; else -> generate."""
        if state.get("grade_decision") == "relevant":
            return "generate"
        if state.get("retry_count", 0) < self._max_rewrites:
            return "rewrite"
        _logger.warning("Rewrite cap reached (%d); forcing generation.", self._max_rewrites)
        return "generate"

    def route_after_hallucination(self, state: GraphState) -> str:
        """grounded -> END ; not_grounded & under cap -> generate ; else -> retrieve."""
        if state.get("hallucination_decision") == "grounded":
            return "end"
        if state.get("generation_attempts", 0) < self._gen_limit:
            return "generate"
        _logger.warning("Generation cap reached (%d); re-retrieving.", self._gen_limit)
        return "retrieve"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log(node: str, **fields: Any) -> None:
        _logger.info("Node %s -> %s", node, fields)


# ------------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------------


def build_graph(
    config: Optional[AppConfig] = None,
    retriever: Optional[HybridRetriever] = None,
    llm: Optional[LLMClient] = None,
    generation_attempt_limit: int = _GENERATION_ATTEMPT_LIMIT,
) -> Any:
    """Construct and compile the LangGraph agent workflow.

    Returns a compiled graph runnable via ``graph.invoke({...})`` or streaming.
    """
    config = config or get_config()
    nodes = AgentNodes(
        config=config,
        retriever=retriever,
        llm=llm,
        generation_attempt_limit=generation_attempt_limit,
    )

    graph = StateGraph(GraphState)

    graph.add_node("query_analyzer", nodes.query_analyzer)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("document_grader", nodes.document_grader)
    graph.add_node("query_rewriter", nodes.query_rewriter)
    graph.add_node("generate", nodes.generate_answer)
    graph.add_node("hallucination_checker", nodes.hallucination_checker)

    graph.add_edge(START, "query_analyzer")
    graph.add_edge("query_analyzer", "retrieve")
    graph.add_edge("retrieve", "document_grader")
    graph.add_edge("query_rewriter", "retrieve")
    graph.add_edge("generate", "hallucination_checker")

    graph.add_conditional_edges(
        "document_grader",
        nodes.route_after_grading,
        {"generate": "generate", "rewrite": "query_rewriter"},
    )
    graph.add_conditional_edges(
        "hallucination_checker",
        nodes.route_after_hallucination,
        {"end": END, "generate": "generate", "retrieve": "retrieve"},
    )

    compiled = graph.compile()
    _logger.info("LangGraph agent compiled (max_rewrites=%d, gen_limit=%d).",
                 config.agent.max_rewrite_retries, generation_attempt_limit)
    return compiled


# ------------------------------------------------------------------
# State projection helpers
# ------------------------------------------------------------------


def run_agent(
    question: str,
    config: Optional[AppConfig] = None,
    retriever: Optional[HybridRetriever] = None,
    llm: Optional[LLMClient] = None,
    generation_attempt_limit: int = _GENERATION_ATTEMPT_LIMIT,
) -> AgentState:
    """Convenience runner: build, invoke, and project the final state."""
    compiled = build_graph(
        config=config, retriever=retriever, llm=llm,
        generation_attempt_limit=generation_attempt_limit,
    )
    result: dict[str, Any] = compiled.invoke({"question": question})
    return _project_to_agent_state(result)


def _project_to_agent_state(result: dict[str, Any]) -> AgentState:
    """Map the TypedDict graph output onto the Pydantic AgentState schema."""
    return AgentState(
        question=result.get("question", ""),
        processed_query=result.get("processed_query", ""),
        query_route=result.get("query_route", "default"),
        retrieved_documents=result.get("retrieved_documents", []),
        retrieval_scores=result.get("retrieval_scores", []),
        generation=result.get("generation", ""),
        retry_count=int(result.get("retry_count", 0)),
        generation_attempts=int(result.get("generation_attempts", 0)),
        grade_decision=result.get("grade_decision"),
        hallucination_decision=result.get("hallucination_decision"),
        evaluation_logs=result.get("evaluation_logs", []),
    )


# ------------------------------------------------------------------
# Internal utilities
# ------------------------------------------------------------------


def _format_context(chunks: list[DocumentChunk]) -> str:
    if not chunks:
        return "(no documents retrieved)"
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] ({chunk.source_file} :: {chunk.section_title} :: p{chunk.page_number})\n{chunk.content}"
        )
    return "\n\n".join(lines)


def _coerce(data: dict[str, Any], schema_cls: type) -> dict[str, Any]:
    """Keep only the keys present in the schema to tolerate model extras."""
    allowed = set(schema_cls.model_fields.keys())
    return {k: v for k, v in data.items() if k in allowed}
