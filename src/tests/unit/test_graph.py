"""Unit tests for the LangGraph control flow (Phase 4).

Exercises every conditional edge and both retry caps using a scripted LLM stub
and an in-memory retriever, so the graph's routing logic is verified without any
network or model dependency.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END

from src.agent.graph import AgentNodes, GraphState, build_graph
from src.common.config import get_config
from src.common.schemas import DocumentChunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid import HybridRetriever


def _chunk(cid: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=cid, content=text, source_file="paper.pdf",
        section_title="Results", page_number=1,
    )


_CORPUS = [_chunk("c1", "The GNN backbone achieves RMSE 0.389 and NLL 1.140.")]


class ScriptedLLM:
    """Returns canned JSON/text responses from a queue, one per call."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []
        self.invoke_calls = 0
        self.invoke_json_calls = 0

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        self.invoke_calls += 1
        self.calls.append((system_prompt, user_prompt))
        return str(self._next())

    def invoke_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.invoke_json_calls += 1
        self.calls.append((system_prompt, user_prompt))
        return dict(self._next())  # type: ignore[arg-type]

    def _next(self) -> Any:
        if not self._responses:
            raise AssertionError("ScriptedLLM response queue exhausted")
        return self._responses.pop(0)


def _retriever(corpus: list[DocumentChunk] = _CORPUS) -> HybridRetriever:
    bm25 = BM25Retriever()
    bm25.add_chunks(corpus)
    return HybridRetriever(config=get_config(), bm25=bm25, vector_index=None, reranker=None)


# ----------------------------------------------------------------------


def test_happy_path_grounds_on_first_generation() -> None:
    # analyzer -> grader(relevant) -> generate -> hallucination(grounded) -> END
    llm = ScriptedLLM([
        {"route": "experimental", "processed_query": "GNN RMSE value"},          # analyzer
        {"score": 0.9, "decision": "relevant", "rationale": "metric present"},    # grader
        "The GNN backbone achieves RMSE 0.389.",                                   # generate
        {"grounded_fraction": 1.0, "decision": "grounded", "unsupported_claims": []},  # hallucination
    ])
    compiled = build_graph(config=get_config(), retriever=_retriever(), llm=llm)
    result = compiled.invoke({"question": "What is the GNN RMSE?"})

    assert result["grade_decision"] == "relevant"
    assert result["hallucination_decision"] == "grounded"
    assert result["retry_count"] == 0
    assert result["generation_attempts"] == 1
    assert "RMSE 0.389" in result["generation"]
    nodes = [e["node"] for e in result["evaluation_logs"]]
    assert nodes == ["query_analyzer", "retrieve", "document_grader", "generate_answer", "hallucination_checker"]


def test_irrelevant_triggers_rewrite_until_relevant() -> None:
    # grader(irrelevant) -> rewrite -> retrieve -> grader(relevant) -> generate -> END
    llm = ScriptedLLM([
        {"route": "default", "processed_query": "RMSE value"},                    # analyzer
        {"score": 0.2, "decision": "irrelevant", "rationale": "metric missing"},  # grader (1)
        {"route": "default", "processed_query": "GNN RMSE"},                      # rewrite
        {"score": 0.9, "decision": "relevant", "rationale": "found"},             # grader (2)
        "RMSE 0.389",                                                              # generate
        {"grounded_fraction": 1.0, "decision": "grounded", "unsupported_claims": []},
    ])
    compiled = build_graph(config=get_config(), retriever=_retriever(), llm=llm)
    result = compiled.invoke({"question": "What is the GNN RMSE?"})

    assert result["retry_count"] == 1
    assert result["grade_decision"] == "relevant"
    nodes = [e["node"] for e in result["evaluation_logs"]]
    assert nodes.count("query_rewriter") == 1
    assert nodes.count("retrieve") == 2  # initial + post-rewrite


def test_rewrite_cap_forces_generation() -> None:
    config = get_config()
    original_cap = config.agent.max_rewrite_retries
    config.agent.max_rewrite_retries = 1  # tight cap for the test
    try:
        llm = ScriptedLLM([
            {"route": "default", "processed_query": "q"},                         # analyzer
            {"score": 0.1, "decision": "irrelevant", "rationale": "x"},           # grader (1)
            {"route": "default", "processed_query": "q2"},                        # rewrite (retry->1)
            {"score": 0.1, "decision": "irrelevant", "rationale": "x"},           # grader (2): cap reached
            "Forced best-effort answer.",                                          # generate (forced)
            {"grounded_fraction": 1.0, "decision": "grounded", "unsupported_claims": []},
        ])
        nodes = AgentNodes(config=config, retriever=_retriever(), llm=llm)
        # Routing decision at cap must fall through to generate.
        assert nodes.route_after_grading({"grade_decision": "irrelevant", "retry_count": 1}) == "generate"
    finally:
        config.agent.max_rewrite_retries = original_cap


def test_hallucination_not_grounded_re_generates_then_halts() -> None:
    config = get_config()
    # gen_limit=2: first gen -> not_grounded -> regen -> (grounded) END.
    llm = ScriptedLLM([
        {"route": "default", "processed_query": "GNN RMSE"},                      # analyzer
        {"score": 0.9, "decision": "relevant", "rationale": "ok"},                # grader
        "The RMSE is 0.999 with extra invented detail.",                          # generate (1)
        {"grounded_fraction": 0.2, "decision": "not_grounded", "unsupported_claims": ["0.999"]},
        "The GNN backbone achieves RMSE 0.389.",                                   # generate (2)
        {"grounded_fraction": 1.0, "decision": "grounded", "unsupported_claims": []},
    ])
    compiled = build_graph(config=config, retriever=_retriever(), llm=llm, generation_attempt_limit=2)
    result = compiled.invoke({"question": "What is the GNN RMSE?"})

    assert result["generation_attempts"] == 2
    assert result["hallucination_decision"] == "grounded"
    assert result["retry_count"] == 0
    assert "RMSE 0.389" in result["generation"]


def test_generation_cap_falls_back_to_retrieve() -> None:
    config = get_config()
    # gen_limit=1: not_grounded once -> over cap -> route to retrieve.
    nodes = AgentNodes(config=config, retriever=_retriever(), llm=None, generation_attempt_limit=1)  # type: ignore[arg-type]
    decision = nodes.route_after_hallucination({"hallucination_decision": "not_grounded", "generation_attempts": 1})
    assert decision == "retrieve"


def test_evaluation_trace_is_append_only() -> None:
    # All nodes must contribute to evaluation_logs (not overwrite each other).
    llm = ScriptedLLM([
        {"route": "default", "processed_query": "q"},
        {"score": 0.9, "decision": "relevant", "rationale": "ok"},
        "Answer.",
        {"grounded_fraction": 1.0, "decision": "grounded", "unsupported_claims": []},
    ])
    compiled = build_graph(config=get_config(), retriever=_retriever(), llm=llm)
    result = compiled.invoke({"question": "q"})
    node_names = [e["node"] for e in result["evaluation_logs"]]
    # Each of these nodes ran exactly once and appears in order.
    assert node_names == ["query_analyzer", "retrieve", "document_grader", "generate_answer", "hallucination_checker"]
    # No log entry is silently dropped.
    assert len(result["evaluation_logs"]) == 5


def test_llm_timeout_does_not_crash_graph() -> None:
    # Per spec: "Do not let token timeouts crash the execution graph."
    # The graph must degrade gracefully and still return a final state.
    class ExplodingLLM:
        offline = False

        def invoke(self, system_prompt, user_prompt):
            raise TimeoutError("API token timeout")

        def invoke_json(self, system_prompt, user_prompt):
            raise TimeoutError("API token timeout")

    llm = ExplodingLLM()
    compiled = build_graph(config=get_config(), retriever=_retriever(), llm=llm)  # type: ignore[arg-type]
    # Must complete (not raise) with degraded, well-defined output.
    result = compiled.invoke({"question": "What is the GNN RMSE?"})
    assert "generation" in result
    assert result["grade_decision"] == "irrelevant"  # safe default from grader fallback
