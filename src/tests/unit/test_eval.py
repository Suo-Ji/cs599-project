"""Unit tests for the evaluation layer (Phase 5).

Validates the deterministic metric scorers, the LLM-judge heuristic fallback,
dataset loading, and aggregation — all without network or model access.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.config import get_config
from src.common.schemas import DocumentChunk, EvaluationResult
from src.evaluation.dataset import load_dataset
from src.evaluation.judge import LLMJudge
from src.evaluation import metrics

_FIXTURE_CORPUS = Path(__file__).resolve().parents[1] / "data" / "sample_paper.txt"


def _chunk(cid: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=cid, content=text, source_file="paper.pdf",
        section_title="Results", page_number=1,
    )


# --- Heuristic metric scorers ------------------------------------------


def test_extract_key_terms_captives_numbers_and_acronyms() -> None:
    terms = metrics.extract_key_terms("The GNN achieves RMSE 0.389 and NLL 1.140.")
    assert "gnn" in terms
    assert "0.389" in terms
    assert "1.140" in terms
    assert "achieves" in terms  # word longer than 4 chars


def test_faithfulness_perfect_when_answer_subsumed_by_context() -> None:
    context = ["The GNN backbone achieves RMSE 0.389 and NLL 1.140 in the benchmark."]
    answer = "The GNN backbone achieves RMSE 0.389."
    assert metrics.score_faithfulness(answer, context) >= 0.99


def test_faithfulness_zero_when_answer_uses_unseen_numbers() -> None:
    context = ["The MLP baseline uses four fully connected layers."]
    answer = "The GNN backbone achieves RMSE 0.389."
    assert metrics.score_faithfulness(answer, context) < 0.5


def test_context_recall_fractions_present_terms() -> None:
    context = ["GNN achieves RMSE 0.389 NLL 1.140."]
    truth = "The GNN achieves RMSE 0.389 NLL 1.140 PICP 0.90."
    recall = metrics.score_context_recall(truth, context)
    # Most key terms present; only 0.90 / picp missing -> recall strictly < 1.
    assert 0.0 < recall < 1.0


def test_context_precision_counts_relevant_chunks() -> None:
    chunks = [
        _chunk("c1", "GNN RMSE 0.389"),       # relevant
        _chunk("c2", "Unrelated methodology text about optimization"),  # not relevant
    ]
    truth = "The GNN RMSE is 0.389."
    assert metrics.score_context_precision(chunks, truth) == pytest.approx(0.5)


def test_answer_relevance_scales_with_term_overlap() -> None:
    high = metrics.score_answer_relevance(
        "What is the GNN RMSE?", "The GNN RMSE is 0.389."
    )
    low = metrics.score_answer_relevance(
        "What is the GNN RMSE?", "The weather is sunny today."
    )
    assert high > low


def test_scorers_return_zero_on_empty_inputs() -> None:
    assert metrics.score_faithfulness("", ["ctx"]) == 0.0
    assert metrics.score_context_recall("gt", []) == 0.0
    assert metrics.score_answer_relevance("", "ans") == 0.0
    assert metrics.score_context_precision([], "gt") == 0.0


# --- Judge fallback ----------------------------------------------------


def test_judge_uses_heuristic_when_offline(monkeypatch) -> None:
    # Force offline regardless of the user's .env so this test is env-independent.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    config = get_config()
    judge = LLMJudge(config=config)  # no API key -> offline
    assert judge.offline is True
    context = ["The GNN backbone achieves RMSE 0.389."]
    faithfulness = judge.faithfulness("The GNN backbone achieves RMSE 0.389.", context)
    recall = judge.context_recall("The GNN RMSE is 0.389.", context)
    assert 0.0 <= faithfulness <= 1.0
    assert 0.0 <= recall <= 1.0


def test_judge_llm_path_used_when_client_provided() -> None:
    class StubLLM:
        offline = False

        def invoke_json(self, system_prompt, user_prompt):
            if "faithfulness" in system_prompt:
                return {"faithfulness": 0.77}
            if "context_recall" in system_prompt:
                return {"context_recall": 0.88}
            if "answer_relevancy" in system_prompt:
                return {"answer_relevancy": 0.95}
            return {}

    judge = LLMJudge(config=get_config(), llm=StubLLM())  # type: ignore[arg-type]
    assert judge.faithfulness("ans", ["ctx"]) == 0.77
    assert judge.context_recall("gt", ["ctx"]) == 0.88
    assert judge.answer_relevance("q", "a") == 0.95


def test_judge_falls_back_on_malformed_llm_output() -> None:
    class BrokenLLM:
        offline = False

        def invoke_json(self, system_prompt, user_prompt):
            raise ValueError("model returned garbage")

    judge = LLMJudge(config=get_config(), llm=BrokenLLM())  # type: ignore[arg-type]
    # Should not raise; should return the heuristic score.
    score = judge.faithfulness("The GNN RMSE 0.389", ["GNN RMSE 0.389"])
    assert 0.0 <= score <= 1.0


def test_judge_clips_out_of_range_llm_scores() -> None:
    class OverflowLLM:
        offline = False

        def invoke_json(self, system_prompt, user_prompt):
            if "faithfulness" in system_prompt:
                return {"faithfulness": 1.5}
            return {"context_recall": -0.3}

    judge = LLMJudge(config=get_config(), llm=OverflowLLM())  # type: ignore[arg-type]
    assert judge.faithfulness("a", ["c"]) == 1.0
    assert judge.context_recall("g", ["c"]) == 0.0


# --- Dataset loading ---------------------------------------------------


def test_load_dataset_has_ten_queries() -> None:
    dataset = load_dataset()
    assert len(dataset.queries) == 10
    assert all(q.ground_truth for q in dataset.queries)
    assert all(q.key_terms for q in dataset.queries)
    categories = {q.category for q in dataset.queries}
    assert {"experimental", "methodology"}.issubset(categories)


def test_load_dataset_validates_structure(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"queries": [{"id": "x"}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path=bad)
