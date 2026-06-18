"""Evaluation pipeline (Phase 5).

Runs the Agentic RAG system over the mock benchmark dataset and scores the four
tracked metrics with an LLM-as-a-judge (deterministic heuristic fallback when no
API key is configured):

    faithfulness      — answer grounded strictly in retrieved context (zero hallucination)
    context_recall    — all ground-truth facts present in retrieved context
    answer_relevancy  — answer resolves the original question
    context_precision — relevance of the retrieved chunk set

Usage:
    python tests/evaluation/run_eval.py            # offline heuristic judge
    OPENAI_API_KEY=... python tests/evaluation/run_eval.py   # LLM-as-a-judge

Writes per-query and aggregate results to the configured results path
(``tests/evaluation/results.json``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agent.graph import run_agent  # noqa: E402
from src.agent.llm_client import LLMClient  # noqa: E402
from src.common.config import AppConfig, get_config  # noqa: E402
from src.common.logging_setup import get_logger, setup_logging  # noqa: E402
from src.common.schemas import EvaluationReport, EvaluationResult  # noqa: E402
from src.evaluation.dataset import EvalDataset, load_dataset  # noqa: E402
from src.evaluation.judge import LLMJudge  # noqa: E402
from src.ingestion.pipeline import IngestionPipeline  # noqa: E402
from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402

_logger = get_logger("evaluation.run_eval")


def build_retriever(config: AppConfig, corpus_path: Path) -> HybridRetriever:
    """Build a BM25-backed retriever over the benchmark corpus."""
    _document, chunks = IngestionPipeline(config=config).parse_and_split(corpus_path)
    bm25 = BM25Retriever()
    bm25.add_chunks(chunks)
    return HybridRetriever(config=config, bm25=bm25, vector_index=None, reranker=None)


def evaluate_query(
    entry: Any,
    config: AppConfig,
    retriever: HybridRetriever,
    judge: LLMJudge,
    llm: LLMClient,
) -> EvaluationResult:
    """Run the agent for one query and judge the result."""
    state = run_agent(entry.query, config=config, retriever=retriever, llm=llm)
    contexts = state.retrieved_documents

    faithfulness = judge.faithfulness(state.generation, contexts)
    context_recall = judge.context_recall(entry.ground_truth, contexts)
    answer_relevancy = judge.answer_relevance(entry.query, state.generation)
    context_precision = judge.context_precision(contexts, entry.ground_truth)

    passed = (
        faithfulness >= config.evaluation.fail_on.get("faithfulness", 0.85)
        and answer_relevancy >= config.evaluation.fail_on.get("answer_relevancy", 0.75)
        and context_recall >= config.evaluation.fail_on.get("context_recall", 0.75)
    )

    return EvaluationResult(
        query=entry.query,
        answer=state.generation,
        ground_truth=entry.ground_truth,
        contexts=[c.content for c in contexts],
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_recall=context_recall,
        context_precision=context_precision,
        passed=passed,
        notes=f"category={entry.category} grade={state.grade_decision} "
              f"hallucination={state.hallucination_decision}",
    )


def aggregate(results: list[EvaluationResult], config: AppConfig) -> EvaluationReport:
    """Compute mean metrics and write the report."""
    def _mean(values: list[float]) -> float:
        return mean(values) if values else 0.0

    report = EvaluationReport(
        results=results,
        mean_faithfulness=_mean([r.faithfulness for r in results]),
        mean_answer_relevancy=_mean([r.answer_relevancy for r in results]),
        mean_context_recall=_mean([r.context_recall for r in results]),
        mean_context_precision=_mean([r.context_precision for r in results]),
        total_queries=len(results),
    )
    _write_report(report, config)
    return report


def _write_report(report: EvaluationReport, config: AppConfig) -> None:
    out_path = config.eval_results_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump()
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _logger.info("Evaluation results written to %s", out_path)


def _print_report(report: EvaluationReport, judge: LLMJudge) -> None:
    mode = "LLM-as-a-judge" if not judge.offline else "offline heuristic judge"
    print(f"\n=== Evaluation Report ({mode}) ===")
    print(f"Queries: {report.total_queries}")
    print(
        f"  faithfulness     : {report.mean_faithfulness:.3f}\n"
        f"  answer_relevancy : {report.mean_answer_relevancy:.3f}\n"
        f"  context_recall   : {report.mean_context_recall:.3f}\n"
        f"  context_precision: {report.mean_context_precision:.3f}"
    )
    print("\n-- per-query --")
    for r in report.results:
        flag = "PASS" if r.passed else "FAIL"
        print(
            f"  [{flag}] faith={r.faithfulness:.2f} recall={r.context_recall:.2f} "
            f"rel={r.answer_relevancy:.2f} prec={r.context_precision:.2f}  {r.query[:55]}"
        )


def main(argv: list[str]) -> int:
    setup_logging()
    config = get_config()
    dataset: EvalDataset = load_dataset(config=config)

    corpus_path = _PROJECT_ROOT / dataset.source_paper
    if not corpus_path.exists():
        _logger.error("Benchmark corpus not found: %s", corpus_path)
        return 1

    retriever = build_retriever(config, corpus_path)
    llm = LLMClient(config=config)
    judge = LLMJudge(config=config, llm=llm)

    results: list[EvaluationResult] = []
    for entry in dataset.queries:
        _logger.info("Evaluating %s: %s", entry.id, entry.query[:60])
        try:
            results.append(evaluate_query(entry, config, retriever, judge, llm))
        except Exception as exc:  # noqa: BLE001 - one bad query must not abort the run
            _logger.error("Evaluation failed for %s: %s", entry.id, exc, exc_info=True)
            results.append(EvaluationResult(query=entry.query, notes=f"error: {exc}"))

    report = aggregate(results, config)
    _print_report(report, judge)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
