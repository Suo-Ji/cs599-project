"""Evaluation run over the 7 real arXiv papers (Phase 5b).

Mirrors run_eval.py but builds the retriever from all PDFs in data/pdfs/ rather
than a single text fixture, and loads the real-paper query dataset
(tests/data/eval_queries_real.json). Scores the four metrics via the LLM-as-judge
(heuristic fallback offline) and writes results to tests/evaluation/results_real.json.

Usage:
    python tests/evaluation/run_eval_real.py
    OPENAI_API_KEY=... python tests/evaluation/run_eval_real.py   # LLM judge
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agent.graph import run_agent  # noqa: E402
from src.agent.llm_client import LLMClient  # noqa: E402
from src.common.config import AppConfig, get_config  # noqa: E402
from src.common.logging_setup import get_logger, setup_logging  # noqa: E402
from src.common.schemas import EvaluationReport, EvaluationResult  # noqa: E402
from src.evaluation.dataset import load_dataset  # noqa: E402
from src.evaluation.judge import LLMJudge  # noqa: E402
from src.ingestion.parser import parse_document  # noqa: E402
from src.ingestion.splitter import SemanticSplitter  # noqa: E402
from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402

_logger = get_logger("evaluation.run_eval_real")

_DATASET = _PROJECT_ROOT / "tests" / "data" / "eval_queries_real.json"
_PDF_DIR = _PROJECT_ROOT / "data" / "pdfs"
_RESULTS = _PROJECT_ROOT / "tests" / "evaluation" / "results_real.json"


def build_retriever(config: AppConfig) -> HybridRetriever:
    """Build a BM25 retriever over every real PDF in data/pdfs/."""
    splitter = SemanticSplitter(config)
    chunks = []
    for pdf in sorted(_PDF_DIR.glob("*.pdf")):
        if pdf.stat().st_size == 0:
            continue
        try:
            document = parse_document(pdf)
            chunks.extend(splitter.split_document(document))
        except Exception as exc:  # noqa: BLE001 - per-file resilience
            _logger.error("Failed to parse %s: %s", pdf.name, exc)
    _logger.info("Built retriever over %d chunks from %d real PDFs.", len(chunks), len(list(_PDF_DIR.glob('*.pdf'))))
    bm25 = BM25Retriever()
    bm25.add_chunks(chunks)
    return HybridRetriever(config=config, bm25=bm25, vector_index=None, reranker=None)


def evaluate_query(entry, config, retriever, judge, llm) -> EvaluationResult:
    state = run_agent(entry.query, config=config, retriever=retriever, llm=llm)
    contexts = state.retrieved_documents

    # Retrieval accuracy: did the top chunk come from the target paper?
    top_source = contexts[0].source_file if contexts else ""
    target_hit = entry.target_arxiv_id in top_source if hasattr(entry, "target_arxiv_id") else None

    faithfulness = judge.faithfulness(state.generation, contexts)
    context_recall = judge.context_recall(entry.ground_truth, contexts)
    answer_relevancy = judge.answer_relevance(entry.query, state.generation)
    context_precision = judge.context_precision(contexts, entry.ground_truth)

    return EvaluationResult(
        query=entry.query,
        answer=state.generation,
        ground_truth=entry.ground_truth,
        contexts=[c.content for c in contexts],
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_recall=context_recall,
        context_precision=context_precision,
        notes=f"target={entry.target_arxiv_id} top_hit={top_source} target_hit={target_hit}",
    )


def main() -> int:
    setup_logging()
    config = get_config()
    dataset = load_dataset(path=_DATASET, config=config)

    retriever = build_retriever(config)
    llm = LLMClient(config=config)
    judge = LLMJudge(config=config, llm=llm)

    results: list[EvaluationResult] = []
    for entry in dataset.queries:
        _logger.info("Evaluating %s (target %s)", entry.id, entry.target_arxiv_id)
        try:
            results.append(evaluate_query(entry, config, retriever, judge, llm))
        except Exception as exc:  # noqa: BLE001
            _logger.error("Eval failed for %s: %s", entry.id, exc, exc_info=True)
            results.append(EvaluationResult(query=entry.query, notes=f"error: {exc}"))

    def _mean(xs): return mean(xs) if xs else 0.0
    report = EvaluationReport(
        results=results,
        mean_faithfulness=_mean([r.faithfulness for r in results]),
        mean_answer_relevancy=_mean([r.answer_relevancy for r in results]),
        mean_context_recall=_mean([r.context_recall for r in results]),
        mean_context_precision=_mean([r.context_precision for r in results]),
        total_queries=len(results),
    )
    _RESULTS.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")

    # Target-paper hit accuracy (retrieval correctness on real corpus).
    hits = sum(1 for r in results if "target_hit=True" in (r.notes or ""))
    mode = "LLM-as-judge" if not judge.offline else "offline heuristic judge"
    print(f"\n=== Real-Paper Evaluation Report ({mode}) ===")
    print(f"Queries: {report.total_queries}  |  Corpus: 7 real arXiv PDFs ({len(results)} results)")
    print(f"  faithfulness     : {report.mean_faithfulness:.3f}")
    print(f"  answer_relevancy : {report.mean_answer_relevancy:.3f}")
    print(f"  context_recall   : {report.mean_context_recall:.3f}")
    print(f"  context_precision: {report.mean_context_precision:.3f}")
    print(f"  target-paper hit : {hits}/{report.total_queries} (top chunk from the correct paper)")
    print("\n-- per-query --")
    for r in results:
        hit = "target_hit=True" in (r.notes or "")
        print(f"  [{'HIT ' if hit else 'miss'}] recall={r.context_recall:.2f} prec={r.context_precision:.2f} "
              f"faith={r.faithfulness:.2f}  {r.query[:50]}")
    print(f"\nResults: {_RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
