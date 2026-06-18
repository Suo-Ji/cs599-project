"""Phase 4 agent demonstration.

Runs the LangGraph agent end-to-end on the fixture corpus:
  * parses + splits the sample paper into chunks
  * builds a real BM25-backed retriever (sparse path; no embedding model needed)
  * executes the self-correcting graph and prints the full decision trace

Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL) to route the LLM through a
real provider; otherwise the LLMClient runs in deterministic offline mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agent.graph import run_agent  # noqa: E402
from src.agent.llm_client import LLMClient  # noqa: E402
from src.common.config import get_config  # noqa: E402
from src.common.logging_setup import setup_logging  # noqa: E402
from src.ingestion.pipeline import IngestionPipeline  # noqa: E402
from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402

_FIXTURE = _PROJECT_ROOT / "tests" / "data" / "sample_paper.txt"
_QUESTIONS = [
    "What is the RMSE of the GNN backbone?",
    "Which model has the lowest NLL?",
]


def main() -> int:
    setup_logging()
    config = get_config()

    _document, chunks = IngestionPipeline(config=config).parse_and_split(_FIXTURE)
    bm25 = BM25Retriever()
    bm25.add_chunks(chunks)
    retriever = HybridRetriever(config=config, bm25=bm25, vector_index=None, reranker=None)
    llm = LLMClient(config=config)

    for question in _QUESTIONS:
        print(f"\n{'='*70}\nQ: {question}\n{'='*70}")
        state = run_agent(question, config=config, retriever=retriever, llm=llm)
        print(f"route={state.query_route}  grade={state.grade_decision}  "
              f"hallucination={state.hallucination_decision}")
        print(f"retries={state.retry_count}  generation_attempts={state.generation_attempts}")
        print("\nDecision trace:")
        for entry in state.evaluation_logs:
            print(f"  - {entry}")
        print(f"\nAnswer: {state.generation[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
