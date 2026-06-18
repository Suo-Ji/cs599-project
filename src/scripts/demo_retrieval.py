"""Phase 3 retrieval demonstration.

Exercises the hybrid pipeline on the fixture corpus without requiring the heavy
embedding / reranker models:
  * parses + splits the sample paper into chunks
  * builds a real BM25 index (sparse retrieval)
  * fuses BM25 with a simulated dense ranking via RRF (k=60)
  * prints the resulting ranked chunks

When ``sentence-transformers`` models are available, swap the simulated dense
ranking for a real :class:`~src.ingestion.vectorstore.VectorIndex` and pass a
:class:`~src.retrieval.reranker.CrossEncoderReranker` to :class:`HybridRetriever`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.config import get_config  # noqa: E402
from src.common.logging_setup import setup_logging  # noqa: E402
from src.common.schemas import DocumentChunk  # noqa: E402
from src.ingestion.pipeline import IngestionPipeline  # noqa: E402
from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402
from src.retrieval.rrf import reciprocal_rank_fusion  # noqa: E402

_FIXTURE = _PROJECT_ROOT / "tests" / "data" / "sample_paper.txt"
_QUERY = "Which backbone achieves the lowest RMSE and NLL with calibrated intervals?"


class _SimulatedDense:
    """Pretend dense retriever: ranks chunks by lexical overlap with the query.

    Stand-in for the bge embedding model so the full RRF pipeline is runnable
    here. Replace with a real VectorIndex in production.
    """

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = chunks

    def query(self, query_text: str, top_k: int = 20, where=None):
        q_terms = set(query_text.lower().split())
        scored = []
        for chunk in self._chunks:
            terms = set(chunk.content.lower().split())
            overlap = len(q_terms & terms)
            scored.append((chunk, float(overlap)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


def main() -> int:
    setup_logging()
    config = get_config()

    _document, chunks = IngestionPipeline(config=config).parse_and_split(_FIXTURE)
    print(f"Corpus: {len(chunks)} chunks")

    # --- Sparse only (BM25) ---
    bm25 = BM25Retriever()
    bm25.add_chunks(chunks)
    sparse_results = bm25.query(_QUERY, top_k=5)
    print("\n-- BM25 (sparse) top-5 --")
    for chunk, score in sparse_results:
        print(f"  {score:7.3f}  [{chunk.section_title}] {chunk.content[:70]}...")

    # --- Hybrid (simulated dense + BM25, fused by RRF) ---
    retriever = HybridRetriever(
        config=config,
        vector_index=_SimulatedDense(chunks),  # type: ignore[arg-type]
        bm25=bm25,
        reranker=None,  # cross-encoder omitted (model not downloaded here)
    )
    fused = retriever.retrieve(_QUERY)
    print("\n-- Hybrid (RRF-fused) top-5 --")
    for chunk, score in fused:
        print(f"  {score:.5f}  [{chunk.section_title}] {chunk.content[:70]}...")

    # --- RRF raw scores (for transparency) ---
    dense_ids = [c.id for c, _ in _SimulatedDense(chunks).query(_QUERY, top_k=5)]
    sparse_ids = [c.id for c, _ in sparse_results]
    raw = reciprocal_rank_fusion([dense_ids, sparse_ids], k=config.retrieval.rrf_k)
    print("\n-- RRF fused id order (k=60) --")
    print("  " + ", ".join(doc_id for doc_id, _ in raw[:5]))
    print(f"\n(No reranker: cross-encoder '{config.retrieval.rerank_model}' "
          f"not loaded in this environment.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
