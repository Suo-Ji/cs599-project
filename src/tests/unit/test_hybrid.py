"""Unit tests for the hybrid retrieval engine (Phase 3, step 3).

Verifies that a mock query returns correctly ordered chunks. BM25 is exercised
for real (no model needed); the dense path is stubbed via a fake vector index,
and the reranker path is exercised via an injected fake. RRF fusion is validated
end-to-end through the integration.
"""

from __future__ import annotations

from typing import Any, Optional

from src.common.config import get_config
from src.common.schemas import DocumentChunk
from src.retrieval.bm25_retriever import BM25Retriever, tokenize
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker


def _chunk(cid: str, content: str, section: str = "Body") -> DocumentChunk:
    return DocumentChunk(
        id=cid, content=content, source_file="paper.pdf",
        section_title=section, page_number=1,
    )


_CORPUS = [
    _chunk("c1", "The GNN backbone achieves the lowest RMSE and NLL on the regression benchmark."),
    _chunk("c2", "The MLP baseline consists of four fully connected layers."),
    _chunk("c3", "We report NLL alongside PICP and MPIW calibration metrics."),
]


class FakeVectorIndex:
    """Drop-in stub for VectorIndex exposing only ``query``."""

    def __init__(self, ranking: list[tuple[DocumentChunk, float]]) -> None:
        self._ranking = ranking
        self.last_where: Optional[dict[str, Any]] = None
        self.last_top_k: Optional[int] = None

    def query(
        self,
        query_text: str,
        top_k: int = 20,
        where: Optional[dict[str, Any]] = None,
    ) -> list[tuple[DocumentChunk, float]]:
        self.last_where = where
        self.last_top_k = top_k
        return self._ranking[:top_k]


class FakeReranker:
    """Drop-in stub for CrossEncoderReranker that inverts candidate order."""

    def __init__(self, top_scores: Optional[dict[str, float]] = None) -> None:
        self._top_scores = top_scores or {}
        self.calls: int = 0

    def rerank(
        self, query: str, candidates: list[DocumentChunk], top_n: int = 5
    ) -> list[tuple[DocumentChunk, float]]:
        self.calls += 1
        # Prefer explicitly-scored chunks; remaining candidates keep their order.
        scored = []
        for chunk in candidates:
            score = self._top_scores.get(chunk.id, 0.0)
            scored.append((chunk, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_n]


# ----------------------------------------------------------------------


def test_tokenize_preserves_metrics_and_numbers() -> None:
    tokens = set(tokenize("The GNN achieves RMSE=0.41 and NLL of 1.2."))
    assert "gnn" in tokens
    assert "rmse" in tokens
    assert "nll" in tokens
    assert "0.41" in tokens  # numeric value survives
    assert "the" not in tokens  # stopword removed


def test_bm25_ranks_exact_match_first() -> None:
    bm25 = BM25Retriever()
    bm25.add_chunks(_CORPUS)
    results = bm25.query("Which model has the lowest RMSE and NLL?", top_k=3)
    ids = [c.id for c, _ in results]
    assert ids[0] == "c1"  # only c1 contains RMSE and NLL


def test_hybrid_fuses_dense_and_sparse_rrf_order() -> None:
    # BM25 ranks c1 first (it has both RMSE and NLL); the dense stub disagrees,
    # ranking c2 first and c1 second. After RRF with k=60:
    #   c1 = 1/(k+2) + 1/(k+1)   (dense r2, sparse r1)
    #   c2 = 1/(k+1) + 1/(k+3)   (dense r1, sparse r3)
    # 1/62 + 1/61 > 1/61 + 1/63  ->  c1 wins the fusion.
    dense_ranking = [(_CORPUS[1], 0.9), (_CORPUS[0], 0.8), (_CORPUS[2], 0.5)]
    fake_dense = FakeVectorIndex(dense_ranking)

    bm25 = BM25Retriever()
    bm25.add_chunks(_CORPUS)

    retriever = HybridRetriever(
        config=get_config(), vector_index=fake_dense, bm25=bm25, reranker=None
    )
    results = retriever.retrieve("Which model has the lowest RMSE and NLL?")
    ids = [c.id for c, _ in results]
    assert ids[0] == "c1", f"RRF should resolve disagreement in c1's favor; got {ids}"
    assert set(ids) == {"c1", "c2", "c3"}


def test_hybrid_passes_metadata_filter_to_dense() -> None:
    dense_ranking = [(_CORPUS[0], 0.9)]
    fake_dense = FakeVectorIndex(dense_ranking)
    bm25 = BM25Retriever()
    bm25.add_chunks(_CORPUS)
    retriever = HybridRetriever(config=get_config(), vector_index=fake_dense, bm25=bm25)

    filt = {"section_title": "Results"}
    retriever.retrieve("query", where=filt)
    assert fake_dense.last_where == filt


def test_hybrid_reranker_path_respects_top_n() -> None:
    # Provide dense ranking so fusion yields all 3 candidates.
    dense_ranking = [(_CORPUS[0], 0.9), (_CORPUS[1], 0.7), (_CORPUS[2], 0.5)]
    fake_dense = FakeVectorIndex(dense_ranking)
    bm25 = BM25Retriever()
    bm25.add_chunks(_CORPUS)

    fake_reranker = FakeReranker(top_scores={"c3": 9.9})  # force c3 to the top
    retriever = HybridRetriever(
        config=get_config(), vector_index=fake_dense, bm25=bm25, reranker=fake_reranker
    )
    results = retriever.retrieve("query", final_top_n=2)
    assert fake_reranker.calls == 1
    ids = [c.id for c, _ in results]
    assert ids == ["c3", "c1"]  # c3 forced first, then RRF runner-up
    assert len(results) == 2


def test_hybrid_returns_empty_on_no_candidates() -> None:
    fake_dense = FakeVectorIndex([])  # dense returns nothing
    bm25 = BM25Retriever()  # empty BM25 index
    retriever = HybridRetriever(config=get_config(), vector_index=fake_dense, bm25=bm25)
    assert retriever.retrieve("anything") == []


def test_reranker_degrades_gracefully_on_failure() -> None:
    class FailingReranker:
        def rerank(self, query, candidates, top_n=5):
            raise RuntimeError("model unavailable")

    dense_ranking = [(_CORPUS[0], 0.9), (_CORPUS[1], 0.7)]
    fake_dense = FakeVectorIndex(dense_ranking)
    bm25 = BM25Retriever()
    bm25.add_chunks(_CORPUS)
    retriever = HybridRetriever(
        config=get_config(), vector_index=fake_dense, bm25=bm25,
        reranker=FailingReranker(),  # type: ignore[arg-type]
    )
    # Should fall back to RRF order instead of raising.
    results = retriever.retrieve("RMSE NLL benchmark")
    assert results  # non-empty fallback
