"""Unit tests for Reciprocal Rank Fusion (Phase 3, TDD-first per spec).

Validates the RRF math against hand-computed expectations and edge cases. Pure
function, no retrieval dependencies required.
"""

from __future__ import annotations

import math

import pytest

from src.retrieval.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion


def test_rrf_single_list_preserves_ranking() -> None:
    fused = reciprocal_rank_fusion([["d1", "d2", "d3"]])
    ids = [doc_id for doc_id, _ in fused]
    assert ids == ["d1", "d2", "d3"]
    assert math.isclose(fused[0][1], 1.0 / (DEFAULT_RRF_K + 1), rel_tol=1e-9)
    assert math.isclose(fused[1][1], 1.0 / (DEFAULT_RRF_K + 2), rel_tol=1e-9)


def test_rrf_favors_cross_method_agreement() -> None:
    # d1 is rank-1 in BOTH methods -> 2/(k+1), strictly the highest score.
    dense = ["d1", "d2", "d3"]
    sparse = ["d1", "d3", "d2"]
    fused = dict(reciprocal_rank_fusion([dense, sparse]))
    expected = {
        "d1": 1.0 / (DEFAULT_RRF_K + 1) + 1.0 / (DEFAULT_RRF_K + 1),
        "d2": 1.0 / (DEFAULT_RRF_K + 2) + 1.0 / (DEFAULT_RRF_K + 3),
        "d3": 1.0 / (DEFAULT_RRF_K + 3) + 1.0 / (DEFAULT_RRF_K + 2),
    }
    for doc_id, score in expected.items():
        assert math.isclose(fused[doc_id], score, rel_tol=1e-9)
    ranked_ids = [doc_id for doc_id, _ in reciprocal_rank_fusion([dense, sparse])]
    assert ranked_ids[0] == "d1"  # rank-1 in both -> wins


def test_rrf_document_absent_from_a_method_gets_no_contribution() -> None:
    dense = ["d1", "d2"]
    sparse = ["d2"]  # d1 missing from sparse
    fused = dict(reciprocal_rank_fusion([dense, sparse]))
    assert math.isclose(fused["d1"], 1.0 / (DEFAULT_RRF_K + 1), rel_tol=1e-9)
    assert math.isclose(
        fused["d2"],
        1.0 / (DEFAULT_RRF_K + 2) + 1.0 / (DEFAULT_RRF_K + 1),
        rel_tol=1e-9,
    )


def test_rrf_union_of_all_documents() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert {doc_id for doc_id, _ in fused} == {"a", "b", "c"}


def test_rrf_empty_inputs() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_invalid_k_raises() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["d1"]], k=0)


def test_rrf_k_parameter_changes_scores() -> None:
    fused_k60 = dict(reciprocal_rank_fusion([["d1", "d2"]], k=60))
    fused_k1 = dict(reciprocal_rank_fusion([["d1", "d2"]], k=1))
    assert fused_k1["d1"] > fused_k60["d1"]
    ratio_k1 = fused_k1["d1"] / fused_k1["d2"]
    ratio_k60 = fused_k60["d1"] / fused_k60["d2"]
    assert ratio_k1 > ratio_k60  # smaller k => steeper rank dropoff


def test_rrf_ties_break_by_first_seen_order() -> None:
    # d1 and d2 get identical scores (both rank 1 in their respective lists).
    fused = reciprocal_rank_fusion([["d1"], ["d2"]])
    ids = [doc_id for doc_id, _ in fused]
    assert ids == ["d1", "d2"]  # d1 seen first, so it wins the tie
