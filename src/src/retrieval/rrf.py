"""Reciprocal Rank Fusion (RRF) — Layer 2.

Fuses multiple ranked result lists into a single ranking using the standard RRF
formula:

    RRF_Score(d) = sum_{m in M} 1 / (k + r_m(d))

where ``r_m(d)`` is the rank of document ``d`` in method ``m`` (1-indexed) and
``k`` is a smoothing constant (default 60). Higher RRF score means stronger
cross-method agreement, so documents ranked highly by several methods surface
first. This is a pure, deterministic function with no I/O.
"""

from __future__ import annotations

from typing import Hashable, Iterable, Sequence

# Default RRF smoothing constant from the original formulation.
DEFAULT_RRF_K: int = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Hashable]],
    k: int = DEFAULT_RRF_K,
) -> list[tuple[Hashable, float]]:
    """Fuse multiple ranked id lists via RRF.

    Args:
        ranked_lists: one ordered list of document ids per retrieval method.
            Each list is a ranking (best first). Duplicates within a single list
            are collapsed to the first (best) rank.
        k: RRF smoothing constant; ``k=60`` by default per the spec.

    Returns:
        ``(id, rrf_score)`` pairs sorted by descending score, then by first-seen
        order to keep the fusion deterministic on ties.
    """
    if k <= 0:
        raise ValueError(f"RRF constant k must be positive, got {k}.")

    scores: dict[Hashable, float] = {}
    first_seen: dict[Hashable, int] = {}
    order = 0

    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0.0
                first_seen[doc_id] = order
                order += 1
            # A document absent from a method contributes nothing for that method.
            scores[doc_id] += 1.0 / (k + rank)

    # Sort: higher score first; ties broken by first-seen order for determinism.
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
    return ordered


def fuse_scores(
    ranked_lists: Iterable[Sequence[Hashable]],
    k: int = DEFAULT_RRF_K,
    top_n: int | None = None,
) -> list[tuple[Hashable, float]]:
    """Convenience wrapper: fuse then optionally truncate to ``top_n``."""
    fused = reciprocal_rank_fusion(list(ranked_lists), k=k)
    return fused[:top_n] if top_n is not None else fused
