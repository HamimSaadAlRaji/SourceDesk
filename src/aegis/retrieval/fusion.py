"""Rank fusion (plan.md Phase 5, steps 4-5): combine independently-ranked
result lists (dense, sparse) into one ranking.

RRF and linear fusion encode different assumptions about what a retrieval
score means:

- **RRF** is rank-based and scale-free. It never looks at the raw score,
  only position, so a BM25 score of 8.3 and a cosine score of 0.83 combine
  with no normalization step and no risk of one method's score range
  silently dominating the other's.
- **Linear fusion** looks at the raw score, so it needs each list
  independently min-max normalized to [0, 1] first, or a method with a
  wider natural score spread would dominate the sum regardless of rank
  quality. It also carries real position information RRF discards
  (how much better rank 1 is than rank 2), which is why it's the
  comparison rather than an obviously-worse alternative -- Phase 6 measures
  which one actually wins on this corpus.

Both are ~20 lines each; the interesting cost is design, not code (plan.md
§1) -- see docs/explainers/05-lexical-dense-and-fusion.md.
"""

from __future__ import annotations

from aegis.config import settings
from aegis.types import Chunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Chunk, float]]],
    weights: list[float] | None = None,
    k: int = settings.retrieval.rrf_k,
) -> list[tuple[Chunk, float]]:
    """score(d) = sum_r( weight_r / (k + rank_r(d)) ), rank 1-indexed.

    A chunk absent from a given list contributes 0 for that list -- it is
    not penalized beyond simply not benefiting from that method's vote.
    """
    weights = weights if weights is not None else [1.0] * len(ranked_lists)

    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, (chunk, _raw_score) in enumerate(ranked_list, start=1):
            chunks[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + weight / (k + rank)

    return _sorted_by_score(scores, chunks)


def linear_fusion(
    ranked_lists: list[list[tuple[Chunk, float]]],
    weights: list[float] | None = None,
) -> list[tuple[Chunk, float]]:
    """score(d) = sum_r( weight_r * normalized_score_r(d) ), min-max
    normalized to [0, 1] within each list before combining.

    A list where every score is equal (max == min) has no discriminating
    signal to normalize -- every member is treated as equally maximal
    (1.0) rather than raising a ZeroDivisionError or arbitrarily zeroing
    the whole list out.
    """
    weights = weights if weights is not None else [1.0] * len(ranked_lists)

    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranked_list, weight in zip(ranked_lists, weights):
        if not ranked_list:
            continue
        raw_scores = [raw_score for _chunk, raw_score in ranked_list]
        lo, hi = min(raw_scores), max(raw_scores)
        spread = hi - lo
        for chunk, raw_score in ranked_list:
            normalized = 1.0 if spread == 0 else (raw_score - lo) / spread
            chunks[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + weight * normalized

    return _sorted_by_score(scores, chunks)


def _sorted_by_score(scores: dict[str, float], chunks: dict[str, Chunk]) -> list[tuple[Chunk, float]]:
    ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [(chunks[chunk_id], scores[chunk_id]) for chunk_id in ranked_ids]
