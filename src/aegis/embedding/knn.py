"""Brute-force exact k-NN — ground truth for measuring HNSW's recall loss
(plan.md Phase 3). Not a competing implementation of ANN search; this is
measurement infrastructure, deliberately the simplest correct thing, used
once to produce `docs/results/ann-recall-vs-latency.md` and then trusted.

Assumes L2-normalized vectors, so cosine similarity reduces to a dot product
— see `docs/explainers/03-embeddings-distance-and-ann.md` for the proof.
"""

from __future__ import annotations

import numpy as np


def brute_force_knn(query: np.ndarray, corpus: np.ndarray, k: int) -> np.ndarray:
    """Indices into `corpus` of the k nearest neighbors to `query`, ranked
    nearest-first."""
    scores = corpus @ query
    k = min(k, len(scores))
    top_k = np.argpartition(-scores, k - 1)[:k]
    return top_k[np.argsort(-scores[top_k])]


def brute_force_knn_batch(queries: np.ndarray, corpus: np.ndarray, k: int) -> np.ndarray:
    """Same as `brute_force_knn`, vectorized over multiple queries at once.
    Returns an (n_queries, k) array of corpus indices."""
    scores = queries @ corpus.T
    k = min(k, scores.shape[1])
    top_k = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    row_order = np.argsort(-np.take_along_axis(scores, top_k, axis=1), axis=1)
    return np.take_along_axis(top_k, row_order, axis=1)


def recall_at_k(approx: list[list[int]], exact: list[list[int]]) -> float:
    """Mean fraction of each query's exact top-k neighbors that also appear
    in its approximate top-k, order-independent.

    Only meaningful when the corpus has no (or few) exact score ties near
    the k-th boundary — see `recall_at_k_score_aware` for corpora that do
    (this project's does: ~42% of chunks share an exact-duplicate embedding
    with at least one other chunk, e.g. 1,354 issues whose entire chunk text
    is the literal string "_No response_", a GitHub issue-template
    placeholder). With heavy ties, ID-set recall penalizes an approximate
    search for returning a *different, equally correct* tied neighbor than
    the arbitrary one brute-force KNN happened to return, which isn't a
    real error. Kept for corpora/tests where it's the right tool.
    """
    per_query = []
    for approx_ids, exact_ids in zip(approx, exact):
        if not exact_ids:
            continue
        overlap = len(set(approx_ids) & set(exact_ids))
        per_query.append(overlap / len(exact_ids))
    return sum(per_query) / len(per_query) if per_query else 0.0


def recall_at_k_score_aware(approx_scores: list[list[float]], exact_kth_scores: list[float], k: int, epsilon: float = 1e-4) -> float:
    """Score-threshold recall: an approximate result counts as correct if
    its similarity score is >= the true k-th-best score (within float
    tolerance), not only if it happens to share the exact same ID as
    brute-force KNN's arbitrary pick among tied candidates. The standard
    fix for measuring ANN recall on a corpus with real duplicate content.

    Always divides by `k`, the number of neighbors actually requested — not
    by `len(scores)`, the number returned. Dividing by the returned count
    was an earlier version's bug: a method returning fewer results than
    asked for (e.g. naive post-filtering discarding most of its candidate
    pool) scored a perfect 1.0 as long as the few it did return were
    correct, exactly hiding the failure mode `docs/results/
    prefilter-vs-postfilter.md` exists to measure. An empty result list is
    scored 0.0, not skipped.
    """
    per_query = []
    for scores, kth in zip(approx_scores, exact_kth_scores):
        correct = sum(1 for s in scores if s >= kth - epsilon)
        per_query.append(min(correct, k) / k)
    return sum(per_query) / len(per_query) if per_query else 0.0
