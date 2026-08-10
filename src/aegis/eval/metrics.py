"""Retrieval metrics via `ranx` (plan.md Phase 6, step 1).

`ranx` is imported -- Precision@k, Recall@k, MRR, MAP@k, NDCG@k are solved
problems (CLAUDE.md §2). Verified once against a hand-computed 5-document
example (`tests/test_metrics.py`) rather than trusted blindly; every other
caller in this project (the eval harness, the ablation grid) goes through
this thin wrapper instead of touching `ranx` directly.
"""

from __future__ import annotations

from ranx import Qrels, Run
from ranx import evaluate as ranx_evaluate

DEFAULT_METRICS = ["precision@10", "recall@10", "mrr", "map@10", "ndcg@10"]


def build_qrels(ground_truth: dict[str, list[str]]) -> Qrels:
    """`ground_truth`: query_id -> list of relevant chunk_ids (binary
    relevance -- the golden set doesn't grade degrees of relevance)."""
    return Qrels({query_id: dict.fromkeys(chunk_ids, 1) for query_id, chunk_ids in ground_truth.items()})


def build_run(retrieved: dict[str, list[tuple[str, float]]]) -> Run:
    """`retrieved`: query_id -> ranked list of (chunk_id, score)."""
    return Run({query_id: dict(hits) for query_id, hits in retrieved.items()})


def compute_metrics(qrels: Qrels, run: Run, metrics: list[str] = DEFAULT_METRICS) -> dict[str, float]:
    # ranx.evaluate returns a bare numpy scalar for a single metric but a
    # dict for multiple -- normalized here so callers never special-case
    # it. Values are cast to plain floats so results are directly
    # JSON-serializable for the harness's cache (Phase 6 step 2).
    results = ranx_evaluate(qrels, run, metrics)
    if not isinstance(results, dict):
        results = {metrics[0]: results}
    return {name: float(value) for name, value in results.items()}
