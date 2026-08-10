"""The eval harness (plan.md Phase 6, step 2): config in, metrics table
out. The backbone of Phases 6, 7, 10, 11, 16, and 17 -- every one of them
scores a config against the golden set the same way, through this module.

`run_eval` takes a `search_fn` callback rather than constructing a real
retriever itself, so the (genuinely project-specific) composition logic
here -- per-stratum breakdown, negative exclusion, config-hash caching --
is fully unit-tested with a fake, independent of live Qdrant/`bm25s`
(`tests/test_harness.py`). `experiments/06_retrieval_ablation.py` supplies
the real `search_fn`, built from Phase 5's `HybridRetriever`.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from aegis.eval.metrics import DEFAULT_METRICS, build_qrels, build_run, compute_metrics
from aegis.types import GoldenQuery

SearchFn = Callable[[GoldenQuery], list[tuple[str, float]]]


@dataclass(frozen=True)
class RetrievalConfig:
    """Every axis of the Phase 6 ablation grid, plus the tunables Phase 6
    step 4 sweeps within a strategy. `name` is a human-readable label only
    -- excluded from the hash, since it doesn't change what gets measured
    (Phase 6 exit criterion: reruns any config from a hash)."""

    name: str
    chunking_strategy: str
    contextual_enrichment: bool
    retrieval_strategy: str
    encoder: str
    top_k: int = 10
    rrf_k: int = 60
    rrf_dense_weight: float = 1.0
    rrf_sparse_weight: float = 1.0
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    def config_hash(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "name"}
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class EvalResult:
    config: RetrievalConfig
    config_hash: str
    overall: dict[str, float]
    per_stratum: dict[str, dict[str, float]]
    n_queries: int


def run_eval(
    config: RetrievalConfig,
    queries: list[GoldenQuery],
    search_fn: SearchFn,
    metrics: list[str] = DEFAULT_METRICS,
    cache_dir: Path | None = None,
    max_workers: int = 8,
) -> EvalResult:
    """Retrieval-quality metrics only -- negatives (`is_negative=True`)
    have no ground-truth chunk to retrieve by construction (they're
    correct-refusal cases, Phase 10's job) and are excluded before
    `search_fn` is ever called on them.
    """
    positives = [q for q in queries if not q.is_negative]

    cache_key = _cache_key(config, positives)
    cache_path = _cache_path(cache_dir, cache_key) if cache_dir is not None else None
    if cache_path is not None and cache_path.exists():
        return _read_cache(cache_path)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        retrieved_lists = list(pool.map(search_fn, positives))
    retrieved = {q.id: hits for q, hits in zip(positives, retrieved_lists)}

    overall = _score(positives, retrieved, metrics)
    per_stratum: dict[str, dict[str, float]] = {}
    for stratum in sorted({q.stratum.value for q in positives if q.stratum is not None}):
        stratum_queries = [q for q in positives if q.stratum is not None and q.stratum.value == stratum]
        per_stratum[stratum] = _score(stratum_queries, retrieved, metrics)

    result = EvalResult(
        config=config,
        config_hash=config.config_hash(),
        overall=overall,
        per_stratum=per_stratum,
        n_queries=len(positives),
    )
    if cache_path is not None:
        _write_cache(cache_path, result)
    return result


def load_cached(cache_dir: Path, config_hash: str) -> EvalResult | None:
    """Look up a cached result by config hash alone -- exact query-set
    match not required, since the caller (re-running a known config
    against the same frozen golden set) is exactly the case this exists
    for. Returns the most recently written match, or None."""
    matches = sorted(cache_dir.glob(f"{config_hash}__*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return _read_cache(matches[0]) if matches else None


def _score(queries: list[GoldenQuery], retrieved: dict[str, list[tuple[str, float]]], metrics: list[str]) -> dict[str, float]:
    ground_truth = {q.id: q.ground_truth_chunk_ids for q in queries}
    qrels = build_qrels(ground_truth)
    run = build_run({q.id: retrieved[q.id] for q in queries})
    return compute_metrics(qrels, run, metrics)


def _cache_key(config: RetrievalConfig, positives: list[GoldenQuery]) -> str:
    # Scoped to *which queries were evaluated*, not just the config --
    # otherwise two different query sets run under an identical config
    # would silently collide on the same cache file (tests/test_harness.py
    # ::test_run_eval_cache_is_scoped_to_the_evaluated_query_set).
    signature = sorted((q.id, tuple(sorted(q.ground_truth_chunk_ids))) for q in positives)
    queries_hash = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    return f"{config.config_hash()}__{queries_hash}"


def _cache_path(cache_dir: Path, cache_key: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}.json"


def _write_cache(path: Path, result: EvalResult) -> None:
    payload = {
        "config": asdict(result.config),
        "config_hash": result.config_hash,
        "overall": result.overall,
        "per_stratum": result.per_stratum,
        "n_queries": result.n_queries,
    }
    path.write_text(json.dumps(payload, indent=2))


def _read_cache(path: Path) -> EvalResult:
    payload = json.loads(path.read_text())
    return EvalResult(
        config=RetrievalConfig(**payload["config"]),
        config_hash=payload["config_hash"],
        overall=payload["overall"],
        per_stratum=payload["per_stratum"],
        n_queries=payload["n_queries"],
    )
