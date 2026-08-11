"""Phase 7 pool-size sweep (plan.md Phase 7, steps 2-3): for each candidate
pool size, rerank Phase 6's winning dense config (heading-aware chunking,
`bge-large-en-v1.5`, no contextual enrichment) with the cross-encoder and
measure NDCG@5 + reranking latency, plus the no-rerank (bi-encoder-only)
baseline for the same queries.

Dense candidates are fetched once per query at the largest swept pool size,
then sliced locally for smaller pool sizes -- avoids re-querying Qdrant per
pool size and keeps the candidate set identical across the sweep (isolating
the reranking variable from HNSW's own query-to-query variance).
"""

from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.eval.metrics import build_qrels, build_run, compute_metrics
from aegis.rerank.cross_encoder import CrossEncoderReranker
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, GoldenQuery

METRICS = ["ndcg@5", "recall@10", "mrr"]
RESULTS_MD = REPO_ROOT / "docs" / "results" / "rerank-tradeoff.md"
RESULTS_PNG = REPO_ROOT / "docs" / "results" / "rerank-tradeoff.png"
RESULTS_RAW = REPO_ROOT / "docs" / "results" / "rerank-sweep-raw.json"


def load_chunks() -> list[Chunk]:
    path = settings.paths.data_processed / "chunks.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def load_positive_queries() -> list[GoldenQuery]:
    path = settings.paths.data_golden / "golden_set_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        queries = [GoldenQuery(**json.loads(line)) for line in f]
    return [q for q in queries if not q.is_negative]


def main() -> None:
    chunks = load_chunks()
    queries = load_positive_queries()
    print(f"{len(chunks)} chunks, {len(queries)} positive golden queries")

    cache = EmbeddingCache(settings.paths.embedding_cache)
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=cache)
    dense_index = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)

    pool_sizes = sorted(settings.rerank.pool_sizes)
    final_top_k = settings.rerank.final_top_k
    max_pool = pool_sizes[-1]

    print(f"Retrieving top-{max_pool} dense candidates per query (Phase 6 winning config)...")
    per_query_pool: dict[str, list[tuple[Chunk, float]]] = {}
    dense_latencies_ms: list[float] = []
    for q in queries:
        start = time.perf_counter()
        per_query_pool[q.id] = dense_index.search(q.text, top_k=max_pool, exclude_doc_id=q.source_issue)
        dense_latencies_ms.append((time.perf_counter() - start) * 1000)
    print(f"Dense retrieval alone: p50={np.percentile(dense_latencies_ms, 50):.1f}ms p95={np.percentile(dense_latencies_ms, 95):.1f}ms")

    # Free the bi-encoder's MPS-resident weights before loading the
    # cross-encoder -- dense retrieval is done, and Phase 3 already found
    # (decision log, 2026-08-02) that two large transformer models resident
    # on MPS at once on this 16GB M1 Pro compounds swap-driven slowdown into
    # outright thrashing (confirmed here: the first attempt at this sweep
    # left both bge-large and bge-reranker-v2-m3 loaded simultaneously and
    # the process was killed by macOS's scheduler as "stuck" under ~5.7GB
    # swap pressure before a single rerank batch completed).
    del encoder, dense_index, cache
    import gc

    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass

    # CPU, not the configured MPS default: measured twice in this session
    # that reranking on MPS under concurrent system memory pressure (this
    # machine's other running apps, on top of Qdrant + PyTorch) puts the
    # process into macOS's "stuck" scheduler state under swap exhaustion
    # before a single batch completes, even after freeing the bi-encoder
    # first. CPU inference at the same batch size completed a benchmark
    # batch cleanly with no incident, at ~30.5ms/pair vs. MPS's ~7.0ms/pair
    # when MPS did work in isolation -- ~4.4x slower per pair, but reliable,
    # and the ~27,000-pair sweep still finishes in minutes on CPU.
    reranker = CrossEncoderReranker(device="cpu", batch_size=8)

    qrels = build_qrels({q.id: q.ground_truth_chunk_ids for q in queries})

    # No-rerank baseline: bi-encoder dense order alone, truncated to final_top_k.
    baseline_run = build_run({q.id: [(c.id, s) for c, s in per_query_pool[q.id][:final_top_k]] for q in queries})
    baseline_metrics = compute_metrics(qrels, baseline_run, metrics=METRICS)
    print(f"No-rerank (bi-encoder only) baseline: {baseline_metrics}")

    sweep: dict[int, dict] = {}
    for pool_size in pool_sizes:
        run_dict: dict[str, list[tuple[str, float]]] = {}
        latencies_ms: list[float] = []
        for q in queries:
            candidates = per_query_pool[q.id][:pool_size]
            start = time.perf_counter()
            reranked = reranker.rerank(q.text, candidates, top_k=final_top_k)
            latencies_ms.append((time.perf_counter() - start) * 1000)
            run_dict[q.id] = [(c.id, s) for c, s in reranked]

        metrics = compute_metrics(qrels, build_run(run_dict), metrics=METRICS)
        p50 = float(np.percentile(latencies_ms, 50))
        p95 = float(np.percentile(latencies_ms, 95))
        print(
            f"pool={pool_size:>4}  ndcg@5={metrics['ndcg@5']:.4f}  recall@10={metrics['recall@10']:.4f}  "
            f"mrr={metrics['mrr']:.4f}  rerank latency p50={p50:.1f}ms p95={p95:.1f}ms"
        )
        sweep[pool_size] = {
            "metrics": metrics,
            "rerank_latency_p50_ms": p50,
            "rerank_latency_p95_ms": p95,
            "latencies_ms": latencies_ms,
        }

    RESULTS_RAW.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_RAW.write_text(
        json.dumps(
            {
                "n_queries": len(queries),
                "dense_latency_p50_ms": float(np.percentile(dense_latencies_ms, 50)),
                "dense_latency_p95_ms": float(np.percentile(dense_latencies_ms, 95)),
                "baseline": baseline_metrics,
                "pool_sizes": {str(k): {kk: vv for kk, vv in v.items() if kk != "latencies_ms"} for k, v in sweep.items()},
            },
            indent=2,
        )
    )
    print(f"Wrote {RESULTS_RAW}")

    # --- Plot: NDCG@5 vs. p95 reranking latency, one point per pool size ---
    fig, ax1 = plt.subplots(figsize=(8, 6))
    x = [sweep[p]["rerank_latency_p95_ms"] for p in pool_sizes]
    y = [sweep[p]["metrics"]["ndcg@5"] for p in pool_sizes]
    ax1.plot(x, y, marker="o", color="tab:blue")
    for p, xi, yi in zip(pool_sizes, x, y):
        ax1.annotate(f"pool={p}", (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax1.axhline(baseline_metrics["ndcg@5"], color="gray", linestyle="--", label="no-rerank baseline")
    ax1.set_xlabel("Reranking latency (ms/query, p95)")
    ax1.set_ylabel("NDCG@5")
    ax1.set_title(f"Cross-encoder reranking: NDCG@5 vs. p95 latency ({len(queries)} golden queries)")
    ax1.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_PNG, dpi=150)
    print(f"Wrote {RESULTS_PNG}")


if __name__ == "__main__":
    main()
