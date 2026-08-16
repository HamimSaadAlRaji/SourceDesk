"""Phase 8 query-understanding ablation (plan.md Phase 8, step 7: "A/B every
one. Each costs latency and tokens. At least one probably won't pay for
itself -- report that honestly.").

Base retriever is Phase 6's winning config unchanged: dense-only,
heading-aware chunking, `bge-large-en-v1.5`, no contextual enrichment (same
setup as `experiments/07_rerank_sweep.py`). Four techniques measured
against it, each A/B'd on the full 150-query dev set except query
rewriting (see below):

- Multi-query expansion (`ExpandedRetriever`): recall@10/NDCG@5 delta +
  latency cost.
- HyDE (`HydeRetriever`): same.
- Iterative retrieval (`IterativeRetriever`): low-confidence threshold
  calibrated from this run's own baseline top-1 score distribution (25th
  percentile -- provisional, see design record 08 §6), then measured the
  same way, plus how many queries actually triggered the retry.
- Query rewriting (`QueryRewriter`): can't use the 150-query dev set
  directly -- it needs (history, follow-up) pairs, and this corpus has no
  real multi-turn data. Uses the 24-item LLM-generated synthetic set from
  `experiments/08b_generate_rewrite_eval_set.py` instead, scored against
  each item's *original* golden query's ground truth. Three conditions per
  item: turn2 alone (context-free), turn2 rewritten with turn1 as history,
  and the original full query text (oracle upper bound, reused directly
  from this script's own baseline run since the 24 items are a subset of
  the same 150).

Per CLAUDE.md's compute-budget rule, real per-call latency was benchmarked
before running this at full scale (~0.5-1.2s/call for these non-JSON-mode
generations, vs. ~7.2s/call measured on the Phase 8 router's JSON-mode
calls) -- full-dev-set projected under 10 minutes total, so this runs
locally/sequentially rather than moving to Kaggle or subsampling.
"""

from __future__ import annotations

import json
import time

import numpy as np

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.eval.metrics import build_qrels, build_run, compute_metrics
from aegis.query.expand import ExpandedRetriever, QueryExpander
from aegis.query.hyde import HydeGenerator, HydeRetriever
from aegis.query.iterative import IterativeRetriever, QueryReformulator
from aegis.query.rewrite import QueryRewriter
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, GoldenQuery

METRICS = ["ndcg@5", "recall@10", "mrr"]
TOP_K = 10
ITERATIVE_THRESHOLD_PERCENTILE = 25
RESULTS_MD = REPO_ROOT / "docs" / "results" / "query-understanding-ablation.md"
RESULTS_RAW = REPO_ROOT / "docs" / "results" / "query-understanding-ablation-raw.json"
REWRITE_EVAL_PATH = REPO_ROOT / "data" / "golden" / "query_rewrite_eval_v1.jsonl"


def load_chunks() -> list[Chunk]:
    path = settings.paths.data_processed / "chunks.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def load_positive_queries() -> list[GoldenQuery]:
    path = settings.paths.data_golden / "golden_set_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        queries = [GoldenQuery(**json.loads(line)) for line in f]
    return [q for q in queries if not q.is_negative]


def measure(name: str, queries: list[GoldenQuery], retriever, qrels) -> dict:
    run_dict: dict[str, list[tuple[str, float]]] = {}
    latencies_ms: list[float] = []
    for q in queries:
        start = time.perf_counter()
        results = retriever.search(q.text, top_k=TOP_K, exclude_doc_id=q.source_issue)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        run_dict[q.id] = [(c.id, s) for c, s in results]

    metrics = compute_metrics(qrels, build_run(run_dict), metrics=METRICS)
    p50, p95 = float(np.percentile(latencies_ms, 50)), float(np.percentile(latencies_ms, 95))
    mean = float(np.mean(latencies_ms))
    print(f"{name:20s} ndcg@5={metrics['ndcg@5']:.4f} recall@10={metrics['recall@10']:.4f} mrr={metrics['mrr']:.4f} "
          f"latency mean={mean:.1f}ms p50={p50:.1f}ms p95={p95:.1f}ms")
    return {"metrics": metrics, "latency_mean_ms": mean, "latency_p50_ms": p50, "latency_p95_ms": p95, "run": run_dict}


def main() -> None:
    chunks = load_chunks()
    queries = load_positive_queries()
    print(f"{len(chunks)} chunks, {len(queries)} positive golden queries")

    cache = EmbeddingCache(settings.paths.embedding_cache)
    # CPU, not the configured MPS default -- this project has already hit
    # (and documented, Phase 3 + Phase 7 decision logs) MPS entering
    # macOS's "stuck" scheduler state under concurrent memory pressure
    # (Qdrant + PyTorch + this session's other load). Confirmed here too:
    # the first attempt at this script hung for 17+ minutes at 0% CPU with
    # ~3GB swap in use before being killed -- the identical signature.
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=cache, device="cpu")
    dense_index = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)

    qrels = build_qrels({q.id: q.ground_truth_chunk_ids for q in queries})

    results: dict[str, dict] = {}

    print("\n--- Baseline (no query understanding, Phase 6 winning config) ---")
    results["baseline"] = measure("baseline", queries, dense_index, qrels)
    baseline_top1 = [run[0][1] if run else 0.0 for run in results["baseline"]["run"].values()]
    threshold = float(np.percentile(baseline_top1, ITERATIVE_THRESHOLD_PERCENTILE))
    print(f"Calibrated low-confidence threshold (p{ITERATIVE_THRESHOLD_PERCENTILE} of baseline top-1 scores): {threshold:.4f}")

    print("\n--- Multi-query expansion ---")
    expander = QueryExpander()
    expanded = ExpandedRetriever(dense_index, expander, enabled=True)
    results["expansion"] = measure("expansion", queries, expanded, qrels)

    print("\n--- HyDE ---")
    hyde_gen = HydeGenerator()
    hyded = HydeRetriever(dense_index, hyde_gen, enabled=True)
    results["hyde"] = measure("hyde", queries, hyded, qrels)

    print("\n--- Iterative retrieval ---")
    reformulator = QueryReformulator()
    iterative = IterativeRetriever(dense_index, reformulator, score_threshold=threshold, max_retries=1, enabled=True)
    results["iterative"] = measure("iterative", queries, iterative, qrels)
    n_low_confidence = sum(1 for s in baseline_top1 if s < threshold)
    print(f"  {n_low_confidence}/{len(queries)} queries were below the low-confidence threshold and triggered a retry")
    results["iterative"]["threshold"] = threshold
    results["iterative"]["n_retried"] = n_low_confidence

    print("\n--- Query rewriting (24-item synthetic multi-turn set) ---")
    rewrite_rows = [json.loads(line) for line in REWRITE_EVAL_PATH.read_text(encoding="utf-8").splitlines()]
    rewriter = QueryRewriter()
    rewrite_qrels = build_qrels({r["id"]: r["ground_truth_chunk_ids"] for r in rewrite_rows})

    context_free_run, rewritten_run, oracle_run = {}, {}, {}
    rewrite_latencies_ms = []
    for r in rewrite_rows:
        cf_results = dense_index.search(r["turn2"], top_k=TOP_K, exclude_doc_id=r["source_issue"])
        context_free_run[r["id"]] = [(c.id, s) for c, s in cf_results]

        start = time.perf_counter()
        rewritten_query = rewriter.rewrite([("user", r["turn1"])], r["turn2"])
        rewrite_latencies_ms.append((time.perf_counter() - start) * 1000)
        rw_results = dense_index.search(rewritten_query, top_k=TOP_K, exclude_doc_id=r["source_issue"])
        rewritten_run[r["id"]] = [(c.id, s) for c, s in rw_results]

        oracle_run[r["id"]] = results["baseline"]["run"][r["original_query_id"]]

    cf_metrics = compute_metrics(rewrite_qrels, build_run(context_free_run), metrics=METRICS)
    rw_metrics = compute_metrics(rewrite_qrels, build_run(rewritten_run), metrics=METRICS)
    oracle_metrics = compute_metrics(rewrite_qrels, build_run(oracle_run), metrics=METRICS)
    rw_p50, rw_p95 = float(np.percentile(rewrite_latencies_ms, 50)), float(np.percentile(rewrite_latencies_ms, 95))
    print(f"  context-free (turn2 alone): {cf_metrics}")
    print(f"  rewritten (turn2 + history): {rw_metrics}")
    print(f"  oracle (original full query): {oracle_metrics}")
    print(f"  rewrite latency p50={rw_p50:.1f}ms p95={rw_p95:.1f}ms")
    results["rewrite"] = {
        "n_items": len(rewrite_rows),
        "context_free": cf_metrics,
        "rewritten": rw_metrics,
        "oracle": oracle_metrics,
        "latency_p50_ms": rw_p50,
        "latency_p95_ms": rw_p95,
    }

    for key in ("baseline", "expansion", "hyde", "iterative"):
        results[key].pop("run", None)

    RESULTS_RAW.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_RAW.write_text(json.dumps({"n_queries": len(queries), **results}, indent=2))
    print(f"\nWrote {RESULTS_RAW}")


if __name__ == "__main__":
    main()
