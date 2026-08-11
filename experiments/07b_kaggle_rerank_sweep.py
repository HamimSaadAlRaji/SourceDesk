"""Run this on Kaggle (GPU notebook), not locally.

Companion to `experiments/07a_export_rerank_candidates.py`, which produced
`rerank_candidates_export.json` -- every Phase 7 golden query's top-100
dense candidate pool (real chunk text + bi-encoder score), so this script
needs no local corpus, no Qdrant, and no `aegis` package import. It's
intentionally self-contained: only pip-installable dependencies.

--- How to run ---
1. https://www.kaggle.com/code -> New Notebook.
2. Settings (right sidebar) -> Accelerator -> GPU T4 x2 (or any GPU).
3. Add Input -> Upload -> `rerank_candidates_export.json`
   (from `data/processed/` in this repo). Note the mounted path Kaggle
   shows you (usually `/kaggle/input/<dataset-slug>/rerank_candidates_export.json`)
   and update CANDIDATES_PATH below to match.
4. Paste this whole file into one cell (or upload it and `!python thisfile.py`).
5. Run. Takes a few minutes on a T4 -- GPU cross-encoder throughput is much
   higher than this project's M1 Pro hit in practice (see design record
   07's §6 for why the local run was abandoned).
6. Download `rerank_sweep_results.json` from the notebook's output, or just
   copy the printed table back -- either is enough to write up the results.
"""

from __future__ import annotations

import json
import time

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CANDIDATES_PATH = "/kaggle/input/rerank-candidates/rerank_candidates_export.json"
OUTPUT_PATH = "/kaggle/working/rerank_sweep_results.json"

# Real corpus chunks run up to ~400 tokens. Since BATCH_SIZE must exceed a
# pool size for sentence-transformers to send that whole pool through in
# ONE un-chunked forward pass, a large BATCH_SIZE (64) meant pool=50 was
# scored as a single batch of 50 long-sequence pairs -- attention memory
# scales with batch_size x sequence_length^2, and that, stacked on
# fragmented memory from ~300 prior forward passes (pool=10, pool=20) with
# no cache-clearing between them, exhausted the T4's ~14.56GB (measured:
# CUDA OOM requesting 3.54 GiB with only 2.95 GiB free, 11.61 GiB already
# resident). 16 forces internal chunking at every pool size in this sweep;
# combined with the cache-clearing below, this is the fix.
BATCH_SIZE = 16


def resolve_candidates_path() -> str:
    """Kaggle mounts uploaded datasets under a slug derived from the
    dataset's title (e.g. a dataset titled "Rerank" mounts at
    `/kaggle/input/rerank/...`, not the name of the file inside it) --
    not predictable from here, so search for the file instead of trusting
    the hardcoded guess above."""
    import os

    if os.path.exists(CANDIDATES_PATH):
        return CANDIDATES_PATH
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "rerank_candidates_export.json" in files:
            found = os.path.join(root, "rerank_candidates_export.json")
            print(f"CANDIDATES_PATH not found; using discovered path instead: {found}")
            return found
    raise FileNotFoundError(
        "rerank_candidates_export.json not found under /kaggle/input -- confirm the dataset is attached "
        "(right sidebar -> Input) and re-run."
    )


def install_missing() -> None:
    import importlib
    import subprocess
    import sys

    for pkg, import_name in [("sentence-transformers", "sentence_transformers"), ("ranx", "ranx")]:
        try:
            importlib.import_module(import_name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def main() -> None:
    install_missing()

    import numpy as np
    import torch
    from ranx import Qrels, Run
    from ranx import evaluate as ranx_evaluate
    from sentence_transformers import CrossEncoder

    with open(resolve_candidates_path(), encoding="utf-8") as f:
        payload = json.load(f)

    queries = payload["queries"]
    pool_sizes = sorted(payload["pool_sizes"])
    final_top_k = payload["final_top_k"]
    model_name = payload["reranker_model"]
    print(f"{len(queries)} queries, pool sizes {pool_sizes}, final_top_k={final_top_k}, model={model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    model = CrossEncoder(model_name, device=device)

    def metrics_for(run_dict: dict[str, list[tuple[str, float]]]) -> dict[str, float]:
        qrels = Qrels({q["id"]: dict.fromkeys(q["ground_truth_chunk_ids"], 1) for q in queries})
        run = Run({qid: dict(hits) for qid, hits in run_dict.items()})
        results = ranx_evaluate(qrels, run, ["ndcg@5", "recall@10", "mrr"])
        return {k: float(v) for k, v in results.items()}

    # No-rerank baseline: bi-encoder order alone (candidates already sorted
    # by dense score at export time), truncated to final_top_k.
    baseline_run = {q["id"]: [(c["chunk_id"], c["bi_encoder_score"]) for c in q["candidates"][:final_top_k]] for q in queries}
    baseline_metrics = metrics_for(baseline_run)
    print(f"No-rerank (bi-encoder only) baseline: {baseline_metrics}")

    results: dict = {"baseline": baseline_metrics, "pool_sizes": {}}
    for pool_size in pool_sizes:
        run_dict: dict[str, list[tuple[str, float]]] = {}
        latencies_ms: list[float] = []
        for q in queries:
            candidates = q["candidates"][:pool_size]
            pairs = [(q["text"], c["text"]) for c in candidates]

            start = time.perf_counter()
            scores = model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)
            latencies_ms.append((time.perf_counter() - start) * 1000)

            reranked = sorted(zip((c["chunk_id"] for c in candidates), scores), key=lambda x: x[1], reverse=True)
            run_dict[q["id"]] = reranked[:final_top_k]

        metrics = metrics_for(run_dict)
        p50 = float(np.percentile(latencies_ms, 50))
        p95 = float(np.percentile(latencies_ms, 95))
        print(
            f"pool={pool_size:>4}  ndcg@5={metrics['ndcg@5']:.4f}  recall@10={metrics['recall@10']:.4f}  "
            f"mrr={metrics['mrr']:.4f}  rerank latency p50={p50:.1f}ms p95={p95:.1f}ms"
        )
        results["pool_sizes"][pool_size] = {"metrics": metrics, "rerank_latency_p50_ms": p50, "rerank_latency_p95_ms": p95}

        # Write after every pool size, not just at the end -- a crash on a
        # later pool size (this sweep OOM'd partway through pool=50 once
        # already) should never lose already-completed measurements again.
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nWrote {OUTPUT_PATH} -- download it from the notebook's Output pane.")
    print("\nFull results (copy this if download is inconvenient):")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
