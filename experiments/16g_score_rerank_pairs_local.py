"""Phase 16, unit 16i: score the CI escalation check's reranker pairs on THIS machine.

Run once instead of the Kaggle handoff (16f), by the user's explicit waiver of the 10-minute
local compute rule for this single job (2026-10-25; CLAUDE.md section 10 otherwise sends it to
Kaggle). Projected from a real 12-pair benchmark: 1.32 s/pair x 538 pairs = ~11.8 min on the M1.

Scores only the pairs the gate needs (min_rank < rerank_pool_size, 538), not the top-5 margin
pairs in the export. Uses the served code path (CrossEncoderReranker's score function, CPU).
Resumable: results are checkpointed after every batch and already-scored keys are skipped.

    uv run python experiments/16g_score_rerank_pairs_local.py
Output: ci/caches/rerank_scores.json (committed; a PR diff shows when a score changes).
"""

from __future__ import annotations

import json
import time

import sentence_transformers
import torch

from aegis.config import REPO_ROOT, settings
from aegis.eval.rerank_cache import ScoreCache
from aegis.rerank.cross_encoder import CrossEncoderReranker

EXPORT = REPO_ROOT / "ci-fixtures" / "rerank_pairs_export.json"
OUT = REPO_ROOT / "ci" / "caches" / "rerank_scores.json"
BATCH = 32


def main() -> None:
    export = json.loads(EXPORT.read_text(encoding="utf-8"))
    model = export["model_name"]
    assert model == settings.rerank.model_name, "export was made for a different reranker model"
    needed = [p for p in export["pairs"] if p["min_rank"] < settings.confidence.rerank_pool_size]
    cache = ScoreCache.load(OUT)
    todo = [p for p in needed if p["key"] not in cache.scores]
    # Similar lengths per batch pad less (the same trick the corpus embedding run needed).
    todo.sort(key=lambda p: len(p["query"]) + len(p["chunk_text"]))
    print(f"{len(needed)} needed pairs, {len(needed) - len(todo)} already cached, scoring {len(todo)}", flush=True)

    score_fn = CrossEncoderReranker(device="cpu")._score_fn
    score_fn([("warm up", "warm up")])  # model load, excluded from the per-pair timing
    t0 = time.perf_counter()
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        for p, s in zip(batch, score_fn([(p["query"], p["chunk_text"]) for p in batch])):
            cache.scores[p["key"]] = float(s)
        cache.meta = {"model_name": model, "device": "cpu (Apple M1 Pro)", "torch": torch.__version__,
                      "sentence_transformers": sentence_transformers.__version__,
                      "n_pairs": str(len(cache.scores)), "scored_by": "experiments/16g_score_rerank_pairs_local.py"}
        cache.save(OUT)
        done = start + len(batch)
        el = time.perf_counter() - t0
        print(f"  {done}/{len(todo)} pairs, {el:.0f}s elapsed, {el / done:.2f}s/pair, "
              f"~{(len(todo) - done) * el / done / 60:.1f} min left", flush=True)
    print(f"done: {len(cache.scores)} scores in {time.perf_counter() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
