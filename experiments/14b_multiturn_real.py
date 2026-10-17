"""Phase 14 Unit 4 -- do follow-ups resolve end to end through the real pipeline?

Phase 8 measured query rewriting at the retriever with history = one short user
turn. The server's history will contain the assistant's real (long) answers, a
regime Phase 8 never measured. For N items from the 24-item synthetic multi-turn
set (data/golden/query_rewrite_eval_v1.jsonl, anchored to real golden ground
truth) this runs the REAL pipeline (real router, retrieval, cross-encoder,
Groq/Gemini generation, real QueryRewriter) four ways per item:

  turn1   turn1 asked alone -> the system's real answer (becomes the assistant turn)
  A       turn2, no history                      (context-free baseline)
  B       turn2, history = [user turn1]          (Phase 8's regime)
  C       turn2, history = [user turn1, assistant <real answer to turn1>]  (server regime)

Metric: did the retrieved top-10 contain a ground-truth chunk (hit@10) and what
fraction of the ground-truth chunks (recall@10). Tiny n: an end-to-end sanity
check of the wiring, not a replacement for Phase 8's 24-item retrieval ablation.
Checkpointed; per-item failures recorded, not fatal. Throwaway caches.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time

from aegis.config import REPO_ROOT, settings
from aegis.factory import build_pipeline
from aegis.types import Query

N = int(os.environ.get("AEGIS_MT_N", "6"))
RAW = REPO_ROOT / "docs" / "results" / "multiturn-real-raw.json"
CACHE = REPO_ROOT / "data" / "cache" / "phase14d"


def metrics(result, truth: set[str]) -> dict:
    ids = [r.chunk.id for r in result.retrieved]
    found = truth & set(ids)
    return {"hit": bool(found), "recall": len(found) / len(truth), "n_retrieved": len(ids),
            "escalate": result.escalate, "rewritten_query": result.rewritten_query,
            "latency_ms": result.stage_latencies_ms, "text": result.text[:300]}


def main() -> int:
    items = [json.loads(l) for l in open(settings.paths.data_golden / "query_rewrite_eval_v1.jsonl", encoding="utf-8")]
    items = random.Random(settings.golden.random_seed).sample(items, N)
    rows = {r["id"]: r for r in json.loads(RAW.read_text())} if RAW.exists() else {}
    shutil.rmtree(CACHE, ignore_errors=True)
    pipeline = build_pipeline(cache_dir=CACHE, corpus_version="phase14d")
    pipeline.warm_up()
    for i, it in enumerate(items, 1):
        if it["id"] in rows and not rows[it["id"]].get("error"):
            continue
        truth = set(it["ground_truth_chunk_ids"])
        try:
            first = pipeline.answer(Query(id=it["id"] + "::t1", text=it["turn1"]))
            time.sleep(2)
            assistant = first.text
            row = {"id": it["id"], "stratum": it["stratum"], "turn1": it["turn1"], "turn2": it["turn2"],
                   "assistant_chars": len(assistant), "assistant_escalated": first.escalate}
            for label, hist in (("A", None), ("B", [("user", it["turn1"])]),
                                ("C", [("user", it["turn1"]), ("assistant", assistant[: settings.api.history_message_chars])])):
                row[label] = metrics(pipeline.answer(Query(id=f"{it['id']}::{label}", text=it["turn2"]), history=hist), truth)
                time.sleep(2)
            row["error"] = None
        except Exception as exc:  # recorded, not fatal
            row = {"id": it["id"], "error": f"{type(exc).__name__}: {exc}"[:300]}
        rows[it["id"]] = row
        RAW.write_text(json.dumps(list(rows.values()), indent=2))
        print(f"[{i}/{len(items)}] {it['id']} " + (row["error"] or " ".join(f"{k}:hit={row[k]['hit']},rec={row[k]['recall']:.2f}" for k in "ABC")))
    shutil.rmtree(CACHE, ignore_errors=True)
    ok = [r for r in rows.values() if not r.get("error")]
    for k in "ABC":
        print(k, f"hit@10 {sum(r[k]['hit'] for r in ok)}/{len(ok)}  mean recall@10 {sum(r[k]['recall'] for r in ok)/len(ok):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
