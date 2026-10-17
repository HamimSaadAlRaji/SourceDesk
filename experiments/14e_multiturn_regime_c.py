"""Phase 14 exit criterion "multi-turn follow-ups resolve correctly" (design record 14i).

Phase 8 measured rewriting with history = one short user turn (regime B) at the
retriever: +52.7% NDCG@5 over the bare follow-up, on the 24-item synthetic set.
The server's history also carries the assistant's real answer (regime C), which
Phase 8 never measured; 14b's n=6 end-to-end check was too small, and not
comparable to Phase 8 (the pipeline doesn't exclude the source issue's own
chunks, which are never ground truth).

Per item, all 24 of data/golden/query_rewrite_eval_v1.jsonl:
  turn1  asked alone through the REAL pipeline -> the real answer (or escalation
         message), which becomes the assistant turn, cut to what the server
         shows the rewriter (ApiSettings.history_message_chars)
  A      turn2 alone
  B      turn2 + [user turn1]                     (Phase 8's regime)
  C      turn2 + [user turn1, assistant answer]   (the server's regime)
For A/B/C: the real QueryRewriter, then dense retrieval scored two ways:
  - "eval":   exclude_doc_id=source_issue, ranx NDCG@5 / Recall@10 / MRR --
              Phase 8's exact protocol, so B here should reproduce Phase 8
  - "served": no exclusion (what a user gets), plus the real router + confidence
              scorer -> would the system have escalated?
No generation for A/B/C: generation doesn't change what was retrieved or
whether the pipeline escalates, and it is the expensive call.

Pre-registered pass criterion (written before any number was seen):
  the exit criterion is MET if, on the "eval" protocol,
    (1) C - A NDCG@5 has a paired-bootstrap 95% CI entirely above 0, and
    (2) C - B NDCG@5 has a 95% CI whose lower bound is above -0.05
        (the real assistant answer in history doesn't measurably hurt).
  Otherwise NOT MET, and reported as such.

Checkpointed per item; reruns only fill gaps. Paced for qwen's 7,000 input
tokens/minute cap. Throwaway caches. Raw: docs/results/multiturn-regime-c-raw.json

    AEGIS_MT_N=1 uv run python experiments/14e_multiturn_regime_c.py   # smoke
    uv run python experiments/14e_multiturn_regime_c.py                 # all 24
"""

from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np

from aegis.config import REPO_ROOT, settings
from aegis.eval.metrics import build_qrels, build_run, compute_metrics
from aegis.factory import build_pipeline
from aegis.guardrails.confidence import decide_escalation
from aegis.router.policy import decide as decide_route
from aegis.types import Query

N = int(os.environ.get("AEGIS_MT_N", "24"))
RAW = REPO_ROOT / "docs" / "results" / "multiturn-regime-c-raw.json"
CACHE = REPO_ROOT / "data" / "cache" / "phase14e"
METRICS = ["ndcg@5", "recall@10", "mrr"]
TOP_K = 10
# One generation sends ~4.5k input tokens against a 7,000/minute cap; the
# rewrites/routers are small. 13c showed what an exceeded per-minute cap does
# to latency (silent SDK retries), so pace instead.
PAUSE_AFTER_TURN1_S = 40.0
PAUSE_BETWEEN_CALLS_S = 2.0
N_BOOT = 10000
SEED = settings.golden.random_seed


def _legs(item: dict, answer: str) -> dict[str, list[tuple[str, str]] | None]:
    cut = settings.api.history_message_chars
    return {
        "A": None,
        "B": [("user", item["turn1"])],
        "C": [("user", item["turn1"]), ("assistant", answer[:cut])],
    }


def _run_item(pipeline, item: dict) -> dict:
    first = pipeline.answer(Query(id=item["id"] + "::t1", text=item["turn1"]))
    row = {"id": item["id"], "stratum": item["stratum"], "turn1_escalated": first.escalate,
           "turn1_answer": first.text, "turn1_provider": first.provider}
    time.sleep(PAUSE_AFTER_TURN1_S)
    for leg, history in _legs(item, first.text).items():
        text = item["turn2"] if history is None else pipeline.rewriter.rewrite(history, item["turn2"])
        eval_hits = pipeline.retriever.search(text, top_k=TOP_K, exclude_doc_id=item["source_issue"])
        served = pipeline.retriever.search(text, top_k=TOP_K)
        route = pipeline.router.classify(text)
        route_escalates = decide_route(route).escalate
        confidence = None if route_escalates else pipeline.confidence_scorer.score(text, served).confidence
        row[leg] = {
            "query": text,
            "eval_run": [(c.id, float(s)) for c, s in eval_hits],
            "served_ids": [c.id for c, _ in served],
            "served_self_matches": sum(c.doc_id == item["source_issue"] for c, _ in served),
            "route": route.value,
            "confidence": confidence,
            "escalate": route_escalates or decide_escalation(confidence, pipeline.tau).escalate,
        }
        time.sleep(PAUSE_BETWEEN_CALLS_S)
    return row


def _per_item(rows: list[dict], items: dict, leg: str, metric: str) -> np.ndarray:
    out = []
    for r in rows:
        qrels = build_qrels({r["id"]: items[r["id"]]["ground_truth_chunk_ids"]})
        run = build_run({r["id"]: [tuple(x) for x in r[leg]["eval_run"]]})
        out.append(compute_metrics(qrels, run, metrics=[metric])[metric])
    return np.array(out)


def _paired_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    d = a - b
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    items = [json.loads(l) for l in open(settings.paths.data_golden / "query_rewrite_eval_v1.jsonl", encoding="utf-8")][:N]
    by_id = {it["id"]: it for it in items}
    done = {r["id"]: r for r in json.loads(RAW.read_text())} if RAW.exists() else {}
    shutil.rmtree(CACHE, ignore_errors=True)
    pipeline = build_pipeline(cache_dir=CACHE, corpus_version="phase14e")
    assert pipeline.rewriter is not None, "rewriting must be enabled for this eval"
    pipeline.warm_up()
    for i, it in enumerate(items, 1):
        if it["id"] in done and not done[it["id"]].get("error"):
            continue
        try:
            done[it["id"]] = _run_item(pipeline, it)
        except Exception as exc:  # noqa: BLE001 -- recorded, retried on the next run
            done[it["id"]] = {"id": it["id"], "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        RAW.write_text(json.dumps(list(done.values()), indent=2, ensure_ascii=False))
        print(f"[{i}/{len(items)}] {it['id']} {'ERROR ' + done[it['id']]['error'] if done[it['id']].get('error') else 'ok'}", flush=True)

    rows = [r for r in done.values() if r["id"] in by_id and not r.get("error")]
    print(f"\n{len(rows)}/{len(items)} items complete; turn1 escalated {sum(r['turn1_escalated'] for r in rows)}")
    if not rows:
        return
    scores = {leg: {m: _per_item(rows, by_id, leg, m) for m in METRICS} for leg in "ABC"}
    print("\nleg | NDCG@5 | Recall@10 | MRR    | escalated (served) | mean self-matches in served top-10")
    for leg in "ABC":
        s = scores[leg]
        print(f" {leg}  | {s['ndcg@5'].mean():.4f} | {s['recall@10'].mean():.4f}    | {s['mrr'].mean():.4f} | "
              f"{sum(r[leg]['escalate'] for r in rows)}/{len(rows)}              | "
              f"{np.mean([r[leg]['served_self_matches'] for r in rows]):.2f}")
    ca = _paired_ci(scores["C"]["ndcg@5"], scores["A"]["ndcg@5"])
    cb = _paired_ci(scores["C"]["ndcg@5"], scores["B"]["ndcg@5"])
    ba = _paired_ci(scores["B"]["ndcg@5"], scores["A"]["ndcg@5"])
    print(f"\nNDCG@5 paired diffs (mean, 95% CI):  B-A {ba[0]:+.4f} [{ba[1]:+.4f}, {ba[2]:+.4f}]  "
          f"C-A {ca[0]:+.4f} [{ca[1]:+.4f}, {ca[2]:+.4f}]  C-B {cb[0]:+.4f} [{cb[1]:+.4f}, {cb[2]:+.4f}]")
    if len(rows) == len(items) == 24:
        met = ca[1] > 0 and cb[1] > -0.05
        print(f"pre-registered criterion: {'MET' if met else 'NOT MET'} (C-A lower {ca[1]:+.4f} > 0; C-B lower {cb[1]:+.4f} > -0.05)")
    else:
        print("pre-registered criterion: not evaluated (needs all 24 items)")


if __name__ == "__main__":
    main()
