"""Phase 17 (design record 17e): end-to-end latency of the real stack, direct vs LangGraph.

Two jobs in one run:
  1. The Phase 16 hand-off: a FRESH direct-path p95 on the current stack (gpt-5-mini generator,
     gpt-4.1-mini router/entailment). The only earlier number (5.6 s) is from the qwen era.
  2. The graph's delta on that same stack.

Design: 40 dev queries (seed 17), each run through BOTH paths back to back, alternating which goes
first, so provider latency drift and GPU wake effects hit both equally. Models, LLM clients and the
retriever are SHARED between the two pipelines (they differ only in orchestration); each path has its
own empty exact cache, the semantic cache is off, so every request is a cold miss. Sequential: a
latency measurement must not contend with itself. Paid (OpenAI ledger); set OPENAI_BUDGET_USD.
"""

import dataclasses
import json
import random
import sys
import tempfile
import time
from pathlib import Path

from aegis.cache.exact_match import ExactMatchCache
from aegis.config import REPO_ROOT, settings
from aegis.eval.escalation_gate import load_dev_queries
from aegis.factory import build_pipeline
from aegis.generation.llm.openai_provider import default_ledger
from aegis.graph.pipeline import LangGraphPipeline
from aegis.observability.latency import stage_latency_summary
from aegis.types import Query

N = 40
SEED = 17
OUT = REPO_ROOT / "docs" / "results" / "orchestration-latency-raw.json"


def main() -> None:
    settings.cache.semantic_cache_enabled = False  # cold path: every request pays every stage
    tmp = Path(tempfile.mkdtemp(prefix="lat17-"))
    direct = build_pipeline(cache_dir=tmp / "direct", backend="direct")
    graph = LangGraphPipeline(
        exact_cache=ExactMatchCache(tmp / "graph.sqlite"), semantic_cache=None, router=direct.router,
        retriever=direct.retriever, confidence_scorer=direct.confidence_scorer, generation=direct.generation,
        tracer=direct.tracer, corpus_version=direct.corpus_version, partition=direct.partition,
    )
    pipes = {"direct": direct, "langgraph": graph}

    queries = random.Random(SEED).sample(load_dev_queries(), N)
    print("warm_up:", direct.warm_up())
    warm = Query(id="warm", text="how do I declare a query parameter")
    for p in pipes.values():  # one unmeasured request each: connection setup, lazy imports
        p.answer(warm)

    spent_before = default_ledger().summary()["spent_usd"]
    rows = []
    for i, q in enumerate(queries):
        for name in (("direct", "langgraph") if i % 2 == 0 else ("langgraph", "direct")):
            start = time.perf_counter()
            r = pipes[name].answer(Query(id=q.id, text=q.text))
            wall = (time.perf_counter() - start) * 1000
            path = "generate" if not r.escalate else ("escalate_policy" if r.confidence is None else "escalate_confidence")
            rows.append({"query_id": q.id, "is_negative": q.is_negative, "backend": name, "order": i, "wall_ms": wall,
                         "path": path, "provider": r.provider, "stages": r.stage_latencies_ms})
        print(f"{i + 1}/{N}", end=" ", flush=True)
    spent = default_ledger().summary()["spent_usd"] - spent_before
    # Raw rows first: a bug in the summary below once cost a full paid run (design record 17e).
    OUT.write_text(json.dumps({"n": N, "seed": SEED, "spent_usd": round(spent, 6), "rows": rows}, indent=2))

    summary = {}
    for name in pipes:
        mine = [r for r in rows if r["backend"] == name]
        summary[name] = {
            "all": {k: dataclasses.asdict(v) for k, v in stage_latency_summary([{"wall": r["wall_ms"]} for r in mine]).items()},
            "by_path": {
                path: {k: dataclasses.asdict(v) for k, v in stage_latency_summary(
                    [{"wall": r["wall_ms"]} for r in mine if r["path"] == path]).items()}
                for path in sorted({r["path"] for r in mine})
            },
            "stages": {k: dataclasses.asdict(v) for k, v in stage_latency_summary([r["stages"] for r in mine]).items()},
        }
    OUT.write_text(json.dumps({"n": N, "seed": SEED, "spent_usd": round(spent, 6), "rows": rows, "summary": summary}, indent=2))
    print("\nspent", round(spent, 4))
    for name, s in summary.items():
        print(name, "all:", {k: round(v, 0) for k, v in s["all"]["wall"].items() if k != "n"}, "n", s["all"]["wall"]["n"])


if __name__ == "__main__":
    sys.exit(main())
