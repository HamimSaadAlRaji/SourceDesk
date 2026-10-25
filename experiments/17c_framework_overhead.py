"""Phase 17 (design record 17e): what does LangGraph itself cost per request?

Every stage is an instant in-memory fake and the tracer records names only, so the whole
request is orchestration: control flow, state copies, node dispatch. Real requests spend
seconds in the reranker and the LLM; this isolates the milliseconds the framework adds on top.

Interleaved in blocks (direct, graph, graph, direct, ...) so drift cannot favour one path, with
a GC before each block. Latency is wall time around `pipeline.answer`, p50/p95 over all requests.
"""

import gc
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "tests")

import test_graph_pipeline as g  # noqa: E402
import test_pipeline as t  # noqa: E402

from aegis.config import REPO_ROOT  # noqa: E402
from aegis.graph.pipeline import LangGraphPipeline  # noqa: E402
from aegis.pipeline import AnswerPipeline  # noqa: E402
from aegis.types import Query  # noqa: E402

BLOCK = 200
BLOCKS_PER_PATH = 10  # 2,000 requests per path per scenario
OUT = REPO_ROOT / "docs" / "results" / "framework-overhead-raw.json"
SCENARIOS = ["generated_answer", "policy_escalation", "low_confidence_escalation", "exact_cache_hit"]


def build(cls, scenario):
    kwargs = g._SCENARIOS[scenario]()
    kwargs.setdefault("exact_cache", t._FakeCache())
    kwargs.setdefault("semantic_cache", t._FakeCache())
    pipeline, tracer = t._pipeline(cls=cls, **kwargs)
    return pipeline, tracer


def time_block(pipeline, tracer, query) -> list[float]:
    out = []
    for _ in range(BLOCK):
        tracer.spans.clear()
        start = time.perf_counter()
        pipeline.answer(query)
        out.append((time.perf_counter() - start) * 1e6)  # microseconds
    return out


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main() -> None:
    query = Query(id="o", text="how do I use dependency injection")
    report: dict = {"block": BLOCK, "requests_per_path": BLOCK * BLOCKS_PER_PATH, "scenarios": {}}
    for scenario in SCENARIOS:
        pipes = {"direct": build(AnswerPipeline, scenario), "langgraph": build(LangGraphPipeline, scenario)}
        for p, tr in pipes.values():  # warm both (first-call imports, lazy compiles)
            time_block(p, tr, query)
        samples: dict[str, list[float]] = {"direct": [], "langgraph": []}
        for b in range(BLOCKS_PER_PATH):
            order = ("direct", "langgraph") if b % 2 == 0 else ("langgraph", "direct")
            for name in order:
                gc.collect()
                samples[name] += time_block(*pipes[name], query)
        row = {
            name: {"mean_us": statistics.fmean(v), "p50_us": pct(v, 0.5), "p95_us": pct(v, 0.95), "p99_us": pct(v, 0.99)}
            for name, v in samples.items()
        }
        row["delta_mean_us"] = row["langgraph"]["mean_us"] - row["direct"]["mean_us"]
        row["delta_p50_us"] = row["langgraph"]["p50_us"] - row["direct"]["p50_us"]
        row["delta_p95_us"] = row["langgraph"]["p95_us"] - row["direct"]["p95_us"]
        report["scenarios"][scenario] = row
        print(f"{scenario:28s} direct p50 {row['direct']['p50_us']:8.0f} us  graph p50 {row['langgraph']['p50_us']:8.0f} us"
              f"  delta p50 {row['delta_p50_us']:+8.0f} us  p95 {row['delta_p95_us']:+8.0f} us  mean {row['delta_mean_us']:+8.0f} us")
    Path(OUT).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
