# 13c — Per-Stage Latency Attribution

**Phase:** 13 · **Date:** 2026-09-30 · **Files:** `src/aegis/pipeline.py`, `src/aegis/generation/pipeline.py`, `src/aegis/observability/latency.py`, `tests/test_latency_summary.py`, `experiments/13c_latency_attribution.py`, `experiments/13d_latency_report.py`, `docs/results/latency-budget.md`
**Topics from the course:** Latency vs. response quality (Module 5, #35); Logging, monitoring & observability (Module 5, #31)

## 1. What I built

`PipelineResult.stage_latencies_ms` (wall-clock ms per stage the request actually ran), a pure percentile aggregator (`stage_latency_summary`), a batch script that pushes real dev queries through the real `AnswerPipeline`, and a report generator that writes `docs/results/latency-budget.md`. Turns "it's slow" into "confidence scoring is 2,520 ms of a 4,012 ms p50."

## 2. Why this approach

Timings are measured inside the pipeline, not read back from Langfuse. Design record 13/13b already showed trace ingestion lags 8–14 s and the readback API changed under us; a latency number that depends on that is a number I can't trust or gate CI on (Phase 16). The same timings are attached to the trace, so both views exist, but the measurement doesn't depend on the trace.

`GenerationPipeline.answer()` did LLM generation and citation validation in one call, so the `generation` and `validation` spans couldn't have honest, separate durations. Split it into `generate()` + `validate()` (`answer()` is their composition; existing tests untouched).

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ------------------ |
| Read span durations back from the Langfuse API | Single source of truth; no new code in the pipeline | Ingestion lag of 8–14 s per trace, v2-only API on this org, and a network dependency inside what will become a CI gate. |
| Time only the whole request | Trivial | Can't answer the question this unit exists for: which stage. |
| Back-to-back batch of N queries (what I ran first) | Fastest to run: 25 queries in ~10 min | Measured rate-limit stalls, not the pipeline (see §5–6). Kept as a labeled second dataset rather than discarded. |

## 4. How it works

```python
@contextmanager
def _timed(timings, stage):
    start = time.perf_counter()
    try: yield
    finally: timings[stage] = (time.perf_counter() - start) * 1000.0
```

One `timings` dict per request, threaded through `_answer` (no instance state, so Phase 14's concurrent requests can't corrupt each other). A stage the request never ran is simply absent: a cache hit has only `cache_lookup`; an escalation has no `generation`. Percentiles are numpy's default linear interpolation, checked against hand-computed values for 1..100 (p50 50.5, p95 95.05, p99 99.01). `n` is reported beside every percentile.

## 5. What the numbers said

Baseline = 20 dev queries (15 positive + 5 negative, seed 20260802), **70 s idle between queries**, caches emptied, first query treated as cold start. Full tables: `docs/results/latency-budget.md`. Warm, generate path (n=12), ms:

| Stage | p50 | p95 |
|---|---|---|
| cache_lookup | 5 | 10 |
| routing | 251 | 334 |
| retrieval | 44 | 66 |
| **confidence** | **2,520** | **3,233** |
| generation | 920 | 1,443 |
| validation | 3 | 656 |
| **sum** | **4,012** | **4,688** |

Budget check (p95 < 3,000 ms): pre-generation path 3,486 (**over**), full generate path 4,688 (**over**), escalation path 3,128 (**over**). Confidence scoring is 67% of the generate-path p50; generation is 25%. Cold start: the first query's confidence stage took 17,435 ms (19.2 s wall) versus ~2.5 s warm.

**Same code, unpaced (25 queries back to back):** generate-path p50 38,080 ms, p95 55,338 ms; routing p50 5,353; generation p50 26,401.

`uv run pytest -q`: 471/471 passed (+4 aggregation tests, +3 pipeline latency tests, TDD'd RED-first).

## 6. What surprised me / what to watch for

**The first run measured the wrong thing.** Routing is a ~10-token classification and took up to 14.4 s. Compute doesn't vary like that. Rerunning with 70 s pauses dropped routing to 251 ms and the generate path from 38 s to 4 s. The likely cause is Groq's free-tier per-minute token cap forcing retry-after waits inside LLM calls. That is supported by the effect only: 429s and retries were not logged, so I did not confirm the mechanism at the HTTP level.

**The dominant cost is not the LLM.** Confidence scoring — local cross-encoder plus encoder work — is 2.5 s of a 4.0 s generate path. Generation is under a second. That reorders the optimization problem for Unit 4.

**Possible contradiction with Phase 12's cost model.** Design record 12f used 12.8 s for a cache miss, built from four earlier generation samples. If those were also taken back to back, they may contain the same stall. Generation here measures 0.9 s p50. I have not re-derived the cost model; this is flagged, not resolved.

**Unexplained:** `confidence` is local CPU with no LLM call, yet was ~45% slower in the burst run (p50 3,656 vs 2,520 ms). Rate limiting can't explain that. Machine load is a candidate but was not measured.

**Sample is small.** n=12 generate-path queries: p95 and p99 are near-max of a handful of points. They bound the tail; they don't measure it. The escalation path (n=7) is thinner still. `validation` p95 (656 ms) versus p50 (3 ms) is bimodal: it only costs anything when tier-2 entailment fires an LLM call.

A user's real latency under traffic above the free-tier cap is the burst number, not the paced one. Both are reported and labeled.

## 7. Where this fits and what's next

Unit 4 optimizes against p95 < 3 s using this as the "before." The attribution already narrows the search: confidence scoring (67%) and the 17 s cold start (warm models at startup) dominate, whereas parallelizing dense/sparse retrieval — a plan.md build-list item — targets a 44 ms stage and, since shipped retrieval is dense-only, has nothing to parallelize. I'll raise that scope question before building rather than after.
