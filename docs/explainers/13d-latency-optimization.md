# 13d — Optimizing Against the Latency Budget

**Phase:** 13 · **Date:** 2026-09-30 · **Files:** `src/aegis/pipeline.py`, `tests/test_pipeline.py`, `experiments/13c_latency_attribution.py`, `13e_confidence_profile.py`, `13f_confidence_conditions.py`, `13g_tracer_overhead.py`, `docs/results/latency-budget.md`
**Topics from the course:** Latency vs. response quality (Module 5, #35)

## 1. What I built

`AnswerPipeline.warm_up()`: runs real retrieval and real confidence scoring once at startup so the first user request doesn't pay lazy model loads. Plus three diagnostic experiments (13e/13f/13g) that try to explain why the `confidence` stage costs more inside the pipeline than alone. **The p95 < 3 s budget is not met**, and one cost is still unexplained.

## 2. Why this approach

Unit 3 said confidence scoring was 67% of the generate-path p50 and the first request's confidence stage took 17.4 s. plan.md's build list also says "parallelize dense and sparse retrieval" — not done: shipped retrieval is dense-only (Phase 6) and retrieval is 37–44 ms, so there is nothing to parallelize and nothing to gain. "Batch rerank inference" is already how it works (one call over the pool). Changing `rerank_pool_size` would change confidence *values* and invalidate Phase 10's calibrated tau, so I did not touch it. That left warm-up as the one change that is both safe and attributable.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ------------------ |
| Shrink the rerank pool (3 -> 2) | Fewer cross-encoder pairs, less work | Changes confidence scores; tau and normalization bounds were calibrated on pool=3 (Phase 10). It would trade a latency win for an unmeasured accuracy change. |
| Parallelize dense + sparse retrieval | Named in plan.md | Shipped retrieval is dense-only; the stage is 44 ms. Nothing to overlap. |
| Speculative warm-up / keep-alive pings against GPU idle slowdown | Might remove the idle penalty | Not tested (see §6), and burns GPU continuously for a hypothesis I have not confirmed. |
| Run the router in parallel with retrieval | Saves <= ~44 ms | Retrieval would run for queries the router escalates, breaking the "billing/legal never reaches retrieval" audit property Phase 8 built. |

## 4. How it works

```python
def warm_up(self) -> dict[str, float]:
    with _timed(timings, "retrieval"):
        candidates = self.retriever.search(_WARM_UP_QUERY, top_k=self.retrieval_top_k)
    with _timed(timings, "confidence"):
        self.confidence_scorer.score(_WARM_UP_QUERY, candidates)
```

Real chunks, real cross-encoder. TDD'd with spies: it never touches the router, the LLM, either cache, or the tracer (`test_warm_up_exercises_retrieval_and_confidence_but_never_router_llm_or_caches`). `uv run pytest -q`: 472/472.

## 5. What the numbers said

**Warm-up works.** First request: 19,208 ms wall -> 2,957 ms (confidence 17,435 -> 1,663 ms). n=1 on each side. The cost moved to startup: 30.3 s (encoder 14.2 s + reranker 16.0 s).

**Steady state did not improve.** Re-run vs. Unit 3, generate path p50/p95 sum of stages: 4,012/4,688 -> 4,421/5,649 ms. Confidence p50 2,520 -> 2,401. Pre-generation path p95 4,030 ms, generate path p95 5,649 ms — both over 3,000. Escalation path p95 2,945 (n=7, not robust).

**The confidence stage, in isolation vs. in the pipeline** (real scorer, zero LLM calls, different queries per experiment, small n): alone back-to-back 560 ms p50 (n=15); alone after 70 s idle 1,031 (n=5); in pipeline with no-op tracer 502 (n=12); with real Langfuse tracer 926, p95 6,070 with one 11.3 s outlier (n=12); full paced pipeline 2,520, then 2,401 on the re-run.

## 6. What surprised me / what to watch for

**I proposed an explanation, tested it, and it was wrong.** The two paced runs disagreed with my isolated profile (2.4 s vs 0.56 s), and an unrelated Docker stack had been running during the baseline, so I suspected background load. The stack was gone for the re-run and the stage was still ~2.4 s. Load was never measured, so I can only say removing that stack did not help.

**Retracted (see 13e):** this section originally said idle gaps and the real tracer each roughly doubled the confidence stage. Follow-up experiments with the *same queries in every arm* (13h, 13i) did not reproduce either effect: the real tracer with 70 s idle gaps measured 656 ms p50, not ~2.4 s. Those earlier readings came from experiments that used different queries per arm and are not reliable. The gap is localized to the cross-encoder call but still unexplained; the remaining untested difference is real generation + validation running in the same process.

**Steady-state cost was unaffected by the one thing I changed.** Warm-up fixes the first request only. Anyone reading "warm-up added" should not conclude steady-state latency improved.


## 7. Where this fits and what's next

Phase 13's p95 < 3 s exit criterion is unmet, and the biggest lever (a ~2 s confidence-stage gap) is unexplained. Still not built: structured logging with the trace ID threaded through (build item 2) and the audit-trail demo (item 5). Phase 16's CI gate will need either this number fixed or an honest, explained threshold — an ungated target is a wish, but a gate on a number nobody can reproduce is worse.
