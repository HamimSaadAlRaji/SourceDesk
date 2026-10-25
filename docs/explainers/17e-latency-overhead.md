# 17e — What Does the Framework Cost in Time? (and a Fresh p95)

**Phase:** 17 · **Date:** 2026-10-25 · **Files:** `experiments/17c_framework_overhead.py`, `experiments/17d_latency.py`, `docs/results/framework-overhead-raw.json`, `docs/results/orchestration-latency-raw.json`
**Topics from the course:** Latency vs. response quality

## 1. What I built

Two measurements. (1) The framework in isolation: every stage an instant in-memory fake, so a request is pure orchestration. (2) The real stack, both paths, 40 dev queries back to back per path, which also produces the fresh direct-path p95 that the Phase 16 hand-off asked for.

## 2. Why this approach

A live comparison cannot see a millisecond. Provider latency varies by seconds, so the question "what does LangGraph cost" has to be answered by removing everything that is not LangGraph; the live run then answers a different question: does anything *visible* change. The live run is paired (each query through both paths back to back, alternating which goes first) so provider drift and GPU wake hit both equally, and the two pipelines share models, LLM clients and retriever so they differ only in orchestration.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Live runs only | One number, realistic | Noise floor (below) is ~100x the effect; it would report "no difference" without saying how little it could detect. |
| 180 queries live | Tighter p95 | Sequential and paid: ~25 min and ~$0.6 for two paths, and the confidence scorer alone is ~4 s CPU per query (Phase 13); 40 was agreed. |
| Microbenchmark with the real tracer | Closer to prod | Tracing is identical code in both paths (17c); a recording tracer keeps its cost out of the difference. |
| Idle pacing between queries (Phase 13 used 70 s) | Comparable to Phase 13 | That pacing existed for Groq's per-minute token caps; the stack is now paid OpenAI and does not need it. A difference from Phase 13's method, stated. |

## 4. How it works

`17c_framework_overhead.py`: 4 scenarios, 2,000 requests per path each, interleaved in blocks of 200 alternating order, GC before each block, wall time around `pipeline.answer`. `17d_latency.py`: seed 17, 40 dev queries, semantic cache off and empty exact caches (every request a cold miss), one unmeasured warm-up request per path, `AnswerPipeline.warm_up()` first; percentiles from Phase 13's `stage_latency_summary` (linear interpolation) so the numbers mean what the Phase 13 ones meant.

## 5. What the numbers said

**Framework cost, stubbed stages (µs per request, graph minus direct):**

| Scenario | Direct p50 | Graph p50 | Delta p50 | Delta p95 |
| --- | --- | --- | --- | --- |
| cache hit | 26 | 634 | +608 | +789 |
| policy escalation | 54 | 859 | +805 | +911 |
| low-confidence escalation | 141 | 1,288 | +1,147 | +1,381 |
| generated answer | 182 | 1,352 | +1,171 | +1,349 |

About 0.2 ms per node visited: +0.6 to +1.2 ms per request. That is a 7x to 24x slowdown of the *orchestration* alone (generated answer 182 → 1,352 µs; cache hit 26 → 634 µs) and about 0.03% of the 3.6 s live median.

**Live, 40 queries per path (ms):**

| | n | p50 | p95 | max |
| --- | --- | --- | --- | --- |
| direct, all | 40 | 3,577 | **11,799** | 16,344 |
| graph, all | 40 | 3,406 | 10,718 | 14,358 |
| direct, generate path | 35 | 3,895 | 12,094 | 16,344 |
| direct, policy escalation | 3 | 831 | 885 | 892 |
| direct, confidence escalation | 2 | 1,150 | 1,178 | 1,181 |

Paired graph-minus-direct wall time: median **-22 ms**, bootstrap 95% CI **[-162, +124] ms**, spread -8,806 to +4,555 ms, graph faster in 21 of 40. Indistinguishable from zero, and unable to resolve 1 ms. All 40 queries took the same path in both runs. Spend $0.133 (a first identical run, $0.131, was lost to a bug in my summary code, so the latency work cost $0.264 in total).

**Fresh direct p95 (the Phase 16 hand-off): 11.8 s** against a 3 s target and the last figure of 5.6 s (qwen era). Generate-path mean shares: generation 53% (p95 7.1 s), confidence 23% (p95 2.2 s, one 10.9 s outlier), routing 14% (p95 0.9 s), validation 9% (p95 2.7 s).

## 6. What surprised me / what to watch for

- **The p95 is much worse than anyone had on file, and the cause moved.** Phase 13 found the confidence scorer dominant (67% of p50). Now generation is (53%), and validation went from 3 ms to a p95 of 2.7 s because citation entailment is an LLM call. p50 alone (3.6 s) is already over budget. This is outside Phase 17's remit (it was handed here only as a baseline) and is recorded, not fixed.
- n = 40 with 35 generate-path requests: p95 is roughly the 2nd-largest value. It bounds the tail; it does not measure it.
- The graph looking 0.3 s faster on the mean is noise, not a finding: the CI spans zero by a wide margin. I would not put it in a README.
- I lost a paid run by computing the summary before saving the raw rows. The script now writes rows first.

## 7. Where this fits and what's next

The framework adds ~1 ms and 6 stack frames (17f). The comparison is in `docs/results/framework-comparison.md`.
