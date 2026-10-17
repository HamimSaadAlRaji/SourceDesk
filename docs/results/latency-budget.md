# Latency Budget — Baseline (Phase 13, Unit 3)

Per-stage wall-clock latency of the real `AnswerPipeline` (real shipped dense retrieval, real cross-encoder
confidence scoring, real Groq router + generation calls), measured by the pipeline itself
(`PipelineResult.stage_latencies_ms`), not read back from Langfuse. All numbers in ms.
Reproduce: `AEGIS_LAT_N_POS=15 AEGIS_LAT_N_NEG=5 AEGIS_LAT_PACE_SECONDS=70 AEGIS_LAT_OUT=latency-attribution-paced-raw.json
uv run python experiments/13c_latency_attribution.py`, then `uv run python experiments/13d_latency_report.py`.

## Sample (paced baseline)

- 20 queries attempted (20 succeeded, 0 errored), dev split, seed 20260802.
- **70 s idle between queries**, so Groq's free-tier per-minute token caps have fully reset before each request. This measures the pipeline, not rate-limit backoff -- see the burst section for why that distinction matters.
- Cold start (first query in process, lazy model loads): reported separately, excluded from percentiles.
- Warm: 19 queries -- 12 generate path, 7 escalated.
- Generation provider mix: {'groq': 12}.
- Caches emptied before the run: every query is a cache miss. Cache-hit latency is Phase 12's number.
- **Small-sample caveat:** n=12 on the generate path. p95/p99 are effectively the max of a handful of points -- they bound the tail, they do not measure it.

## Cold start (first query in process)

| Stage | ms |
|---|---|
| cache_lookup | 2 |
| routing | 339 |
| retrieval | 36 |
| confidence | 17,435 |
| generation | 974 |
| validation | 417 |
| **wall total** | 19,208 |

## Generate path (warm, paced)

| Stage | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| cache_lookup | 12 | 6 | 5 | 10 | 10 | 10 |
| routing | 12 | 265 | 251 | 334 | 349 | 352 |
| retrieval | 12 | 46 | 44 | 66 | 68 | 68 |
| confidence | 12 | 2,546 | 2,520 | 3,233 | 3,371 | 3,405 |
| generation | 12 | 938 | 920 | 1,443 | 1,527 | 1,548 |
| validation | 12 | 178 | 3 | 656 | 666 | 668 |
| **sum of stages** | 12 | 3,979 | 4,012 | 4,688 | 4,777 | 4,799 |

## Escalation path (warm, paced)

| Stage | n | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| cache_lookup | 7 | 5 | 5 | 6 | 6 | 6 |
| routing | 7 | 248 | 239 | 310 | 314 | 315 |
| retrieval | 6 | 47 | 40 | 68 | 71 | 72 |
| confidence | 6 | 1,898 | 1,844 | 2,846 | 2,943 | 2,967 |
| escalation | 7 | 1 | 0 | 1 | 1 | 1 |
| **sum of stages** | 7 | 1,921 | 2,084 | 3,128 | 3,241 | 3,269 |

## Against the p95 < 3 s budget

| Path | n | p50 | p95 | Budget | Verdict |
|---|---|---|---|---|---|
| Pre-generation (cache + routing + retrieval + confidence), all warm | 19 | 2,695 | 3,486 | 3,000 | OVER |
| Full generate path | 12 | 4,012 | 4,688 | 3,000 | OVER |
| Escalation path | 7 | 2,084 | 3,128 | 3,000 | OVER |

## Share of generate-path p50 by stage (paced)

- cache_lookup: 0.1% (5 ms)
- routing: 6.7% (251 ms)
- retrieval: 1.2% (44 ms)
- confidence: 67.3% (2,520 ms)
- generation: 24.6% (920 ms)
- validation: 0.1% (3 ms)

## Burst behavior (unpaced, back-to-back) -- why the baseline is paced

The first run sent 25 queries back to back with no idle time. Same code, same corpus, same
models -- and the numbers were wildly different. The likely cause is Groq's free-tier per-minute token
caps (Phase 9: 7,000 ITPM on this model) triggering retry-after waits inside the LLM calls, which would
land in whichever stage made the throttled call. **Supported by the effect, not by direct observation of
the mechanism:** spacing queries 70 s apart removed the stalls (routing p50 5,353 -> 251 ms), but 429s and
retries were not logged, so this was not confirmed at the HTTP level. Generate path, warm:

| Stage | paced p50 | burst p50 | paced p95 | burst p95 |
|---|---|---|---|---|
| cache_lookup | 5 | 2 | 10 | 2 |
| routing | 251 | 5,353 | 334 | 10,977 |
| retrieval | 44 | 54 | 66 | 82 |
| confidence | 2,520 | 3,656 | 3,233 | 6,650 |
| generation | 920 | 26,401 | 1,443 | 37,542 |
| validation | 3 | 3 | 656 | 6,348 |
| **sum of stages** | 4,012 | 38,080 | 4,688 | 55,338 |

**Unexplained:** `confidence` is local CPU work (cross-encoder + encoder, no LLM call) yet was ~45% slower in the burst run (p50 3,656 vs 2,520 ms). Rate limiting cannot explain that. Machine load during the burst run is a possibility (other processes were active), but it was not measured or isolated.

Burst run: n=18 generate-path queries, 6 escalated, raw data in `latency-attribution-burst-raw.json`. Burst rows are real measurements of what a user sees when traffic exceeds the free-tier cap; they are not a measurement of the pipeline's own latency.


---

# Unit 4 -- optimization against the budget

One code change was made: `AnswerPipeline.warm_up()` (real retrieval + real confidence scoring at startup,
no LLM calls, no cache/trace side effects). Measured with `AEGIS_LAT_WARMUP=1`, same 70 s pacing, same
sampling as the baseline. **Warm-up cost at startup: 30.3 s** (encoder load 14.2 s + reranker load 16.0 s) -- moved, not removed.

## First request: before vs. after warm-up

| Stage | before (cold) | after warm-up |
|---|---|---|
| cache_lookup | 2 | 1 |
| routing | 339 | 433 |
| retrieval | 36 | 52 |
| confidence | 17,435 | 1,663 |
| generation | 974 | 627 |
| validation | 417 | 172 |
| **wall total** | 19,208 | 2,957 |

## Steady state: re-baseline vs. Unit 3 baseline (generate path, warm)

| Stage | Unit 3 p50 | re-run p50 | Unit 3 p95 | re-run p95 |
|---|---|---|---|---|
| cache_lookup | 5 | 6 | 10 | 7 |
| routing | 251 | 396 | 334 | 737 |
| retrieval | 44 | 37 | 66 | 44 |
| confidence | 2,520 | 2,401 | 3,233 | 3,357 |
| generation | 920 | 1,151 | 1,443 | 2,211 |
| validation | 3 | 2 | 656 | 969 |
| **sum of stages** | 4,012 | 4,421 | 4,688 | 5,649 |

| Path | Unit 3 p95 | re-run p95 | Budget | Verdict (re-run) |
|---|---|---|---|---|
| Pre-generation | 3,486 | 4,030 | 3,000 | OVER |
| Full generate path | 4,688 | 5,649 | 3,000 | OVER |
| Escalation path | 3,128 | 2,945 | 3,000 | within |

**Steady state did not improve.** The re-run is within noise of the baseline (some stages slightly worse).
An unrelated Docker stack had been running during the baseline and was gone for the re-run (background load was never
measured, so its effect on the baseline is unknown). Removing it did not reduce the confidence stage's cost, so it does
not appear to explain that cost. **The p95 < 3 s budget is not met.** (Escalation-path n=7: its "within" is not robust.)

## The unresolved gap: `confidence` costs ~2.4 s in the pipeline but far less in isolation

The `confidence` stage is always the real `ConfidenceScorer` (cross-encoder on MPS, pool of 3 pairs). What else ran
in the process varies by row (stated in each). Separate experiments, small n; 13h/13i reuse the same queries across arms:

| Condition | n | p50 | p95 | Source |
|---|---|---|---|---|
| Scorer alone, back-to-back | 15 | 560 | 1,470 | 13f |
| Scorer alone, 70 s idle before each call | 5 | 1,031 | 3,202 | 13f |
| Inside pipeline, no-op tracer, back-to-back | 12 | 502 | 1,344 | 13g |
| Inside pipeline, real Langfuse tracer, back-to-back | 12 | 926 | 6,070 | 13g |
| Full real pipeline (real router + real generation + real validation + real tracer), 70 s idle (Unit 3 baseline) | 12 | 2,520 | 3,233 | 13c |
| Same, re-run with warm-up | 12 | 2,401 | 3,357 | 13c |
| 13h: real Groq router + real tracer, fake generation, 10 s gap (same 7 queries) | 7 | 638 | 1,207 | 13h |
| 13h: same, 0 s gap | 7 | 469 | 969 | 13h |
| 13h: same, 70 s gap (3 usable rows; router quota then ran out) | 3 | 1,497 | 1,656 | 13h |
| 13i: fixed router + real tracer, fake generation, 70 s gap, plain (paired, same queries) | 6 | 656 | 1,072 | 13i |
| 13i: same, with a 1-pair throwaway rerank first ("wake") | 6 | 494 | 855 | 13i |

**What this rules out (as sufficient explanations):** background load (unrelated stack removed, no change); the
real Langfuse tracer (13g's ~2x median did not replicate: 13i with the real tracer is 656 ms); idle gaps of 70 s
(13i plain, 656 ms); the real router LLM (13h); GPU wake-up (a throwaway rerank saves ~127 ms typical across 6/6
pairs but costs 199 ms p50 itself -- net loss). **Localized:** in 13h the extra time is entirely the cross-encoder call
(encode ~2 ms, everything else ~1 ms). **Same-query check:** query `2689` costs 855-1,072 ms in 13i but 2,299-3,326 ms
in the paced/burst batches, so it is the run configuration, not the data. **The one difference not yet tested:** the
batches also run real generation and real citation validation in-process; the fast experiments use fakes for both.
Test prepared (`AEGIS_I_REAL_STACK=1 uv run python experiments/13i_gpu_wake_paired.py`), blocked on Groq quota.
Earlier drafts of this section claimed idle gaps and the tracer each roughly doubled the cost; those claims did not
survive a paired, same-query design and are retracted.

## Follow-up 2026-10-01: the untested hypothesis is now tested, and refuted

- **Real generation + validation in-process (13i, `AEGIS_I_REAL_STACK=1`, same queries, 70 s gaps, no abort):** plain
  confidence p50 647 ms (n=6, max 1,034); with a throwaway rerank 516 ms. Same as the fake-stack run (656 ms), not ~2.4 s.
- **First-call / per-shape compile cost (13k, one process, 3 repeated scorings per query):** call1 ~= call2 ~= call3 for
  all six queries (e.g. 833/793/779 ms), including padded shapes never seen before. Refuted. Cost follows the padded
  sequence length: (3,318) 238 ms, (3,353) 265, (3,495) 364, (3,512) 480-880.
- **Power state:** the machine is on battery with Low Power Mode ON and the fast runs were fast. Not the explanation.
- **Re-running the original baseline script (13c) on the exact 6-query morning-pilot sample, 70 s pacing, warm-up on:**
  confidence ms, pilot (Sep 30) vs rerun (Oct 1): 2689 1,040 vs 972; 2884 1,420 vs 655; 3212 1,375 vs 527; **753 2,448 vs 2,586**;
  219 2,318 vs 1,106. Generate-path wall times in the rerun: 3,234 / 3,734 / 2,745 ms (n=3).
- **What this leaves:** query `753` costs ~0.5-0.8 s in 13i and 13k but ~2.5 s in 13c, reproduced across two 13c runs. Something
  specific to the 13c harness (it reuses ONE pipeline instance and caches across queries; 13i rebuilds the pipeline per run)
  inflates that stage for some queries. **Not bisected.** Consequence: the p95 verdict depends on the harness. 13c numbers say the
  generate path is over budget (p95 5.6 s in the 20-query run); the other harnesses imply a materially faster confidence stage but
  were not run end to end, so no p95 can be claimed from them. The budget criterion remains unmet and unresolved.
