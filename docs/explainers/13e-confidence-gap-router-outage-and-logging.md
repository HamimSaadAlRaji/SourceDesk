# 13e — The Confidence-Stage Gap, a Silent Router Outage, and Structured Logging

**Phase:** 13 · **Date:** 2026-09-30 · **Files:** `src/aegis/observability/logging.py`, `src/aegis/pipeline.py`, `src/aegis/router/classifier.py`, `tests/test_structured_logging.py`, `experiments/13h_confidence_localize.py`, `13i_gpu_wake_paired.py`
**Topics from the course:** Logging, monitoring & observability (Module 5, #31); Latency vs. response quality (#35)

## 1. What I built

Structured JSON logging with the trace ID attached to every record of a request (Phase 13 build item 2), a log line for every router failure path that used to be silent, and two experiments that narrow down why the `confidence` stage costs ~2.4 s in the paced pipeline batches but ~0.5-0.7 s in isolation.

## 2. Why this approach

Chasing the confidence gap by comparing experiments run on different queries gave me conclusions I then had to retract (13d §6). This round fixes the design: the **same queries in every arm**, arm order interleaved (13i: each query run with and without a throwaway "GPU wake" rerank, alternating order), and sub-timers inside the stage so time is attributed to a component instead of inferred from a total.

Structured logging is stdlib `logging` + a `ContextVar` trace ID + a JSON formatter — no structlog/loguru. The requirement is "JSON lines with a trace_id", which is ~40 lines; CLAUDE.md §2 says a dependency needs a reason and there isn't one.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ------------------ |
| `structlog` | Mature, processors/pipelines | The only requirement is JSON + trace_id; a dependency for that is the direction CLAUDE.md says needs defending. |
| Make the router raise instead of failing safe on API errors | Failure is loud | Wrong trade: raising turns a quota outage into 500s; the fail-safe (escalate) is the correct North-Star behavior. The defect was *invisibility*, so the fix is a log line, not a behavior change. |
| Keep chasing with different-query experiments | Cheaper | Produced two claims I had to retract. |

## 4. How it works

```python
with bind_trace_id(trace_id), self.tracer.start_as_current_observation(...):
    log_event(_log, logging.INFO, "pipeline.request_start", query_id=query.id)
    ...
    log_event(_log, logging.INFO, "pipeline.request_end", cache_status=..., escalate=..., stage_latencies_ms=...)
```

The formatter reads the trace ID from a context variable, so any module (the router, the caches) logging inside a request is joined to it without passing the ID around. `pipeline.escalated` is logged at WARNING with the reason and whether it happened before or after retrieval. The trace ID is the same one on `PipelineResult` and the Langfuse trace.

## 5. What the numbers said

`uv run pytest -q`: 483/483 (TDD'd RED-first; +11).

**Localization (13h, real Groq router + real tracer, fake generation, same 8 queries; sub-timed):** in every row the extra time is entirely the cross-encoder call — `rerank` ≈ the whole stage, `encode` ≈ 2 ms, remainder ≈ 1 ms. Stage p50: 10 s gap 638 ms (n=7), 0 s gap 469 ms (n=7).

**GPU wake-up (13i, paired, 6 queries × 2 arms, 70 s gap, fixed router + real tracer):** plain p50 656 ms (max 1,072); with a throwaway 1-pair rerank first p50 494 ms (max 855). The wake arm was faster in 6/6 pairs by ~127 ms typical, but the throwaway costs 199 ms p50 itself: net loss, not a fix.

**Ruled out as sufficient explanations for 2.4 s:** background load, the real tracer, 70 s idle gaps, the real router, GPU idle wake-up. **Same-query check:** query `2689` costs 855-1,072 ms in 13i, 2,299-3,326 ms in the earlier batches — so it's the configuration, not the data.

## 6. What surprised me / what to watch for

**I exhausted the shared Groq daily quota and the router hid it.** `qwen/qwen3.8-27b` hit its 200,000 tokens/day cap (`Used 199902`). The router's `except APIError: return ""` fed the fail-safe, so *every* query escalated as `billing_account_legal` — real docs questions, even the same query routing differently across runs — with no log, no trace field, nothing. A trivial test call still passed (it requested few enough tokens to fit in the remaining ~98), which is why it wasn't obvious. This cost me one unusable experiment run and it is exactly the failure mode Phase 13 exists to surface. It also means today's quota is gone for the user's deferred Phase 9 sampling sweep, which needs that same model.

**Still invisible at the result level.** The escalation reason on `PipelineResult` for an outage-driven escalation reads "billing_account_legal always hard-escalates...", which is misleading; only the log line and trace ID reveal the real cause. Fixing that needs the classifier to return more than a `Route` (an interface change I did not make here).

**The one untested difference.** The fast experiments use fakes for generation and validation; the slow batches run real ones in-process. `AEGIS_I_REAL_STACK=1 uv run python experiments/13i_gpu_wake_paired.py` runs the same paired design with the real stack (needs Groq/Gemini quota). Not run.

**Small n throughout** (6-15 per arm). A 127 ms paired difference across 6 pairs is a direction, not a measurement.

## 7. Where this fits and what's next

Structured logging closes build item 2. Still open in Phase 13: the audit-trail demo (build item 5) and the p95 < 3 s exit criterion (unmet; the ~1.5-2 s confidence gap is the largest unexplained piece and is now narrowed to one candidate). The audit trail can now be built on both the trace and the trace-ID-tagged logs.
