# 13b — The Answer Pipeline

**Phase:** 13 · **Date:** 2026-09-28 · **Files:** `src/aegis/pipeline.py`, `tests/test_pipeline.py`, `experiments/13b_pipeline_live_trace.py`
**Topics from the course:** Logging, monitoring & observability (Module 5, #31); Latency vs. response quality (Module 5, #35)

## 1. What I built

`AnswerPipeline.answer(query) -> PipelineResult` — the first real composition of every independently-built stage (exact/semantic cache, router, retrieval, confidence scoring, generation, citation validation) into one end-to-end request. Input: a `Query`. Output: a `PipelineResult` carrying the final text plus every piece of evidence a later audit needs (cache status, route, retrieved chunks with scores, confidence signals, provider, prompt version, citations, and a real Langfuse `trace_id`). Every stage opens its own Langfuse span with real input/output attached, satisfying Phase 13 build item #1 for the first time against a real request rather than a synthetic one.

## 2. Why this approach

Phase 13's own build item #1 ("tracing every stage: routing decision, retrieved chunks with scores, rerank scores, confidence, prompt version, generation, validation result, cache status") presupposes something exists to trace. It didn't: five phases (5–12) each built and unit-tested one stage in isolation, wired together only ad hoc inside experiment scripts, never as one real `answer(query)` call. Phase 14 ("API, Streaming & UI") is where an orchestrator would normally get built — but Phase 13 needs it *now*, to attach spans to. Flagged this gap directly to the user rather than either quietly building a throwaway orchestrator under Phase 13's name or faking "every stage traced" against an experiment script with no real request concept. Confirmed via `AskUserQuestion`: build it now, as real shared infrastructure, so Phase 14 wraps this exact object in FastAPI/SSE instead of rebuilding it.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ------------------ |
| Trace at the experiment-script level only (add Langfuse spans inside e.g. `11e_judge_validation_batch.py`) | No new production code, ships tracing immediately | There's no real "end-to-end request" to point the audit-trail demo (build item #5) at — a batch-eval script isn't a request. Phase 14 would still have to build the real orchestrator from scratch later, so this doesn't reduce total work, it defers it. |
| Reuse `router/policy.py`'s and `guardrails/confidence.py`'s existing `handle()` functions as black boxes (`Callable[[str], str]` generate/escalate closures) | Already-tested single-fork-point functions, zero new enforcement logic | Their `str`-in/`str`-out shape can't carry the structured evidence (`ConfidenceResult`, retrieved chunks, `EscalationPayload`) the audit trail needs, and can't have a span attached per stage without restructuring them anyway. Reused their pure decision functions (`decide`, `decide_escalation`) instead, and wrote the orchestrator's own enforcement with its own spy-based proof tests — same audit discipline, richer return type. |
| Wire `corpus_version` to `IndexManifest.corpus_version()` now, resolving Phase 12's disclosed open item | "Real" version tracking instead of a placeholder string | No manifest has ever been written for the live collections (`data/processed/manifests/` doesn't exist) — `IndexManifest.load()` on a missing path returns an *empty* manifest, so its `corpus_version()` would hash zero real content while looking like a real value. Building a real manifest means running delta-indexing against production Qdrant collections for the first time, a real, separate, deliberate decision — not a side effect of an observability unit. Kept the honest placeholder string instead. |

## 4. How it works

Stage order matches `router/classifier.py`'s own documented architecture: cache → router → retrieval → confidence → generation. Every collaborator is dependency-injected (same shape as `GenerationPipeline`/`RerankedRetriever`), including the tracer:

```python
def _answer(self, query: Query) -> PipelineResult:
    cached, status = self._check_cache(query)
    if cached is not None:
        return PipelineResult(query=query, cache_status=status, escalate=False, text=cached.text, ...)

    route = self._route(query)
    if decide_route(route).escalate:                       # billing/legal: never reaches retrieval
        return self._escalate(query, status, route=route, reason=...)

    candidates = self._retrieve(query)
    confidence = self._score_confidence(query, candidates)
    outcome = decide_escalation(confidence.confidence, self.tau)
    if outcome.escalate:                                    # low confidence: never reaches generation
        return self._escalate(query, status, route=route, reason=outcome.reason, ...)

    return self._generate(query, status, route, candidates, confidence)
```

Each `_route`/`_retrieve`/`_score_confidence`/`_generate` method wraps its own real span:

```python
def _retrieve(self, query: Query) -> list[tuple[Chunk, float]]:
    with self.tracer.start_as_current_observation(
        name="retrieval", as_type="retriever", input={"top_k": self.retrieval_top_k}
    ) as span:
        candidates = self.retriever.search(query.text, top_k=self.retrieval_top_k)
        span.update(output={"chunks_returned": len(candidates),
                             "scores": [{"chunk_id": c.id, "score": round(s, 4)} for c, s in candidates]})
        return candidates
```

`as_type` per stage, matching design record 13 §6's mapping: retrieval → `retriever`, confidence → `evaluator`, generation → `generation` (with `model=`), routing/escalation/validation → `guardrail` (each is a safety-relevant gate, not a plain computation step). A successful (non-escalated) generation writes through to **both** caches — a real answer, including a genuine honest refusal, is exactly what should be served to an identical future query (Phase 11's faithfulness root-cause work already established that a correct refusal isn't a defect to avoid caching).

The two structural audit properties are proven the same way `router/policy.py`/`guardrails/confidence.py` already prove theirs — a spy collaborator that raises `AssertionError` if ever called:

```python
def test_billing_route_escalates_before_retrieval_or_generation():
    pipeline, tracer = _pipeline(router=_FixedRouter(Route.BILLING_ACCOUNT_LEGAL), retriever=_RaisingRetriever())
    result = pipeline.answer(Query(id="q3", text="please refund my invoice"))
    assert result.escalate is True and result.retrieved == []
```

## 5. What the numbers said

`uv run pytest -q`: 464/464 passed (+9 new, TDD'd first — confirmed RED via `ModuleNotFoundError` before `pipeline.py` existed, then GREEN). Real live run against the actual corpus (19,236 chunks), real Qdrant, real Groq router + generation:

```
real query: 'golden::issue::604' -> 'Query parameters with hyphens...'
cache_status: miss
route: docs_howto
escalate: False
confidence: 0.7953
provider: groq
citations: ['issue::4842::heading_aware::8']
text: To retrieve a query parameter that contains hyphens, you must use the `alias` argument...
```

All 7 real spans (`answer_query`, `cache_lookup`, `routing`, `retrieval`, `confidence`, `generation`, `validation`) confirmed present in Langfuse Cloud for this trace. A second real run against the same query correctly short-circuited to `cache_status: exact_hit` with only 2 spans (`answer_query`, `cache_lookup`) — the write-through from the first run being served back, proving the cache/tracing composition works, not just each piece alone.

## 6. What surprised me / what to watch for

**Real ingestion latency scales with trace size, not a fixed constant.** Unit 1's synthetic 3-span trace was fully queryable at 8 seconds. This real 7-span trace (with real generation text and citations, larger payloads) still had 4 of 7 spans missing at 8 seconds and wasn't complete until ~14. A single hardcoded sleep is the wrong tool for verifying real traces; rewrote the live-trace script to poll every 5 seconds up to 30 seconds instead. Anything that reads traces back programmatically (not just views them in the Langfuse UI) needs to tolerate this, and the tolerance isn't a fixed number.

**The verification script's own first version had a real bug, not the pipeline.** Its second run (same query, same throwaway cache file) correctly hit the exact-match cache and short-circuited to 2 spans — exactly what the pipeline is supposed to do. The verification script's hardcoded "expect all 7 spans" check flagged this as a failure. Fixed by making the expectation branch-aware on `result.cache_status`/`result.escalate`, matching the pipeline's own real branching. Worth remembering for the eventual CI eval gate (Phase 16): any check against real traced behavior has to account for which branch a given real query legitimately takes, not assume every request reaches every stage.

**Confidence came out high (0.7953) on a real production query on the first try**, comfortably above tau, and the generation was a real, well-grounded, single-citation answer. No new confidence-scorer or retrieval issue was surfaced by this — a mildly reassuring, if statistically meaningless (n=1), sign that the shipped Phase 6/7/10 configuration behaves the way its own measurements said it should on a real end-to-end path, not just in an isolated ablation.

## 7. Where this fits and what's next

This is the object every remaining Phase 13 build item attaches to: per-stage latency attribution (build item #3) reads real span durations off this same trace; the p95 < 3s optimization work (build item #4) profiles this exact call path; the audit-trail demo (build item #5) reconstructs evidence from exactly the `PipelineResult`/trace pair this unit produces. It's also Phase 14's real starting point — `POST /chat` wraps `AnswerPipeline.answer()` rather than rebuilding request orchestration, and the `trace_id` field exists specifically because plan.md's Phase 14 already names a `GET /trace/{id}` endpoint that needs it. Next: Unit 3, per-stage latency attribution (p50/p95/p99) against a real batch of queries through this same pipeline.
