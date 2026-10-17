# 14 — The Streaming Pipeline

**Phase:** 14 · **Date:** 2026-10-01 · **Files:** `src/aegis/events.py`, `src/aegis/pipeline.py`, `src/aegis/generation/pipeline.py`, `src/aegis/generation/llm/fallback.py`, `src/aegis/cache/exact_match.py`, `tests/test_pipeline_stream.py`, `experiments/14a_stream_smoke.py`
**Topics from the course:** Latency vs. response quality (Module 5, #35); Handling hallucinations (#25)

## 1. What I built

`AnswerPipeline.answer_stream(query)` yields typed events — `token`, `citation`, `escalation`, `done`, `error` — and `answer(query, on_event=...)` is the same request with a callback. The contract is fixed by tests: answer = `token+ citation* done`; cache hit = `token citation* done`; escalation = `escalation done`; failure = `token* error` (terminal, never followed by `done`). This is the layer Phase 14's HTTP endpoint will wrap; no HTTP code exists yet.

## 2. Why this approach

Citation validation needs the *complete* answer, so tokens cannot wait for it without making streaming cosmetic. Decision (user-confirmed, 2026-10-01): **stream tokens live, validate after, then emit `citation` events.** The argument that this isn't a safety regression: the existing non-streaming pipeline already serves the answer text even when some claims fail validation — validation only decides which citations are *listed*. So the user-visible guarantee ("a citation shown was validated") is the same; the only new thing is a short window where raw `[n]` markers are visible before they're confirmed, which the UI can show as pending.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ------------------ |
| Validate first, replay the finished text as a stream | Every visible `[n]` is already validated | Streaming becomes cosmetic and the first token lands ~0.6 s later (see §5). Doesn't change which text is served, so it buys no real safety. |
| Stream live, retract (escalate) if validation fails | Strongest against a confidently wrong answer | Changes Phase 9/10 behavior and needs a new threshold decision; beyond plan.md's Phase 14 list. Left for Phase 15's output-side checks. |
| A generator that holds the Langfuse span context managers open across `yield`s | No worker thread | A web server pulls a sync generator one `next()` at a time, potentially on different threads; OTel context and the trace-id `ContextVar` tokens must be reset on the thread that set them. I did not reproduce a failure — this is a design-time judgement, guarded by a test that all spans open and close on one thread. |
| Add a separate streaming code path beside `answer()` | Leaves `answer()` untouched | Two copies of the cache/route/escalate/generate branching would drift. One implementation, with an optional event sink. |

## 4. How it works

`_run` is the single implementation. With no `on_event`, generation takes the original `generate_with_provider` path, so the eval harness and every pre-existing test see identical behavior. With one, `GenerationPipeline.generate(on_token=...)` iterates the provider's stream (`generate_with_provider` is itself a join of that stream, so the final text is the same), emitting each token. `answer_stream` runs `_run` on one daemon worker thread and hands events across a `queue.Queue`, so the span context managers and `bind_trace_id` never straddle threads. Any exception becomes an `ErrorEvent` carrying only the exception class and a fixed message — provider exceptions can include account identifiers (a Groq 429 includes the org id), so the detail goes to the log, not the client.

Cache entries now store `CitedChunk` (citation number, chunk id, doc id, heading path, source text) so a cache hit replays real `citation` events with no corpus lookup; the field defaults to `[]`, so entries written before this change still load. `FallbackLLMClient._stream_with_provider` became public `stream_with_provider`.

## 5. What the numbers said

`uv run pytest -q`: **508/508** (+12 streaming tests, TDD'd RED-first). The test that matters most: a spy validator records how many tokens had reached the consumer when validation began — it sees all 4, proving validation runs after delivery.

Real run (real Groq router + generation, Qdrant, cross-encoder; `golden::issue::604`; n=1, warm): **first token 1,298 ms**; 85 token events over 252 ms (last at 1,550 ms); `citation` at 1,921 ms (371 ms after the last token); `done` at 1,927 ms. Cache replay of the same query: `token`, `citation`, `done` all at 3 ms. Against validate-first, which would show text at ~1,921 ms, live streaming moves first visible text ~0.6 s earlier.

## 6. What surprised me / what to watch for

**Streaming helps less than the plan assumes.** The model emits the whole answer in ~250 ms, so the user waits ~1.3 s for the *first* token whether or not we stream — that wait is routing + retrieval + confidence scoring, the same stage Phase 13 couldn't make fast. The perceived-latency lever is still the pre-generation path, not the token stream.

**Mid-stream failure is a real state, not an edge case.** The fallback client deliberately never switches providers after the first token, so a client can receive partial text and then `error`. That is tested with fakes only; I did not provoke a real mid-stream failure.

**Not verified:** behavior when the HTTP client disconnects mid-stream (the worker thread would run to completion; not exercised), and any real browser consumer. The unvalidated-`[n]` window was 371 ms in one run; with a tier-2 entailment call it can be longer (validation p95 was ~0.7-1.0 s in Phase 13's batches).

## 7. Where this fits and what's next

Unit 2: a FastAPI `POST /chat` that serializes these events as SSE, plus `/health`, with a pipeline factory (three experiment scripts already duplicate the construction code) and a real streamed request against a running server. FastAPI/uvicorn aren't dependencies yet; adding them is justified by CLAUDE.md §2's "import FastAPI" and will be recorded there. Later units: `/trace/{id}` (wraps `audit.py`), conversation memory, the UI, feedback, rate limiting.
