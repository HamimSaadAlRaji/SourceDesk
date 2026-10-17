# 14b — FastAPI Chat Endpoint over SSE

**Phase:** 14 · **Date:** 2026-10-01 · **Files:** `src/aegis/api/app.py`, `src/aegis/api/main.py`, `src/aegis/factory.py`, `src/aegis/config.py` (`ApiSettings`), `tests/test_api.py`
**Topics from the course:** Latency vs. response quality (Module 5, #35)

## 1. What I built

`POST /chat` takes `{"query": ...}` and streams the pipeline's typed events (`token`, `citation`, `escalation`, `done`, `error`) as Server-Sent Events. `GET /health` returns `{"status":"ok"}`. `create_app(pipeline)` takes the pipeline as an argument; `factory.build_pipeline()` is the one place that assembles the real one, and the server builds and warms it in the app lifespan. Not built yet: `/trace/{id}`, `/feedback`, rate limiting, conversation memory, the UI.

## 2. Why this approach

- **SSE, one direction.** The server pushes tokens; the client sends one request per question. Nothing in the contract needs the client to talk mid-stream, and SSE rides on a plain POST with ordinary HTTP status codes, proxies, and `curl`.
- **Hand-written frames, no SSE library.** A frame is `event: <type>\ndata: <json>\n\n`. The event schema already exists in `aegis.events`, so a library would add a dependency to produce one f-string.
- **Pipeline injected.** The tests use a fake that yields canned events or raises if reached, so the wire format and the validation boundary are tested without models. This is the same trick as Unit 1.
- **Validation is HTTP 422 before anything runs; pipeline failure is HTTP 200 with a terminal `error` event.** Once the first byte is sent the status code is fixed, so a mid-stream failure can only be an event. The test `test_invalid_requests_rejected_before_pipeline` uses a pipeline that raises if touched.
- **Dependencies added:** `fastapi` and `uvicorn` (CLAUDE.md §2 lists FastAPI as an import, not a build), `httpx` as a dev dependency (FastAPI's `TestClient` needs it).

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| WebSockets | Bidirectional; cancel messages | Nothing here is client-to-server mid-stream. Costs a custom protocol, no automatic status codes, and harder to demo with `curl`. Reconsider only if the UI needs in-stream cancel. |
| `sse-starlette` | Handles keep-alives, disconnect detection | One more dependency for functionality not yet needed; the disconnect behavior below is measured and acceptable. |
| `EventSource` in the browser | Native reconnect | It only does GET. The UI will use `fetch` + a stream reader on POST instead (Unit 4). |
| Per-experiment pipeline copies (status quo) | Experiments stay self-contained | The server is the fourth copy. I did **not** rewrite the existing experiments onto the factory: they are records of what produced published numbers, and re-pointing them changes what they ran. |
| `max_query_chars = 2000` | Tight and conventional | **Rejected on measurement.** Over the 180 golden queries: p50 773 chars, p95 4,351, max 16,755; 43 are over 2,000 (whole issue bodies). A 2,000 cap rejects 24% of the golden set. Set to 20,000, a guard rail above the observed max, not a tuned value. |

## 4. How it works

```python
def sse_frame(event): return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"

@app.post("/chat")
def chat(req: ChatRequest):          # sync def: Starlette pulls the generator in a threadpool
    def frames():
        try:
            for event in app.state.pipeline.answer_stream(query): yield sse_frame(event)
        except Exception: yield sse_frame(ErrorEvent(...))   # stream always ends typed
    return StreamingResponse(frames(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```

The Unit 1 worker-thread-plus-queue design is what makes the sync generator safe here: the thread pool pulls from a queue, and the OTel span and trace-id context live entirely on the pipeline's own worker thread.

## 5. What the numbers said

`uv run pytest -q`: **513 passed** (+5 in `tests/test_api.py`, RED first on the missing module).

Real server (`uvicorn aegis.api.main:app`, real Groq/Qdrant/cross-encoder, throwaway cache files), real HTTP requests. Small samples, one run each; treat as observations.

| Request | Observed |
| --- | --- |
| Fresh dev queries, generated answers (n=3) | first token over HTTP 965 / 1,235 / 1,824 ms |
| Fresh dev queries that escalated (n=3) | `escalation` then `done`, 777–2,886 ms total |
| Repeat of an already-answered query | exact cache hit, 7 ms total (curl) |
| `{"query":"   "}` | HTTP 422 |

Pre-first-token stages in the server log for the generated answers: routing 216–335 ms, retrieval 24–52 ms, confidence 338–929 ms.

**Client disconnect (n=1).** I closed the connection after the first token. The server stayed healthy, and the pipeline still ran to completion and logged `request_end`, with the answer written to the cache. Abandoned requests cost full LLM spend; they are not cancelled.

## 6. What surprised me / what to watch for

- **Slow outliers are still there.** One fresh query had a **10,023 ms confidence stage** (first token over HTTP 10.7 s), another 2,550 ms, versus 338–929 ms for the rest. This is the same unexplained confidence-stage variance from Phase 13; I did not bisect it here. It is the largest threat to the p95 < 3 s criterion in the real server too.
- **One request spent 28.7 s in `validation`** (total 31 s) while its tokens had already arrived at 1.8 s. Validation only costs time when tier-2 entailment fires an LLM call, so a stalled provider call is my likely explanation, but I did not check the 429/retry logs. Streaming hides this from the answer text but not from the citations, which arrive only after validation.
- **Cancellation does not exist.** A disconnect does not stop the worker thread (see §5). Fine at portfolio scale; a real deployment needs cancellation or a concurrency cap (rate limiting is a later unit).
- **The sample is thin.** Nothing here is a latency measurement worth a percentile; it shows the endpoint streams and which stalls occur. The p95 claim stays open until a real load run.
- **Not yet verified:** a browser consuming this stream, concurrent requests, and a provider failure mid-stream through HTTP.

## 7. Where this fits and what's next

Unit 3 wraps `observability/audit.py` as `GET /trace/{id}` and adds feedback storage. Then conversation memory (Phase 8 rewriting), then the static UI, which consumes this exact event contract. Rate limiting and the real escalation handler remain after that.
