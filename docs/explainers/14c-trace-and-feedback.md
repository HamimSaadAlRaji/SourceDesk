# 14c — `/trace/{id}` and `/feedback`

**Phase:** 14 · **Date:** 2026-10-01 · **Files:** `src/aegis/api/app.py`, `src/aegis/api/store.py`, `src/aegis/api/audit_service.py`, `src/aegis/config.py` (`ApiSettings`), `tests/test_api.py`, `tests/test_interaction_store.py`
**Topics from the course:** Logging, monitoring & observability (Module 5, #31); Feedback loops / continuous improvement (Module 5)

## 1. What I built

`GET /trace/{trace_id}` returns the Phase 13 audit trail plus the resolved source chunks for one request. `POST /feedback` records a thumbs up/down (+ optional comment) against a `trace_id`. To make the second one useful, `/chat` now records each completed interaction (query, the answer text the user saw, escalated?, cache status) in a sqlite store keyed by `trace_id`.

## 2. Why this approach

- **The server records the interaction; the client only sends `trace_id`, rating, comment.** The feedback → review queue → golden-set-candidate loop (plan.md Phase 14 step 5) needs the query and the answer. If they came from the client, anyone could put arbitrary "answers" in the review queue. If they stayed in Langfuse, the loop would expire with its 30-day retention. A vote for a `trace_id` the server never issued gets 404.
- **Only completed streams are recorded** (a `done` event). A stream that ended in `error` has no issued answer worth reviewing; a test pins that feedback on it is a 404.
- **A second vote replaces the first** (primary key = `trace_id`): users change their mind, and one row per request keeps the review queue countable.
- **`/trace` is a thin wrapper** over `fetch_observations` → `build_audit_trail` → `resolve_evidence`, which already exist and were verified live in Phase 13. The path parameter must match 32 hex chars, so arbitrary strings never reach Langfuse.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Send query + answer back in the `/feedback` body | Stateless server | Client-controlled content in the review queue; trivially poisoned. |
| Store feedback only as a Langfuse score on the trace | Reuses the tracing backend, no new store | Ties the golden-candidate loop to 30-day retention and to ingestion lag; the review queue needs a local, queryable join. |
| Postgres / SQLAlchemy | Concurrency, migrations | Single-process portfolio server; the project already uses sqlite for caches; stdlib `sqlite3` is enough and adds no dependency. |
| Make `/trace` wait until the trace is fully ingested, unbounded | Always returns | Ingestion lag has been 8–14 s+; an unbounded wait ties up a worker. Capped at `ApiSettings.trace_max_wait_s` = 20 s, then 404 "not found (or not yet ingested)". |

## 4. How it works

`InteractionStore` (`api/store.py`): two tables, `interactions(trace_id PK, query_text, answer_text, escalated, cache_status, created_at)` and `feedback(trace_id PK → interactions, rating CHECK in ('up','down'), comment, created_at)`. One connection per call (requests run on a thread pool), a lock around writes. In `/chat`'s frame generator: tokens (or the escalation message) are accumulated, and the `done` event triggers `record_interaction` before the `done` frame is yielded, so a client that sees `done` can immediately vote.

## 5. What the numbers said

`uv run pytest -q`: **524 passed** (+11: 5 store, 6 API; RED first on the missing modules).

Real server, real Langfuse Cloud, real pipeline, one request (n=1):
- `/chat` → `/feedback {"rating":"down","comment":"test vote"}` → `200 recorded`. A syntactically valid but never-issued trace id → `404`.
- `/trace/{id}` on that fresh trace → `200`, **24.8 s wall time**. It rebuilt: miss, route `docs_howto`, not escalated, provider groq, all six expected stages, **no warnings, no inconsistencies**, and 10 evidence items, all `status: ok`.
- After killing and restarting the server the vote was still in the store with the right query and the answer text.

## 6. What surprised me / what to watch for

- **`/trace` on a fresh trace took 24.8 s**, above the 20 s cap I set. The cap is checked between polls and each poll is a network call plus a 5 s sleep, so it is a soft limit. It is a debugging endpoint, not a user path, but a UI that calls it needs a spinner, and an old trace still pays at least one poll interval. I did not optimise it.
- **`/trace` is unauthenticated and returns the user's query text and source chunks.** Fine for a local portfolio server; it is an information-disclosure surface for Phase 15 (knowledge-base leakage). Same for the `trace_id`, which is the only capability needed to vote on a request.
- **No rate limit on `/feedback` yet** (a later Phase 14 unit), so votes can be spammed.
- **The review queue and golden-set-candidate export are not built.** This unit gives the store they read from (`list_feedback()` joins vote + query + answer); the queue itself is next.
- **A test-isolation bug I introduced and caught:** the first test run created the real `data/feedback/interactions.sqlite` because older tests used the default store path. Fixed with an autouse fixture, and the stray file deleted.
- **Not verified:** a browser calling these, concurrent writers.

## 7. Where this fits and what's next

Next: conversation memory (Phase 8 history-aware rewriting wired into `/chat`), then the static UI (consumes `/chat`, shows citations from `citation` events, escalation as its own state, and thumbs buttons calling `/feedback`), then the review queue/golden-candidate export and rate limiting.
