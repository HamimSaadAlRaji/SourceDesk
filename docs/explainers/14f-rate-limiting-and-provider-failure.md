# 14f — Rate limiting and provider-failure handling

**Phase:** 14 · **Date:** 2026-10-17 · **Files:** `src/aegis/api/ratelimit.py`, `src/aegis/api/app.py`, `src/aegis/generation/llm/fallback.py`, `src/aegis/router/classifier.py`, `src/aegis/guardrails/citations.py`, `src/aegis/pipeline.py`, `src/aegis/events.py`, `src/aegis/api/static/app.js`, `src/aegis/config.py` (`ApiSettings`), `tests/test_rate_limit.py`, `tests/test_fallback_llm.py`, `tests/test_pipeline_stream.py`, `tests/test_api.py`, `tests/test_structured_logging.py`, `tests/test_ui_static.py`, `experiments/14c_provider_failure_real.py`
**Topics from the course:** What makes production challenging (Module 5, #29); Guardrails / escalation (Module 5); Latency vs. response quality (Module 5, #35)

## 1. What I built

Plan.md Phase 14 step 6. Three pieces:

1. **A per-client token bucket on `/chat`.** Over the limit, the client gets HTTP 429 with `Retry-After` and the pipeline is never reached.
2. **LLM outage → escalation with its own cause.** When an LLM stage fails on every provider before any text has been shown, the request goes to a human with `cause="provider_unavailable"`. This applies to the router and to generation. It is no longer a dead-end `error`, and the router case is no longer disguised as a billing escalation.
3. **No silent failures.** Every failed provider attempt, every failover, and every tier-2 entailment API error is now a structured log event.

## 2. Why this approach

**The outage path is a North Star problem, not a plumbing problem.** Before this unit, an LLM outage had three different outcomes, and two of them were misleading:

| Stage that failed | Before | What it claimed |
| --- | --- | --- |
| Router (Groq) | escalation, `route=billing_account_legal` | "this is a billing question": false. The trace, the audit trail and the UI ("this kind of request is always handed to a person") all repeated it. In an eval run, an outage was scored as a billing *prediction*. |
| Generation (every provider) | terminal `error`, "could not be completed" | nothing false, but the user is at a dead end after the system had already decided the question was answerable |
| Tier-2 entailment (Groq) | `False`, no log line | indistinguishable from "every claim is unsupported" |
| Fallback (Groq → Gemini) | silent | the cold browser run in 14e was answered by Gemini, and nothing said why Groq was skipped |

For a support desk, a request it can't answer right now belongs with a person. That is the same reasoning as low-confidence abstention. The user picked "escalate, with its own reason" over "keep the error and only fix the labels" (`AskUserQuestion`, 2026-10-17).

**Escalating is only honest before any text is shown.** `AllProvidersFailedError` is raised only when no provider produced a first token (`fallback.py`: providers are never spliced mid-stream). So by construction nothing has reached the user, and a handoff doesn't contradict anything on screen. A failure after the first token is a different exception and stays a terminal `error`: the UI says the answer above is incomplete. A test pins that split (`test_mid_stream_failure_stays_a_terminal_error`).

**`cause` is a closed set on the event, separate from `reason`.** The UI used to infer *why* from `after_retrieval`. That gives two buckets, and an outage after retrieval would have read "wasn't confident enough", which is false again. `EscalationCause = policy | low_confidence | provider_unavailable` lets the UI explain each case without parsing free text. A test fails if a cause is added without UI text (`test_ui_explains_every_escalation_cause`).

**The rate limit protects quota, not CPU.** One fresh request can make up to 3 Groq calls before any answer (rewrite, router, generation), plus one per claim for tier-2 entailment, all from a single shared free-tier quota. Phase 13 already measured what an exhausted per-minute cap looks like from inside: routing went from 251 ms to 14.4 s (design record 13c). A script hammering `/chat` would turn every other user's requests into that, then into outage escalations.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| `slowapi` for rate limiting | Standard FastAPI limiter; Redis backend available later | One uvicorn process, no Redis, so the shared-store feature buys nothing. It adds a dependency (CLAUDE.md §2 needs a justification) and its decorator needs a `Request` parameter on each route. The mechanism is one formula and ~40 lines, TDD'd with an injected clock. User agreed (`AskUserQuestion`). |
| Keep the all-providers-failed `error`, only fix labels and logging | Smallest change; an error is not a wrong answer | Leaves the user at a dead end for a question the system had already judged answerable. A human handoff is the support-desk outcome. User chose escalation. |
| Router outage stays folded into the billing fail-safe, just with a better `reason` string | No new exception type; `classify()` keeps its `Route` return | `route=billing_account_legal` would still be on the event, the result, and the router eval's predictions. Raising `RouterUnavailableError` means an eval run stops on an outage instead of silently scoring it. Unparseable *model output* still fails safe to billing: that's a model behavior, not an outage. |
| Retry the same provider with backoff before failing over | Transient 429s/503s often clear in seconds | The latency budget is already missed (p95 < 3 s unmet, Phase 13). A retry wait adds to the very requests that are already slow, and the 13c data suggests Groq's SDK may already retry internally on 429 (not confirmed at the HTTP level). Not added; noted in §6. |
| Key the limiter on `X-Forwarded-For` | Correct behind a reverse proxy | The client controls the header, so anyone could reset their own bucket by sending a new value. There is no proxy in this deployment. Keyed on the socket peer; trusted-proxy handling goes to Phase 15. |

## 4. How it works

**Token bucket** (`api/ratelimit.py`), per client key:

```
tokens(t) = min(capacity, tokens(t0) + (t - t0) * rate)      rate = refill_per_minute / 60   [tokens/s]
allowed   = tokens(t) >= 1          → tokens -= 1
refused   → retry_after = (1 - tokens(t)) / rate   seconds   (a refused request costs nothing)
```

Defaults (`ApiSettings`): `chat_rate_capacity=5`, `chat_rate_per_minute=6.0` (one token every 10 s), `rate_limit_max_clients=10000`. These are guard rails, not tuned values: there is no real traffic to tune against. The comment in `config.py` gives the sizing argument from per-request call counts. Buckets live in an `OrderedDict` behind a lock, ordered by last access. Past `max_clients` the least recently seen client is evicted and comes back with a full bucket: at most one extra burst, never a lockout. The check runs after FastAPI validates the body, so a 422 costs no token (tested). Cache hits do cost a token (one rule, no cache lookup before the limiter).

**Outage → escalation** (`pipeline.py`):

```python
try:
    with _timed(timings, "routing"):
        route = self._route(query)
except RouterUnavailableError:              # raised by the default classify_fn on groq.APIError
    return self._escalate(query, status, emit, route=None, cause="provider_unavailable",
                          reason=_PROVIDER_UNAVAILABLE_REASON.format(stage="routing"))
...
try:
    draft, context = self._generate_draft(query, retrieved, timings, emit)
except AllProvidersFailedError:             # only raised when no provider yielded a first token
    return self._escalate(query, status, emit, route=route, cause="provider_unavailable",
                          reason=_PROVIDER_UNAVAILABLE_REASON.format(stage="generation"),
                          candidates=candidates, confidence=confidence)   # evidence a human picks up from
```

`_escalate` never writes to the caches, so an outage escalation can't be served later as a cached answer (tested with a cache whose `set` raises). The generation block moved into `_generate_draft` unchanged, so the try covers exactly the LLM call and its span.

**Logged events** (JSON, trace id attached): `llm.provider_failed` (provider, attempt, error_type, status_code, truncated detail, elapsed_ms), `llm.failover` (provider that answered, providers skipped), `router.api_error` (existing), `citation.entailment_api_error`, `api.rate_limited` (client, retry_after_s). `status_code` reads `status_code` (Groq SDK) or `code` (google-genai). The first real run logged Gemini's as `None` until this was fixed.

## 5. What the numbers said

Unit tests: `uv run pytest -q` → **563 passed** (+21 this unit). This includes the Qdrant delta-indexing integration test, run with Qdrant up.

**Real-stack failure injection** (`experiments/14c_provider_failure_real.py`, raw `docs/results/provider-failure-real-raw.json`). Each case is a real SDK's real auth error, forced with an invalid key, n=1 each:

| Case | What happened | Wall |
| --- | --- | --- |
| A. Failover (Gemini invalid → Groq real) | answered by `groq`; `llm.provider_failed` gemini `ClientError` 400 (559 ms to fail), `llm.failover` skipped `["gemini"]` | 1,005 ms (first run) |
| B. Router outage (Groq invalid) | `escalation` (`cause=provider_unavailable`, `route=None`, `after_retrieval=false`), `done`; no retrieval; `router.api_error` `AuthenticationError` 401 | 162 ms |
| C. Generation outage (both invalid) | `escalation` (`cause=provider_unavailable`, `route=docs_howto`, `after_retrieval=true`), `done`; Groq 401 in 153 ms, Gemini 400 in 291 ms; **not cached** afterwards | 1,000 ms |

Time to the escalation in C was 1.0 s, of which generation (both failed attempts) was 452 ms. Failing fast on auth errors is cheap. A 429 with SDK-internal retries is a different shape and was not measured (§6).

**Real HTTP rate limit** (real server, defaults, same question repeated):

- Requests 1–6 → 200. Request 1 was a real miss (11.1 s; generation was not broken down), requests 2–6 were exact-cache hits (~5 ms each).
- Requests 7–9, sent back to back → **429**, `Retry-After: 1`, ~2 ms each, each logged as `api.rate_limited`.
- `/health` 200 throughout.

Six requests got through, not five. That matches the formula: the 11.1 s first request refilled 11.1 × 0.1 ≈ 1.1 tokens, capped at 5, before request 2 arrived. Behavior was not tested with real concurrent clients.

## 6. What surprised me / what to watch for

- **The test caught a real bug in my own wiring.** `limiter or TokenBucketLimiter(...)`: the limiter defines `__len__`, so a freshly built (empty) one is falsy, and the injected limiter was silently replaced by the default. The 429 tests failed for that reason. Fixed with `is not None`, with a comment.
- **Slow requests refill the bucket.** At 6/min a client is effectively limited by how long answers take, plus the burst. That's fine for its purpose (quota protection), but "5 per burst" is not a hard per-minute cap.
- **The router eval now stops on an outage.** `experiments/08a_router_eval.py` would raise `RouterUnavailableError` mid-run instead of recording billing predictions. That's intended (an outage is not a prediction), but the script has no checkpointing.
- **Not handled: a 429 slow path.** Groq/Gemini SDKs may retry 429s internally before raising, which would make the fallback slow, not fast. Design record 13c's 14.4 s routing hints at this, unconfirmed. Worth checking SDK retry settings (`max_retries`) when the latency budget is revisited.
- **Rewriter outage is still a silent downgrade for the user.** It falls back to the raw follow-up (logged since 14d), which then usually low-confidence-escalates. That's honest but not labeled as an outage. Left as is.
- **Tier-2 entailment outage still serves the answer** with every citation struck through (fail-closed, now logged). Under the "stream live, validate after" decision (design record 14) this is consistent, but a user sees an answer with no validated sources during an outage. Candidate for a follow-up: escalate when validation failed for infrastructure reasons rather than evidence reasons.
- Pre-existing, fixed in passing: `app.py` and `rewrite.py` used `get_logger(__name__)`, which produced logger names like `aegis.aegis.api.app`.

## 7. Where this fits and what's next

Closes plan.md Phase 14 build item 6. The fallback logs are what were missing to explain 14e's cold run (Gemini answered, 17.9 s): the next time it happens, the log will say which provider failed, with what, and how fast. Remaining Phase 14 work: the review queue and golden-candidate export (build item 5), the semantic-cache collision and cold `cache_lookup` investigations, and the regime-C multi-turn eval that the open exit criterion depends on. Rate-limit keying behind a proxy, and `/trace`'s unauthenticated 20 s worker hold, go to Phase 15.
