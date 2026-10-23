# 15c — Access Control: Public Chat, Owned Traces, Operator-Only Audit

**Phase:** 15 · **Date:** 2026-10-18 · **Files:** `src/aegis/api/auth.py`, `src/aegis/api/app.py`, `src/aegis/api/store.py`, `src/aegis/eval/review_queue.py`, `src/aegis/factory.py` (`build_pipelines`), `src/aegis/config.py` (`ApiKeyGrant`, `ApiSettings`), `tests/test_auth.py`, `tests/test_interaction_store.py`, `tests/test_api.py`, `tests/security/test_attack_http_surface.py`
**Topics from the course:** Security — knowledge-base leakage, LLM data leakage (Module 5); What makes production challenging (Module 5, #29)

## 1. What I built

Three kinds of caller:
- **Anyone** can chat. Each browser gets a server-issued ownership cookie, and `/feedback` only accepts votes on traces that browser created.
- **API keys** (configured by hash) map a named principal to a partition, and optionally to `/trace` access.
- **`/trace` is operator-only**, capped at 2 concurrent calls, and an upstream failure is a 502.

Also: `/feedback` has its own rate limit, and the review tool escapes everything it prints.

## 2. Why this approach

**The constraint that decided it: Phase 18 ships a public URL, and a public web page can't keep a secret.** Requiring a key on `/chat` (the obvious reading of "unauthenticated → authenticate") would put that key in `app.js`, where every visitor can read it. That's a lock with the key taped to the door. The user chose (2026-10-18, after a plain-language explanation of the three options): **anonymous chat + per-browser ownership tokens, with `/trace` behind API keys.**

**What each attack needed, rather than "add auth":**

| Attack (record 15) | What was actually wrong | Fix |
|---|---|---|
| A5b: read another user's trace | A trace ID was the only credential, and IDs travel (logs, UI, screenshots) | `/trace` needs an operator key. Anonymous callers never reach it, not even for their own trace (the UI doesn't use it). |
| A5c: overwrite another user's vote | Same: knowing the ID was enough | The interaction records its owner. A vote is accepted only from that owner. |
| A5d: 40 slow `/trace` calls starve the pool | A sync endpoint holding a thread for up to 20 s with no cap | Operator-only *and* a non-blocking semaphore (2), then an immediate 503. |
| A5e: feedback flood | No limit | Token bucket per socket peer, 10 burst / 20 per min. |
| A5g: terminal escapes in the reviewer's terminal | Anonymous text printed raw | `_safe()`: every non-printable character is shown as an escape. |

**An ownership token is not a login, and must not be a rate-limit key.** Anyone can mint a fresh token by dropping their cookie. That's fine for ownership (a new token owns nothing), but keying the limiter on it would hand out unlimited buckets. The limiter stays on the socket peer (14f).

**A bad key is a 401, never a fallback to anonymous.** Falling back would turn a mistyped operator key into silent anonymous access, and the operator would debug the wrong problem.

**Why `/feedback` returns 404 for someone else's trace, not 403.** A 403 would confirm the trace ID exists. 404 gives the same answer for "never issued" and "not yours."

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
|---|---|---|
| API key required on every endpoint | Closes A5a literally; simplest story | The public demo would need a shared key in the page source. It looks authenticated without being authenticated. |
| Signed (HMAC) session tokens | Server could reject tokens it didn't mint | A forged token owns nothing, so signing protects nothing here. It would add a server secret to manage and rotate for no gain. Random 256-bit tokens, stored only as a hash. |
| Token returned in the JSON/SSE body, kept in `localStorage` | No cookie handling | Readable by any script on the page, so one XSS steals ownership. An `HttpOnly` cookie isn't readable by page JS, and `SameSite=Strict` stops other sites from sending it. The UI needed **no change**: same-origin `fetch` sends cookies by default. |
| `/trace` open to the trace's own owner | Users could audit their own answers | The UI doesn't call it. The payload is a debug trail (every stage, all chunk text), and each call holds a worker for up to 20 s. The user chose operator-only. |
| Key the rate limiter on the principal | Fairer behind a shared NAT | Session tokens are free to mint, so this would be an unlimited-bucket bypass. Revisit only for keyed principals, if a partner integration needs it. |
| `slowapi` / an auth library (`fastapi-users`, OAuth) | Standard, maintained | There are no user accounts, passwords or OAuth providers here: the surface is a random cookie and a hash lookup (~60 lines, TDD'd). CLAUDE.md §2 requires a dependency to buy something, and these would buy account management this design deliberately doesn't have. |

## 4. How it works

**Resolution** (`auth.resolve_principal`, pure, tested on its own):

```
Authorization: Bearer <k>   → sha256(k) compared (hmac.compare_digest) against ApiSettings.api_keys
                               match → Principal(kind="key", owner="key:<name>", partition, can_trace)
                               no match / wrong scheme → Unauthorized → 401  (never anonymous)
no Authorization, cookie matches ^[A-Za-z0-9_-]{43}$
                             → Principal(kind="session", owner="session:"+sha256(token)[:32], anonymous_partition)
otherwise                    → new token (secrets.token_urlsafe(32), 256 bits), set as cookie on /chat
```

The store records `owner`, never the token. A leaked `interactions.sqlite` hands out no usable sessions (tested: `token not in owner`).

**Routing a request to its partition:** `create_app` holds `pipelines: {partition: pipeline}`, and `/chat` uses `pipelines[principal.partition]`. `factory.build_pipelines` builds one per partition the config references, sharing bge-large, the cross-encoder, the LLM clients and the caches (safe because cache keys carry the partition, 15b). If a key names a partition with no pipeline, **startup fails** (tested), instead of the first keyed request getting a 500.

**Keys are configured by hash:** `uv run python -m aegis.api.auth new-key --name operator --partition full --trace` prints the key once, plus the `API_API_KEYS` JSON entry. That entry holds the sha256 and the grant only.

**Store migration:** `interactions` gains `owner TEXT` via `ALTER TABLE` if it's missing. Rows from before 15c get NULL, which matches no caller, so old interactions can no longer be voted on by anyone (tested against a Phase 14-schema file).

## 5. What the numbers said

`uv run pytest -q`: **643 passed, 2 xfailed** (was 612 + 8). That's +18 `test_auth.py`, +3 store ownership tests, +8 HTTP-surface tests, and 6 attacks flipped from xfail to pass. The 2 remaining xfails are the injection attacks A3/A4 (15e).

**Before → after (deterministic, `tests/security/test_attack_http_surface.py`):**

| Attack | Before (15) | After (15c) |
|---|---|---|
| A5a anonymous access | All 3 endpoints open | `/chat` open **by decision**; `/trace` 401 anonymous, 403 for a key without trace rights; any invalid key → 401 on all 3 |
| A5b read another's trace | 200 with query + chunk text | 401; the asker's query is not in the response |
| A5c overwrite another's vote | Accepted | 404; the owner's vote is unchanged |
| A5d `/trace` starvation | `/health` no answer in 2 s | Attacked **with a valid operator key**: `/health` answers; 38/40 calls get 503 |
| A5e feedback flood | 200/200 | 429 after the burst |
| A5f XFF spoof | Defended | Still defended |
| A5g terminal escapes | Raw ANSI escape reached `print` | ANSI, BEL, bidi override (U+202E) and a newline faking a queue line are all shown escaped |

**Real server** (`uvicorn`, operator key minted with the CLI, throwaway DBs, n=1 each). The `/chat` request was a billing question, which hard-escalates after the router, so the only LLM spend was one router call (tokens not measured).

- `/chat` anonymous → `escalation` (`cause=policy`) + `done`; `Set-Cookie: aegis_session=…; HttpOnly; Max-Age=2592000; Path=/; SameSite=strict`.
- Vote on Alice's trace with no cookie → **404**. With a different fresh cookie → **404**. With Alice's cookie → **200**.
- `/trace` anonymous → **401**. Wrong key → **401**.
- 30 back-to-back votes on one cookie → **9 × 200, then 21 × 429**. That's 9, not 10: the first vote in step 4 had already spent a token.
- **40 concurrent `/trace` with the operator key → 2 ran, 38 × 503 immediately; `/health` during the burst 200 in 3.6 ms.**
- `/trace` with the key → **502 "trace backend unavailable"** after ~6 s. Explained in §6.

## 6. What surprised me / what to watch for

- **The first real `/trace` with a valid key was a 500.** I stopped to find out why. The traceback is `httpx.ReadTimeout` inside Langfuse's `observations.get_many`. Calling Langfuse directly, outside the server, gave a ReadTimeout at 5.8 s, then a success at 4.9 s (just under the client's timeout), then an `ApiError` on `trace.list`. So Langfuse was slow and flaky at the time, independent of these changes. The real defect was that an upstream failure surfaced as an unlabeled 500. It now becomes `api.trace_backend_error` plus a 502 (tested). **I never got a successful real `/trace` 200 in this session**, so the happy path is verified only with the fake audit service.
- **The starvation test needed rewriting to stay meaningful.** Once anonymous callers get a 401, the old attack (anonymous `/trace` × 40) passes trivially, because nobody reaches the slow path. It now attacks with a valid operator key, the leaked-key case, so it tests the cap and not the auth.
- **A5a's original assertion was replaced by the decided policy, not satisfied.** The 15 test asserted "all three endpoints 401 for anonymous callers." That was my assumption before the user decided anonymous chat is intended. The test now asserts the decided policy. Recorded here so it doesn't read as a defense passing its original test.
- **TDD slip:** I wrote `test_auth.py` before `auth.py` but didn't run it red before implementing. The store ownership tests were run red first (3 failing, TypeError).
- **A regex edit to the test files also "fixed" the test that deliberately omits `owner`.** I caught it from that test failing, then restored it. Same lesson as the 15 starvation probe: check that a security test can still fail after any bulk edit.
- **Not verified in a browser.** The cookie flow is verified with `curl` and TestClient. The UI's same-origin `fetch` should send it by default, but that was read from the spec, not observed.
- **`session_cookie_secure=False` is the default** so `http://localhost` works. Phase 18 (HTTPS) must set it to True, otherwise the cookie travels in clear over any plain-HTTP hop.
- **Rate limits are still per IP.** Behind a reverse proxy every client shares one bucket (14f's open item). Trusted-proxy handling belongs with deployment (Phase 18), where the proxy actually exists.

## 7. Where this fits and what's next

Every A5 attack is now closed or decided. The remaining xfails are A3 (forged context blocks) and A4 (the validator passing an injected commitment), which is 15e. Before that comes **15d (input boundary)**: PII redaction before the store, the feedback candidates and Langfuse, and the flag-only injection regex with its false-positive rate *and* a measured rephrasing-evasion rate.
