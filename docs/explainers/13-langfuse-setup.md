# 13 — Langfuse Setup

**Phase:** 13 · **Date:** 2026-09-26 · **Files:** `src/aegis/config.py`, `src/aegis/observability/tracing.py`, `experiments/13a_langfuse_smoke_test.py`, `.env.example`, `pyproject.toml`
**Topics from the course:** Logging, monitoring & observability (Module 5, #31); Latency vs. response quality (Module 5, #35)

## 1. What I built

`LangfuseSettings` (`config.py`) plus a cached singleton client getter (`aegis.observability.tracing.get_tracer()`) wrapping the official `langfuse` v4 Python SDK. Input: nothing yet from the real pipeline — this unit is pure plumbing. Output: one process-wide `Langfuse` client, configured from `.env`, that later units attach real per-stage spans to. A connectivity smoke test (`experiments/13a_langfuse_smoke_test.py`) builds a fake two-span trace (retriever + generation, the same shape real instrumentation will use), flushes it, and reads it back from the Langfuse Cloud API to prove the round trip actually works before any real stage is wired in.

## 2. Why this approach

The build list says "Langfuse (self-hosted)." I checked that literally rather than assuming it's a quick `docker-compose` add, per this session's own instruction. Langfuse v3's official self-host stack is `langfuse-web` + `langfuse-worker` + Postgres + ClickHouse + Redis + MinIO — six containers — and Langfuse's own docs recommend 4 cores/16 GiB *for that stack alone*. Measured on this machine: Docker Desktop is allocated 7.75 GiB total, of which Qdrant is already using ~750 MiB, and the host's full 16 GiB also has to leave headroom for MPS embedding inference running outside Docker (Phase 3 batches, any future re-embed). That's a real, load-bearing capacity conflict, not a hypothetical one — self-hosting the full stack risked swapping or OOM the next time an embedding job runs alongside it.

Langfuse Cloud's free tier is sized for exactly this project's real scale: 50k units/month (a unit = one trace, span, or score), no card required, 30-day retention. Phase 13's own build list traces 7 stages per query (routing, retrieval, rerank, confidence, generation, validation, cache status) — even at a few hundred real dev/eval queries across the rest of this project, that's nowhere near 50k/month.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ------------------ |
| Full self-hosted Langfuse v3 stack (official docker-compose: web, worker, Postgres, ClickHouse, Redis, MinIO) | Matches plan.md's literal "self-hosted" wording; no third-party service holds query/answer data | Official min spec (4 cores/16 GiB) for the stack alone exceeds what's currently allocated to Docker on this 16 GB host (7.75 GiB), which already shares that budget with Qdrant and, outside Docker entirely, MPS embedding inference. Real risk: resource contention exactly when an embedding batch job runs. |
| Self-hosted Langfuse v2 (Postgres-only, no ClickHouse/Redis/MinIO) | Still genuinely local/self-hosted, much lighter footprint (~1-2 extra containers) | v2 is the deprecated line — Langfuse's own current docs point to v3, and running a version they're phasing out risks losing patches/support mid-project for a component whose whole job is being a reliable audit trail. |
| Langfuse Cloud (free tier) — **chosen** | Zero local infra, same SDK/instrumentation code path as either self-hosted option, sized correctly against real measured traffic | Deviates from plan.md's literal "self-hosted" text — disclosed explicitly (Decision Log, 2026-09-26) rather than silently substituted, and traded off with the user directly (`AskUserQuestion`) rather than picked unilaterally, since it also means trace data (queries, retrieved chunks, generated answers) leaves the machine. |

## 4. How it works

`LangfuseSettings` follows the project's existing per-group `BaseSettings` pattern (`env_prefix="LANGFUSE_"`), read once into the shared `settings` singleton:

```python
class LangfuseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LANGFUSE_", env_file=".env", extra="ignore")
    public_key: str = ""
    secret_key: str = ""
    base_url: str = "https://cloud.langfuse.com"
    environment: str = "dev"
```

`get_tracer()` is `lru_cache`d to exactly one instance per process — the SDK owns a background batching/flush thread and its own OpenTelemetry `TracerProvider`; constructing a second client would mean two independent export threads writing to the same project, for no benefit:

```python
@lru_cache(maxsize=1)
def get_tracer() -> Langfuse:
    return Langfuse(
        public_key=settings.langfuse.public_key,
        secret_key=settings.langfuse.secret_key,
        host=settings.langfuse.host,
        environment=settings.langfuse.environment,
    )
```

The smoke test exercises the real shape later units need — a root span with nested typed observations, each carrying real `input`/`output` payloads:

```python
trace_id = client.create_trace_id()
with client.start_as_current_observation(
    trace_context={"trace_id": trace_id}, name="phase13-smoke-test", as_type="span",
    input={"query": "does the qdrant client support hybrid search?"},
) as root:
    with client.start_as_current_observation(name="retrieval", as_type="retriever", input={"top_k": 10}) as s:
        s.update(output={"chunks_returned": 10, "top_score": 0.83})
    with client.start_as_current_observation(name="generation", as_type="generation", model="qwen/qwen3.8-27b") as g:
        g.update(output={"answer": "yes, via named vectors"}, usage_details={"output": 12})
client.flush()
```

Verification doesn't stop at "the SDK didn't raise" — it reads the trace back from Langfuse's own API (`client.api.trace.get(trace_id)`) and asserts the observation count matches, so a silently-dropped span (e.g. a flush timing issue) would fail loudly instead of looking like success.

## 5. What the numbers said

**Verified live against a real Langfuse Cloud project, 2026-09-27**, after the user signed up and added real keys. First run failed with a real `401 Unauthorized` — root-caused and fixed (§6), not retried blindly. Second run failed with a real `410` from the readback step — also root-caused and fixed (§6). Third run, real output:

```
auth_check() passed -- credentials are valid.
trace_id: 1350343caaf045fcb70958e6e805b1f6
trace_url: https://us.cloud.langfuse.com/project/cmujtds4508cbad0dj9jedaxo/traces/1350343caaf045fcb70958e6e805b1f6
fetched observation names: ['generation', 'phase13-smoke-test', 'retrieval']
Real trace + 2 nested observations confirmed round-trip via the Langfuse Cloud API.
```

`uv run pytest -q`: 455/455 passed throughout (unchanged — this unit adds configuration/wiring only, no new logic that needs a unit test; the real verification is this live integration run).

## 6. What surprised me / what to watch for

The self-host footprint was bigger than "add one more service to docker-compose" — Langfuse v3 restructured around ClickHouse for trace storage specifically (moved off Postgres for that data), which is what pulls in Redis and MinIO too. Worth remembering if the project ever moves to a machine where self-hosting is actually viable: it's a 6-container commitment, not a 1-container one.

The v4 Python SDK is OpenTelemetry-based and dropped the older `@observe` decorator / `trace()`-call style this project's earlier exposure to Langfuse docs might suggest — the current idiom is explicit `start_as_current_observation` context managers per span, with an `as_type` literal (`span`/`generation`/`retriever`/`guardrail`/etc.) that map cleanly onto this phase's actual stages (retrieval → `retriever`, generation → `generation`, confidence/validation → likely `evaluator`/`guardrail`). That mapping will matter for Unit 2's real per-stage instrumentation — picking the right `as_type` per stage now, rather than defaulting everything to a plain `span`, is what makes the Langfuse UI's own filtering/timeline views useful later instead of just a flat list of spans.

**Two real bugs found by actually running this against a live account, not caught by anything short of that:**

1. **Langfuse Cloud is multi-region (US/EU), and my initial `LangfuseSettings` didn't account for it.** I named the config field `host`, so pydantic-settings only ever read `LANGFUSE_HOST` — unset, so it silently fell back to my hardcoded EU default (`cloud.langfuse.com`) against a real US-region project, producing a real `401 Unauthorized` whose own error body said "confirm you've configured the correct host." The user's `.env` already had the *correct* value under `LANGFUSE_BASE_URL` (Langfuse's own dashboard snippet name) — my field just didn't match it. Fixed by renaming the field to `base_url`, which now reads `LANGFUSE_BASE_URL` for free. Lesson: match config field names to what the vendor's own setup docs/dashboard actually emit, not an assumed canonical name.

2. **This Langfuse org (created 2026-09-27, after the platform's own 2026-09-16 cutover) has no legacy API at all.** `client.api.trace.get()` — the single-trace lookup I'd planned to verify the round trip with — returned a real `410`, `LEGACY_API_UNAVAILABLE_FOR_NEW_ORGANIZATION`, pointing at `GET /api/public/v2/observations` instead. Rewrote the readback to `client.api.observations.get_many(trace_id=..., from_start_time=..., to_start_time=...)`. Separately, a first attempt at that with only a 3-second wait after `flush()` returned zero observations — a real async-ingestion race, not a query bug (confirmed directly: an isolated single-span debug trace also returned 0 results at 3s and 1 at 8s). Settled on an 8-second wait before the readback call. Both are now the kind of sharp edge worth remembering for Unit 2's real instrumentation: any code that reads traces back programmatically (not just views them in the UI) needs to tolerate ingestion lag, and needs to target the v2 API from the start rather than whatever an LLM's or my own training-time knowledge of the SDK assumes is current.

## 7. Where this fits and what's next

Connectivity is now real, verified, and round-tripped through the current (non-legacy) Langfuse API — Unit 1 is done. This unblocks Unit 2 (structured logging + a trace ID threaded through the real pipeline) and the per-stage instrumentation of routing, retrieval, rerank, confidence, generation, validation, and cache status that Phase 13's build item #1 actually asks for. Next: wire `get_tracer()` into the real pipeline entry point (not a throwaway fake trace), choosing the right `as_type` per stage per SS6's mapping, and thread one real trace ID through every stage of one real end-to-end query.
