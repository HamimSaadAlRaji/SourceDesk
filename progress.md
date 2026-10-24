# Aegis — Progress Log

**Last updated:** 2026-10-22 (Phase 16 unit 16a)
**Current phase:** Phase 16 — CI Evaluation Gate (🟡 in progress — 16a `test.yml` ✅, 16b-c ✅, 16d `eval.yml` retrieval gate ✅ green on the runner 2026-10-24; LLM-backed checks, p95 gate, PR comment, deliberate-regression demo remain). Phase 15 — Security Hardening (✅ complete 2026-10-22 — units 15, 15b-15e done: attack suite + before results, tenant partitioning, access control, input boundary, injection defenses, Qdrant lock-down + vector leakage, OpenAI provider + $10 spend guard (15g); generation bake-off (15g1: planned gpt-4.1-mini lost to qwen) + prompt fit (15g2: stricter prompt worse, negative result); gpt-5-mini ties qwen (15g3) and is now the served generator; router re-eval (15g4): nano kept, billing 8/8; real-stack security batch (15h): v3 served, poisoned obedience 5/50 -> 0/50; OQ #18/#19/#20 fixed (15k/15j/15l); wrap-up (15i) done, all 4 exit criteria met). Phase 14 — API, Streaming & UI (✅ complete 2026-10-17 — all 6 build items, all 5 exit criteria met). Phase 13 — Observability & the Latency Budget (🟡 in progress — build list complete, exit criteria 3 of 4 met -- p95 < 3 s unmet). Phase 12 — Caching, Delta Indexing & Cost (✅ complete — all 6 units, exit criteria met). Phase 11 — Generation Eval, RAGAS & Judging the Judge (🟡 **wrapped up for this session** — every unit, fix, and root-cause investigation that could be done without waiting on quota is done and documented; exactly 2 real measurements are left, both **explicitly deferred to the user** to run themselves whenever quota allows, not queued as a future session's task; see "Phase 11 wrap-up" note below).

## Phase 12 — Caching, Delta Indexing & Cost

**Scope** (plan.md, Day 14, topics 33-34): 6 units — (1) exact-match cache, (2) semantic cache (similarity
threshold tuned on the golden set, with an explicit near-miss test), (3) hit-rate simulation against a
Zipfian query distribution, (4) delta indexing (content-hash change detection, purge deleted docs, a test
proving a deleted doc becomes unretrievable — the Air Canada failure mode as a regression test), (5)
quantization (int8/binary/Matryoshka truncation, recall vs. size/latency), (6) a cost-per-query model.
Fully unblocked by today's Groq/Gemini quota state — none of these 6 build items call an LLM.

**Unit 1 complete as of 2026-09-26** (design record 12, `src/aegis/cache/exact_match.py`, 10 new tests).
`ExactMatchCache` — sqlite-backed, keyed on a normalized-query hash folded together with an opaque
`corpus_version` string (so a reindex automatically orphans stale entries rather than needing an explicit
purge). Stores a purpose-built `CachedAnswer` (text, cited chunk ids, provider), not the full
`GenerationResult` — deliberately decoupled from `PackedContext`'s full `Chunk` payload, which is already
durably stored elsewhere and would bloat the cache for no serving benefit. `corpus_version` itself is
accepted as a parameter, not computed here — that's Unit 4's (delta indexing) job. Same sqlite
thread-safety pattern as `EmbeddingCache` (Phase 3). TDD'd (10 tests: normalization, case/whitespace
insensitivity, corpus-version scoping, persistence across instances). `uv run pytest -q`: 430/430 passed
(+10 new).

**Unit 2 complete as of 2026-09-26** (design record 12b, `src/aegis/cache/semantic.py`,
`experiments/12a_semantic_cache_threshold_sweep.py`, `docs/results/semantic-cache-threshold.md`, 10 new
tests). Real, local, zero-LLM-cost threshold tuning: computed the 15 highest-similarity pairs among
genuinely *different* real dev-positive golden queries (real negatives, up to 0.8694 similarity) and 15
hand-written paraphrases of real golden queries (positives, 0.744-0.948 similarity) against `bge-large`.
**Found real overlap between the two populations** — no threshold perfectly separates "same question,
different words" from "different question, same topic" on this corpus. Chose 0.88 (lowest threshold with
zero real false positives), same asymmetric-cost logic as Phase 10's tau — a false positive here serves a
confidently wrong cached answer, a false negative just costs one extra real pipeline call. Real hit rate at
that threshold: 9/15 (60%) paraphrases caught — a disclosed, moderate ceiling, not inflated. `SemanticCache`
(brute-force cosine, sqlite-persisted, same `corpus_version`/`CachedAnswer` shape as Unit 1) TDD'd with a
fake deterministic encoder (9 tests) plus one real integration test using the actual shipped threshold
against the real hardest near-miss pair found (`golden::issue::1320` vs. `332`, real 0.8694 similarity,
correctly does not hit) — the explicit near-miss test plan.md's exit criteria ask for. `uv run pytest -q`:
440/440 passed (+10 new).

**Unit 3 complete as of 2026-09-26** (design record 12c, `experiments/12b_hit_rate_simulation.py`,
`docs/results/cache-hit-rate-simulation.md`). Zero LLM cost: real `ExactMatchCache`/`SemanticCache`
classes (Units 1-2, unmodified) driven by a simulated Zipfian-distributed stream of up to 9,000 requests
over the 150 real dev-positive golden intents, cache-miss cost bootstrapped from the 4 real
`(latency_ms, completion_tokens)` samples already on disk (`experiments/11b_qwen_reasoning_check.py`'s raw
output) rather than fresh live generations -- spending quota to measure how much caching avoids spending
quota would defeat the point. Real `bge-large` embeddings decide every semantic-tier hit/miss (165 real
encode calls total, memoized by exact text since a deterministic encoder gives identical vectors for
identical strings). **No real production traffic log exists (never deployed)**, so the Zipf skew itself is
swept (0.8/1.2/1.6), not asserted, and only 15/150 intents have a real known paraphrase (Unit 2's set), so
the semantic-tier share here is a disclosed conservative lower bound. **Real, load-bearing finding: traffic
volume dominates skew once N is large relative to the intent pool.** All three skews converge to ~98% hit
rate / ~98% latency+token reduction by N=9,000 (60 requests/intent), but diverge sharply at a cold start
(N=150, 1 request/intent): 56.7% (skew=0.8) vs. 86.7% (skew=1.6). Because this golden set only has 150
real unique intents, the measured ~98% ceiling is a property of this evaluation set's intent diversity, not
a general real-world hit-rate claim -- a production deployment facing a much larger real question space
would sit further down this same curve until its own traffic volume caught up, explicitly disclosed rather
than reported as a flat headline number. Reconciled an apparent tension, not a real one: semantic-tier hit
share here is small (0.2-1.6% of all traffic) despite Unit 2's 60% real paraphrase catch rate, because only
10% of intents can ever produce a cross-phrasing repeat at all -- the two numbers measure different
populations. `uv run pytest -q`: 440/440 passed (no new tests -- I/O-bound experiment script, same
convention as Phase 9/10's sweep scripts).

**Unit 4 complete as of 2026-09-26** (design record 12d, `src/aegis/ingest/delta.py`,
`src/aegis/embedding/vector_store.py`, `tests/test_delta_indexing.py`, 15 new tests). `IndexManifest`
(chunk_id -> content_hash, persisted/diffable), `diff()` (added/changed/deleted/unchanged against a prior
manifest), `corpus_version()` (a real hash over the manifest -- finally a real value for the placeholder
string Units 1-3's caches were handed), `apply_delta()` (embeds only what changed, deletes what's gone,
touches nothing else). **Found and fixed a real, previously-latent bug before delta indexing could work
safely at all**: `embedding/vector_store.py`'s `upsert_chunks` assigned each Qdrant point ID from its
position in the chunks list (`id=i`), which only ever worked because every real call site
(`embedding/run.py`) also fully wipes and rebuilds the collection every time
(`create_collection`->`recreate_collection`) -- one deletion anywhere shifts every later chunk's position,
and therefore its point ID, silently corrupting any future incremental upsert/delete. Verified directly,
live against the local Qdrant instance: an arbitrary chunk_id string (Qdrant point IDs must be an unsigned
int or UUID) is rejected outright. Fixed with `chunk_point_id()` (deterministic `uuid.uuid5` over the real
chunk_id) plus non-destructive `ensure_collection`/`delete_chunks` primitives (never call
`recreate_collection` on the delta path). **Real integration test proves the literal plan.md exit
criterion** (a throwaway `_test_delta_indexing` Qdrant collection, never the real `aegis_chunks_*`
collections other phases' results depend on): indexes 2 fake documents (4 chunks), deletes one document,
asserts *both* the real Qdrant point count drops 4->2 (checked directly, not just through
`DenseIndex` -- its in-memory safety net could otherwise mask an actually-broken delete) and a rebuilt
`DenseIndex` no longer surfaces that document's chunks while the surviving document's chunks stay fully
retrievable. TDD'd (15 tests: manifest round-trip, `corpus_version` determinism/sensitivity, all four
`diff()` categories, then the one real Qdrant integration test). **Explicitly not done, flagged rather than
silently skipped:** the live production collections (`aegis_chunks_bge_large`/`_small`/`_m3`) still carry
the old positional point IDs from their last full rebuild -- harmless as-is (nothing reads point IDs
directly, and Phase 6/7's results are unaffected), but a first real delta operation against any of them
would need one full re-upsert under the new stable-ID scheme first; and Units 1-3's cache call sites still
pass a placeholder `corpus_version` string, not yet wired to `IndexManifest.corpus_version()`. `uv run
pytest -q`: 455/455 passed (+15 new).

**Unit 5 complete as of 2026-09-26** (design record 12e, `experiments/12c_quantization_sweep.py`,
`docs/results/quantization-tradeoff.md`/`.png`). Zero new corpus embedding calls -- brute-force cosine/int8/
Hamming search over the real, already-computed `vectors_bge_large.npy` (19,236 chunks, Phase 3), swept
across 3 dims (1024/512/256, Matryoshka-style truncation) x 3 precisions (float32/int8/binary). Real,
consistent pattern: **int8 is nearly free** (4x smaller, costs 1.1-3.4 points of recall@10 at every dim);
**binary is not** (32x smaller, costs 9-13 points at a given dim, compounding with truncation down to
recall@10=0.1203 at 256d/binary vs. 0.3194 at 1024d/float32). **Real, load-bearing side-finding, investigated
rather than shipped past (CLAUDE.md: stop on a contradicted expectation):** this sweep's own float32/1024/
no-quantization baseline (recall@10=0.3194) does not match `docs/results/rerank-tradeoff.md`'s already-
published no-rerank baseline (0.4324) for the identical queries/encoder/exclude-doc-id rule. Ruled out
query-vector differences, corpus-vector staleness, and a top-k-selection bug directly (all confirmed
byte-identical / reproducible). **Real cause: live Qdrant HNSW search and exact brute-force cosine agree on
only 57.8% of top-10 results on average**, and of real ground-truth chunks either search finds, 348 are
found only by the live index vs. 95 found only by brute force (~3.6x asymmetry) -- most likely because this
golden set's ground truth (Phase 4) was labeled from candidates the live index itself surfaced, so labels
are structurally biased toward that specific approximate index rather than true nearest-neighbor-by-cosine.
Doesn't invalidate Phase 6/7's numbers (correctly measuring the system as actually served) but means this
golden set's recall is partly measuring agreement with the labeling mechanism, not embedding quality alone
-- flagged as context for Phase 16/19, not chased further here. Also disclosed: binary search measured
*slower* than float32 here (unoptimized NumPy popcount loop vs. BLAS matmul, not a real property of binary
quantization -- a real SIMD-accelerated index would invert this), and `bge-large-en-v1.5` was never trained
with a Matryoshka objective, so the truncation numbers measure a real, harder, non-MRL cost. `uv run
pytest -q`: 455/455 passed (unchanged -- I/O-bound sweep script, no new tests, same convention as Phase 9's
sampling sweep).

**Unit 6 complete as of 2026-09-26 — Phase 12 build list done, all 6 units** (design record 12f,
`experiments/12d_cost_model.py`, `docs/results/cost-per-query.md`, `docs/results/cache-performance.md`).
Zero new measurements -- assembles real numbers already published across this project (retrieval latency,
Phase 3; rerank latency, Phase 7; confidence-scoring latency, Phase 10; generation latency/tokens, Phase
9/11; cache-hit latency, Phase 12 Units 1-3) into 5 real per-query cost profiles. Real result: cache miss
(shipped, no rerank) = 12,787.5 ms / 254.7 completion tokens; with rerank (pool=10, not shipped) = 14,347.5
ms / 254.7 tokens; semantic hit = 0.8 ms / 0 tokens; exact hit = 0.0 ms / 0 tokens; escalation (partial,
see below) = 4,095.5 ms / 0 tokens. **Real, non-obvious finding:** confidence scoring runs unconditionally
on every non-cache-hit query (before the escalate-vs-generate fork), so escalation isn't free -- but it's
still ~3x faster than generating an answer and the only non-cache-hit path with zero LLM token cost, since
it never reaches the generation call. Used the real full 180-query dev-sweep confidence-scoring wall time
(design record 10 §6: 736s/180 queries = 4,088.9 ms/query) rather than a smaller per-pair estimate that
design record 10 already found undersells the real number by ~4x -- caught this almost being repeated
while drafting this unit. Deliberately reports tokens/latency, not dollars: this project has never left
Groq/Gemini free tiers, so no real dollar figure exists to report (inventing one would violate CLAUDE.md's
never-invent-a-metric rule). Escalation's cost is explicitly flagged as a partial lower bound --
`escalate_fn` itself (the real human-handoff mechanism) doesn't exist anywhere in this codebase yet
(Phase 14's job); only its payload construction (already counted in confidence-scoring cost) is built.
`docs/results/cache-performance.md` written as the phase-level synthesis doc the exit criteria ask for,
tying together Units 1-6's real results and the two required regression tests. `uv run pytest -q`:
455/455 passed (unchanged -- pure assembly of already-measured numbers, no new logic to guard).

**Phase 12 exit criteria: all 5 met** -- `docs/results/cache-performance.md` (written), the deleted-doc
regression test (Unit 4), the semantic-cache near-miss test (Unit 2), `docs/results/quantization-
tradeoff.md` (Unit 5), the cost-per-query table (Unit 6, above).

**Open items carried forward, not silently dropped:** (1) the live production Qdrant collections still
carry the old positional point-ID scheme (design record 12d §7) -- harmless as shipped, but a real delta
operation against them needs one full re-upsert first; (2) `corpus_version` at the cache call sites is
still a placeholder string, not yet wired to `IndexManifest.corpus_version()` (design record 12d); (3)
escalation's real handler doesn't exist yet (Phase 14).

**Next action:** Phase 13 (Observability & the Latency Budget) -- Langfuse tracing, per-stage latency
attribution, and the p95 < 3s budget gate. Its own per-stage measurements will be the first place to check
Unit 6's assembled cost-model numbers against real traced requests instead of numbers stitched together
from separate experiments run at different times.

## Phase 13 — Observability & the Latency Budget

**Scope** (plan.md, Day 15, topics 31 & 35): 5 build items -- (1) Langfuse tracing every stage (routing,
retrieved chunks w/ scores, rerank scores, confidence, prompt version, generation, validation result, cache
status), (2) structured logging with a trace ID threaded throughout, (3) per-stage latency attribution
(p50/p95/p99), (4) optimize against the p95 < 3s budget (parallelize dense/sparse retrieval, batch rerank,
warm models at startup, stream earlier), (5) the audit-trail demo.

**Unit 1 complete as of 2026-09-27** (design record 13, `src/aegis/config.py`'s new `LangfuseSettings`,
`src/aegis/observability/tracing.py`, `experiments/13a_langfuse_smoke_test.py`). **Real capacity check before
committing to an approach:** the build list says "Langfuse (self-hosted)," so checked its actual
dependency footprint rather than assuming a quick `docker-compose` add, per this session's explicit
instruction. Official v3 self-host stack is 6 containers (web, worker, Postgres, ClickHouse, Redis, MinIO),
with Langfuse's own docs recommending 4 cores/16 GiB for that stack alone. Measured this machine: Docker
Desktop allocated 7.75 GiB total, Qdrant already using ~750 MiB of it, and the host's full 16 GiB also has
to leave headroom for MPS embedding inference running outside Docker. Real, load-bearing capacity conflict
-- presented 3 options (full self-host, self-hosted v2/Postgres-only, Langfuse Cloud free tier) with a
recommendation via `AskUserQuestion`; **user chose Langfuse Cloud**, a disclosed deviation from plan.md's
literal "self-hosted" wording (Decision Log, 2026-09-26). `uv add langfuse` (v4, OTel-based SDK, clean
install, no dependency conflicts) -- `LangfuseSettings` added following the project's existing per-group
`BaseSettings` pattern, `get_tracer()` a process-wide `lru_cache`d singleton (the SDK owns its own
background flush thread + OTel `TracerProvider`; more than one per process would mean two independent
export threads to the same project).

**Real live verification, 2026-09-27, two real bugs found and fixed before it passed clean:** (1) first run
hit a real `401 Unauthorized` -- `LangfuseSettings` originally named the field `host` (reads
`LANGFUSE_HOST`, unset), silently falling back to a hardcoded EU default (`cloud.langfuse.com`) against the
user's real US-region project, whose correct URL was already in `.env` under `LANGFUSE_BASE_URL` (Langfuse's
own dashboard snippet name) -- fixed by renaming the field to `base_url` so it reads that env var directly.
(2) Second run hit a real `410 LEGACY_API_UNAVAILABLE_FOR_NEW_ORGANIZATION` -- this org (created 2026-09-27,
after Langfuse's own 2026-09-16 API cutover) has no legacy single-trace endpoint at all; rewrote the
verification readback from `client.api.trace.get()` to the current `client.api.observations.get_many(
trace_id=..., from_start_time=..., to_start_time=...)`, and separately found (and fixed) a real
async-ingestion race -- a 3-second wait after `flush()` returned zero observations, an isolated debug trace
confirmed 0 at 3s vs. 1 at 8s, settled on an 8-second wait. Third run, real output: `auth_check() passed`,
a real trace + 2 nested observations (`retrieval`, `generation`) round-tripped and confirmed via the live
Langfuse Cloud API (not just "the SDK didn't raise"), trace URL:
`https://us.cloud.langfuse.com/project/cmujtds4508cbad0dj9jedaxo/traces/1350343caaf045fcb70958e6e805b1f6`.
`uv run pytest -q`: 455/455 passed throughout (unchanged -- this unit is configuration/wiring, no new logic
needing a unit test; the real verification is this live integration run). Full detail: design record 13 SS5-6.

**Unit 2 complete as of 2026-09-28** (design record 13b, `src/aegis/pipeline.py`, `tests/test_pipeline.py`,
`experiments/13b_pipeline_live_trace.py`). **Real gap found before building:** no orchestrated end-to-end
pipeline existed anywhere in this codebase -- Phases 5-12 each built and unit-tested one stage in isolation
(router, retrieval, rerank, confidence, generation, citation validation, cache), wired together only ad hoc
inside experiment scripts. Phase 13's own build item #1 ("tracing every stage") presupposes a real request
to attach spans to; Phase 14 ("API, Streaming & UI") is where an orchestrator would normally get built.
Flagged this directly (`AskUserQuestion`) rather than silently picking a scope; **user confirmed: build a
minimal orchestrator now**, so Phase 14 wraps this exact object in FastAPI/SSE instead of rebuilding it.

`AnswerPipeline.answer(query) -> PipelineResult` composes cache -> router -> retrieval -> confidence ->
generation (stage order matches `router/classifier.py`'s own documented architecture), every collaborator
dependency-injected including the tracer, mirroring `GenerationPipeline`/`RerankedRetriever`'s existing DI
style. Each stage opens its own real Langfuse span (`as_type` per design record 13 SS6's mapping: retrieval
-> `retriever`, confidence -> `evaluator`, generation -> `generation`, routing/escalation/validation ->
`guardrail`). Two structural audit properties proven the same spy-based way `router/policy.py`/
`guardrails/confidence.py` already prove theirs (a collaborator that raises if ever called): a cache hit
never reaches the router or generation; an escalation (hard-route or low-confidence) never reaches
generation. `corpus_version` deliberately kept as an honest placeholder string, not wired to
`IndexManifest.corpus_version()` -- no manifest has ever actually been written for the live collections
(`data/processed/manifests/` doesn't exist), so that hash would encode zero real content while looking
real; resolving it for real means a deliberate, separate delta-indexing pass against production Qdrant, not
a side effect of this unit. TDD'd first (RED confirmed via `ModuleNotFoundError`, then GREEN): 9 new tests.
`uv run pytest -q`: 464/464 passed (+9 new).

**Real live end-to-end run, 2026-09-28**, real corpus (19,236 chunks), real Qdrant, real Groq router +
generation, one real golden-set query (`golden::issue::604`, "query parameters with hyphens"): cache miss ->
`docs_howto` route -> real retrieval -> confidence 0.7953 (above tau) -> real Groq generation, single valid
citation (`issue::4842::heading_aware::8`), grounded, correct answer. All 7 real spans (`answer_query`,
`cache_lookup`, `routing`, `retrieval`, `confidence`, `generation`, `validation`) confirmed present in
Langfuse Cloud. **A second real run against the same query correctly short-circuited to `exact_hit` with
only 2 spans** -- the first run's write-through being served back, proving the cache+tracing composition
together, not just each piece alone.

**Two real, disclosed findings from this live run:** (1) ingestion latency scales with real trace size, not
a fixed constant -- Unit 1's tiny synthetic trace was queryable at 8s, but this real 7-span trace (larger
real payloads) still had 4 spans missing at 8s and wasn't complete until ~14s; rewrote the live-trace
script's readback from a fixed sleep to a poll loop (every 5s, up to 30s). (2) The verification script's own
first version had a real bug, not the pipeline -- its hardcoded "expect all 7 spans" check failed on the
second run's legitimate 2-span cache-hit trace; fixed by making the expectation branch-aware on
`result.cache_status`/`result.escalate`. Full detail: design record 13b SS5-6.

**Next action:** Unit 3 -- per-stage latency attribution (p50/p95/p99) against a real batch of queries
through this same pipeline, then Unit 4 (optimize against the p95 < 3s budget) and Unit 5 (the audit-trail
demo, which now has a real `PipelineResult`/trace pair to reconstruct evidence from).

**Unit 3 complete as of 2026-09-30** (design record 13c, `docs/results/latency-budget.md`,
`src/aegis/observability/latency.py`, `experiments/13c_latency_attribution.py`/`13d_latency_report.py`).
`PipelineResult.stage_latencies_ms` measured inside the pipeline (not read back from Langfuse, whose
ingestion lags 8-14s); `GenerationPipeline.answer()` split into `generate()` + `validate()` so the two stages
get honest separate timings; percentile aggregator TDD'd against hand-computed values. `uv run pytest -q`:
471/471 passed (+7 new). **First measurement was wrong and was caught:** 25 back-to-back queries gave a
generate-path p50 of 38,080 ms (routing, a ~10-token call, up to 14.4s) -- rerunning with 70s pauses gave
4,012 ms p50 / 4,688 p95 (n=12), routing 251 ms. Likely cause: Groq free-tier per-minute token cap causing
retry-after waits; supported by the effect only, 429s were not logged. Both datasets kept and labeled.
**Baseline (paced, generate path p50/p95 ms):** confidence 2,520/3,233 (67% of p50), generation 920/1,443,
routing 251/334, retrieval 44/66. Budget p95 < 3,000: pre-generation path 3,486 OVER, full generate path
4,688 OVER, escalation path 3,128 OVER. Cold start: first query's confidence stage 17,435 ms (lazy reranker
load). **Flagged, not resolved:** (1) Phase 12's cost model used 12.8 s per cache miss from earlier
possibly-unpaced generation samples -- generation measures 0.9 s here; needs re-deriving. (2) `confidence`
(local CPU, no LLM) was ~45% slower in the burst run for an unexplained reason.

**Next action:** Unit 4, optimize against p95 < 3 s. Attribution says confidence scoring + cold start
dominate; plan.md's "parallelize dense and sparse retrieval" targets a 44 ms stage and shipped retrieval is
dense-only -- raise that scope question with the user before building.

**Unit 4 done, budget NOT met, one cost unexplained -- 2026-09-30** (design record 13d,
`docs/results/latency-budget.md` Unit 4 section, `AnswerPipeline.warm_up()`, experiments `13e`-`13g`).
Only `warm_up()` was built: plan.md's "parallelize dense and sparse retrieval" has nothing to act on (shipped
retrieval is dense-only, 37-44 ms), "batch rerank" already holds, and shrinking the rerank pool would change
confidence values and invalidate Phase 10's tau. **Warm-up works:** first request 19,208 ms -> 2,957 ms wall
(n=1 each side); cost moved to a 30.3 s startup. **Steady state did not improve** (generate path p50/p95
4,012/4,688 -> 4,421/5,649 ms; pre-generation p95 4,030; all over 3,000 ms). **Wrong hypothesis, tested and
refuted:** I suspected background load (an unrelated Docker stack was running during the baseline); it was gone
for the re-run and confidence stayed ~2.4 s. Isolated `ConfidenceScorer`: 560 ms back-to-back, 1,031 ms after
idle gaps; inside the pipeline with a real Langfuse tracer 926 ms (p95 6,070, one 11.3 s outlier) vs 502 ms with a
no-op tracer -- suggestive that tracing costs latency, not established (different queries per arm, n=12). Also
fixed en route: Docker Desktop itself had stopped (twice), taking Qdrant with it -- restarted, not a code issue.
`uv run pytest -q`: 472/472.

**Phase 13 is NOT done.** Remaining: structured logging with a trace ID threaded through (build item 2 -- NOT
built; Unit 2 only threaded the ID through Langfuse spans), the audit-trail demo (item 5), and the p95 < 3 s exit
criterion, which is unmet. **Open items:** (1) confidence stage ~2.4 s in pipeline vs 0.5-1.0 s isolated,
unexplained; (2) Phase 12 cost model used 12.8 s/miss from possibly-throttled samples, generation measures ~1 s;
(3) 429s/retries are not logged, so the rate-limit explanation for the burst run is unconfirmed.

**Confidence-gap chase + structured logging + a silent router outage -- 2026-09-30** (design record 13e,
`src/aegis/observability/logging.py`, `experiments/13h`/`13i`). **Structured logging built (build item 2):** stdlib
JSON logging, trace ID via `ContextVar` attached to every record of a request, pipeline emits
`request_start`/`escalated`/`request_end`; no new dependency. `uv run pytest -q`: 483/483 (+11). **Real incident
found:** I exhausted `qwen/qwen3.8-27b`'s 200,000 tokens/day Groq cap (`Used 199902`); the router's
`except APIError: return ""` fed its fail-safe, so *every* query silently escalated as `billing_account_legal` --
no log anywhere. Fail-safe behavior kept; router now logs `router.api_error`/`router.unparseable_output`. **This
consumed today's quota for the user's deferred Phase 9 sampling sweep (same model).** Result-level gap remains:
an outage-driven escalation's `escalation_reason` still reads as a billing escalation.
**Confidence gap, narrowed and two earlier claims retracted:** same-queries/paired experiments localize the extra
time entirely to the cross-encoder call (encode ~2 ms), and rule out background load, the real tracer, 70 s idle
gaps, the real router, and GPU wake-up (throwaway rerank: -127 ms typical but costs 199 ms itself). Earlier
claims that idle gaps and the tracer each ~doubled the cost did **not replicate** and are retracted in 13d/
latency-budget.md. Same query `2689`: 855-1,072 ms here vs 2,299-3,326 ms in the batches. **One difference left
untested:** batches run real generation + validation in-process. Ready: `AEGIS_I_REAL_STACK=1 uv run python
experiments/13i_gpu_wake_paired.py` -- blocked on Groq quota.

**Phase 13 still NOT done.** Remaining: audit-trail demo (item 5); p95 < 3 s unmet. Structured logging: done.

**Unit 5 (audit-trail demo) complete -- 2026-09-30** (design record 13f, `src/aegis/observability/audit.py`,
`experiments/13j_audit_trail_demo.py`, `docs/results/audit-trail-example.md`). **Found first that the trace could
not answer "what was [1]?"** -- it recorded top-10 retrieved and validated citation ids but not which chunks were
packed into the prompt, their `[n]` numbering, truncation, or budget drops; fixed before building the reader
(`ContextEntry` with `content_hash` on `PipelineResult.context` and the generation span's input, with prompt
version + generation params). `build_audit_trail` (pure) / `resolve_evidence` (attaches chunk text; flags
ok/changed/missing/unverified vs the recorded hash) / `render_markdown` / `fetch_observations` (polling; the v2 API
omits `input`/`output` unless `fields="core,basic,io"` and returns JSON strings). Missing stages, a missing
context map, and span-vs-root disagreement are reported as warnings/inconsistencies, never hidden. Tests generate
observations from a real `AnswerPipeline` run so span-format drift breaks a test. `uv run pytest -q`: 496/496
(+13). **Live, real, 3 queries (answered / hard-route escalation / confidence escalation `0.2295 < tau 0.5`):
reconstructed from the stored Langfuse trace alone, then compared to the live pipeline's returned object: 0
warnings, 0 inconsistencies, 0 differences on all three.** Known limits (13f SS6): no per-claim validation
detail, no stored prompt text (deterministic, not stored), generation `model` field is the client class name,
tau only inside the reason string, **Langfuse Cloud free tier retains 30 days -- not a legal artifact until
exported to durable storage.** Incidental (n=1, not evidence): the live answered request took ~1.9 s end to end.

**Phase 13 build list is complete; exit criteria 3 of 4 met.** Met: every request traced, `latency-budget.md`,
worked audit-trail reconstruction. **Unmet: p95 < 3 s** (paced batches 4.4 s p50 / 5.6 s p95 generate path; cause
of the ~2.4 s confidence stage in those batches still unexplained -- localized to the cross-encoder call, five
candidate causes ruled out; the real-generation-in-process test remains unrun). Phase board stays 🟡 on that
criterion. Open items carried: (1) that confidence-stage gap; (2) Phase 12 cost model's 12.8 s/miss vs ~1 s
measured generation; (3) audit-trail retention/export; (4) an outage-driven escalation's `escalation_reason`
still reads as a billing escalation.

**Phase 11 judge-κ rescore run, 2026-10-01 (partial)** (`docs/results/judge-agreement.md` update). Ran
`experiments/11g_judge_rubric_rescore.py`: Groq judge now complete (29/29); Gemini scored 11/29 before its
20/day cap. **Paired N=11: κ = 0.2143, raw agreement 0.636 (pre-fix: -0.0161 at N=9); still below the 0.6 bar.**
Caveat: Gemini gave 10/11 paired answers a 5, so κ is unstable at this skew; all 4 disagreements are Groq 4 /
Gemini 5; the "reserve 5" anchor did not cure Gemini's ceiling effect. 18 Gemini scores remain (script is
resumable, ~1 day of Gemini quota). **Correction to earlier notes: `make eval` does NOT pick up the rescore** --
`EvalSettings.generation_batch_raw_path` still points at the original pre-fix raw file, so the scorecard still
reports κ = -0.0161. Decision needed: report the fixed-rubric κ (and/or both) in the scorecard. Not changed.

**Phase 13 confidence-gap follow-up, 2026-10-01** (`docs/results/latency-budget.md` follow-up section,
`experiments/13k_shape_first_call.py`). Ran the previously-untested hypothesis: real router + generation +
validation in-process (13i real-stack, no abort): confidence p50 647 ms plain / 516 ms woken -- **refuted**, same
as fakes. Also refuted: per-shape first-call compile cost (13k: call1 ~= call2 ~= call3) and power state
(battery + Low Power Mode ON, still fast). The stage's cost does follow padded sequence length (238 ms at 318
tokens -> 480-880 ms at 512). **Still unexplained:** query `753` costs ~0.5-0.8 s in 13i/13k but ~2.5 s in the 13c
batch script (reproduced in 2 runs: pilot and rerun); something harness-specific (13c reuses one pipeline and its
caches across queries) inflates the stage; not bisected. **The p95 < 3 s criterion remains unmet and
unresolved; the verdict depends on the harness.** Rerun generate-path wall times 3,234/3,734/2,745 ms (n=3).
**Phase 9 sampling sweep NOT run:** needs ~100k qwen tokens; today's window still holds this morning's real-stack
runs and yesterday's evening batches, headroom unknown (a probe only proves >=16k). Run once the window clears
(after ~20:00 local): `uv run python experiments/09a_sampling_sweep.py` (resumable).

## Phase 16 — The CI Evaluation Gate

**Scope** (plan.md, Day 18, topic 30): (1) `test.yml`, (2) `eval.yml` failing the build on metric regressions incl. p95, (3) cost control
(full suite on PRs, fast subset on pushes), (4) deliberate regression proves the gate, screenshotted, (5) metric diffs as PR comment.
Exit: both workflows green on main; regression demonstrably fails the gate; diffs on PRs. **Never cut.**

**Decisions (user, 2026-10-22):** lint = narrow gate (pinned ruff, explicit `E4,E7,E9,F,I`; mypy on 18 core modules; no mass reformat) ·
eval runs on a **GitHub-hosted runner with a restored index** (`vectors_bge_large.npy` 75 MB + `chunks.jsonl` 27 MB, BM25 rebuilt; no
corpus embedding on the runner) · LLM-backed checks in the PR gate, capped **~$0.15 per PR run, ~$1.50 reserved** (of ~$9.65 left),
real cost of one run to be measured and reported before any workflow is wired.

**Unit 16a complete as of 2026-10-22** (design record 16a, `.github/workflows/test.yml`, `pyproject.toml` ruff/mypy config + pinned dev
deps, `docs/future-work.md` created). Clean-checkout rehearsal (fresh venv, no `.env`/`data/processed`, key-less Qdrant v1.18.3
container): ruff clean, mypy clean (18 files), **744 passed, 2 skipped, 109 s** (local: 746 passed, 18 s). The 2 skips are the real-corpus
tenant-bypass tests; they skip with a reason on a bare checkout and **fail** under `AEGIS_REQUIRE_CORPUS=1` (to be set by `eval.yml`).
Gates verified to discriminate with a probe file (unused import -> F401; `-> str` returning `int` -> mypy return-value). Fixed along the
way: 36 ruff findings (33 autofix, 3 by hand), 1 mypy error (`chunking/semantic.py` `_MODEL_CACHE`). Ruff 0.17.1 *defaults* report 103
findings vs 45 with the explicit list, hence pinned + explicit. **GitHub run 1 failed in 7 s** (`setup-uv@v10` is not a real tag; setup-uv publishes exact tags only -> pinned `v10.3.2`); **run 2 green: 3 m 27 s, 744 passed / 2 skipped in 57.4 s, install 57 s with CUDA torch (acceptable).**
**Resolved:** uncommitted Phase 15 (user pushed b5e97c2, b9b4a7b); CUDA torch (fine). **Open:** GitHub repo is private (2,000 free Linux min/month), so CI minutes are a budget.
**Unit 16b measured 2026-10-23 -- the original restore plan does not work; decision pending** (design record 16b, `src/aegis/eval/ci_index.py` +
6 tests, `experiments/16a-16c`). Rebuilding the index from `vectors_bge_large.npy` scores recall@10 0.252/0.256, MRR 0.644/0.668 vs the live
index's 0.437/0.864 (two rebuilds agree on only 54% of top-10s). Diagnosed (not ties: 4% of exact top-10; not the filter: Qdrant `exact=True`
with filter = 98.7% of numpy exact): the served HNSW returns only **~67% of the true cosine top-10** (id overlap 0.671, score recall 0.682),
and the golden labels came from this specific graph, so exact cosine scores *lower* on it (recall@10 0.320, MRR 0.708). A Qdrant snapshot of the
live collection (334 MB, 103 MB gzipped, restore 3.3 s) restores deterministically (150/150 repeat) and lands near live (recall@10 0.4439, MRR
0.864, 130/150 identical top-10s). Other costs: CPU query encode 269 ms/query (150 queries 40.3 s, cache bypassed), bge-large load 13.9 s.
Added to `docs/future-work.md` for Phase 19. `uv run pytest -q`: 752 passed. $0 spent. **Next:** user picks the gate's index source
(snapshot recommended), then thresholds + `eval.yml` (16c).

**Decision (user, 2026-10-23): gate on a Qdrant snapshot of the live collection** (release asset, ~106 MB: gzipped snapshot 102.9 MB + chunks 3.9 MB).

**Unit 16c complete as of 2026-10-23** (design record 16c; `eval/gate.py` 13 tests, `eval/ci_fixtures.py` 6 tests, `eval/retrieval_gate.py`,
`GateSettings`, `ci/baselines/retrieval.json`, `experiments/16d_encoder_noise.py`). Gate fails on anything but a clean in-band comparison
(missing/NaN metric, fingerprint or n_queries mismatch, empty spec list). Export 16 s, restore 5.8 s, restored chunks byte-identical.
Gate run on restored snapshot (M1 CPU, live encode, cache bypassed), repeated twice -> identical to all digits: **recall@10 0.4317, MRR 0.8674,
NDCG@10 0.7100**; search p95 ~17 ms; encode 0.271 s/query. Tolerance set from a noise measurement: benign delta (cached vs live query vectors,
<=3.4e-7 apart) = recall 0.0122 / MRR 0.0034 / NDCG 0.0089, batch-size noise = 0 -> bands **0.025 / 0.01 / 0.02** (2x, rounded up; recall band <
the 0.035 = 8% relative example). Synthetic 8% drop -> FAIL exit 1. First tolerance guess (0.02, from 0.0067) was wrong, corrected. Baseline is
**provisional (M1)**; 16d checks the hosted-runner delta. `uv run pytest -q`: **771 passed**; ruff + mypy (21 files) clean. $0 spent.
**Needs from the user:** upload `ci-fixtures/` as a release (`gh release create ci-fixtures-v1 ci-fixtures/*`). **Next:** 16d `eval.yml`.

**Unit 16d in progress (2026-10-24) -- first two hosted-runner runs of `eval.yml` found a real defect, fixed, awaiting run 3.**
Run 1 (2m24s): restore failed -- Qdrant status `yellow` right after the snapshot upload; my check demanded green. Run 2 (10m): restore + 31 security tests
passed, retrieval ran (7m41s; **2.43 s/query CPU encode on the runner vs 0.27 on the M1**), compare FAILED: recall@10 **0.2687**, MRR **0.5842**, NDCG@10
**0.4496** vs baseline 0.4317 / 0.8674 / 0.7100, same fingerprint `e99f16de52c7d00f`. Cause (reproduced locally with `docker run --cpus=2`): on a
2-CPU host Qdrant's optimizer **merges the restored segments and rebuilds HNSW** (yellow ~21 s, segments 4->5->2); the gate then measures a *different*
graph (local 2-CPU: 0.2859 / 0.6306 / 0.4842). Fix: run the CI Qdrant with `QDRANT__STORAGE__OPTIMIZERS__MAX_OPTIMIZATION_THREADS=0`; with it, a
2-CPU container reproduces the baseline (0.4344 / 0.8674 / 0.7118; segments stay 4, status stays yellow permanently). Restore now requires the optimizer
disabled + exported segment count (`segments_count: 4` added to MANIFEST; fingerprint unchanged) held for 3 consecutive polls, and refuses otherwise
(verified against real containers, both ways). 778 tests pass. **Needs from the user:** `gh release upload ci-fixtures-v1 ci-fixtures/MANIFEST.json
--clobber` BEFORE pushing, then push; run 3 shows the true x86-vs-M1 delta and decides whether the provisional bands stand.

**Unit 16d complete as of 2026-10-24** (design record 16d). Run 3 **green** (11m02s): same snapshot now preserved with the optimizer disabled;
runner (x86) vs M1 baseline = recall@10 0.4435 (+0.0118), MRR 0.8640 (-0.0033), NDCG@10 0.7188 (+0.0089) -- 47/33/45% of the 0.025/0.010/0.020 bands,
equal to the 16c benign delta, so the bands stand. **Baseline re-taken from the runner** (artifact of run 3; same fingerprint `e99f16de52c7d00f`).
Runner costs: restore 12 s, 31 security tests 10.3 s, retrieval step 9m03s (encode 2.87 s/query vs M1 0.27; run 2 was 2.43), search p95 13.9 ms.
CI budget: ~14 min per PR push (test 3 + eval 11) of 2,000 free min/month = ~140 pushes. Not yet measured: two identical runs on the runner (run 4, the
one after rebaselining, is the first). **Next:** 16e LLM-backed checks (measure one run's real cost first; budget ~$0.15/run) + p95 gate; 16f PR comment; 16g
deliberate regression screenshot. Exit criteria status: both workflows green on main -- test.yml yes, eval.yml (retrieval only) yes; regression demo and
PR-diff comment not yet.

**Unit 16e (router + model probe) complete as of 2026-10-24 -- first CI run green (run 40068902808, workflow_dispatch): router = baseline exactly (0.7759 / billing 1.0 / 1.0), probe ok, real cost $0.01536 (60 calls, 1m16s), retrieval-gate also = runner baseline exactly (repeatable). CI spend not in local ledger: project ~$0.446 of $10.** Original notes: (design record 16e; `eval/router_gate.py`, `eval/model_probe.py`, `ci/baselines/router.json`,
`llm-gate` job). **Production defect found: `gpt-4.1-nano` was removed from the OpenAI API** (404 "deprecated"; `/v1/models` still listed it) -- router, rewrite and
citation-entailment were all failing (every query would escalate `provider_unavailable`). User chose **gpt-4.1-mini for all three** (config swapped). Router eval
(58 labeled dev queries): gpt-4.1-mini 0.7759 / billing 8/8 precision 1.0 / p50 724 ms; gpt-4o-mini 0.7586 / 719 ms; gpt-5-nano 0.7759 / **13.6 s max** (rejected); old
nano 0.6379. Same model, temp 0: 0.7759, 0.7586, 0.7759 (one query flips) -> bands accuracy 0.035, billing precision 0.05, **billing recall 0.0**. Per-run cap
$0.15 via `OPENAI_BUDGET_USD`, verified real (tiny-cap test: 7 calls, $0.001534, then BudgetExceededError, no results written). Rewrite/entailment on mini are
**unmeasured**. 789 tests pass. Spend this unit $0.0801; **project total ~$0.432 of $10**. **Needs from the user:** `gh secret set OPENAI_API_KEY`, push, then
`gh workflow run eval.yml` (llm-gate runs only on PRs and manual dispatch). **Next:** generation-subset gate (measure cost first), escalation check as path-filtered
job (CPU-bound, ~2 h on runner by extrapolation), p95 gate, PR comment, deliberate regression.

**Unit 16f written 2026-10-24, not yet seen on a real PR** (design record 16f; `eval/pr_comment.py` stdlib-only + 8 tests; `report` job: `always()`, same-repo PRs,
sticky comment found by hidden marker, `startswith` so a human quoting it is not overwritten). Verdicts: failure -> red, cancelled -> incomplete, skipped -> "N gates not run"
(never "passed"). Comment rendered from real run 40068902808 artifacts; jq marker filter checked on sample JSON. Also this session: `eval` concurrency group now per event
(a manual dispatch had cancelled the push-to-main run). 797 tests pass. **Needs from the user:** push, then open a smoke PR (commands in chat) to watch the comment appear
and update in place. **Next:** 16g deliberate regression (screenshot), then generation-subset / escalation / p95 tiers.

## Phase 15 — Security Hardening

**Scope** (plan.md, Day 18½, topic 36): the five attack classes (indirect injection, KB leakage, tenant separation,
vector reconstruction, input validation). Exit: `tests/security/` >= 10 attacks; injection attempts fail, documented
before/after; tenant bypass provably impossible; `docs/results/security-assessment.md`.

**Unit 1 complete as of 2026-10-18 -- attacks written and run BEFORE any defense** (design record 15,
`tests/security/{conftest,test_attack_retrieval_partition,test_attack_indirect_injection,test_attack_http_surface}.py`,
`experiments/15a_security_baseline.py`, raw `docs/results/security-baseline-raw.json`, `docs/results/security-assessment.md`).
Zero LLM calls (qwen TPD exhausted today). Convention: each test asserts the *secure* behavior; succeeding attacks are
`xfail(strict=True)`, so a landing defense turns the test red until the marker is removed and the after is recorded.
`uv run pytest -q`: **586 passed, 11 xfailed** (+5 passing controls/defended, +11 documented attacks; Qdrant up).
**Results (before):**
- **A1 BM25 filter gap:** `HybridRetriever` passes `doc_type`/`product_area` to dense only. Real corpus,
  180 dev queries, `doc_type=documentation`: **609/1,800 top-10 slots (33.8%) are issue chunks, 161/180 queries leak**;
  dense-only 0. Latent: the served pipeline uses `DenseIndex` with no filter.
- **A2:** exact-cache key = sha256(corpus_version::normalized text), so no partition.
- **A3:** issue text can forge a `[n] (source: doc::...)` block (fixed `---` framing).
- **A4:** an injected refund commitment cited to the poisoned chunk is `SUPPORTED_LEXICAL` with 0 LLM calls. The validator
  checks support, not authority.
- **A5 HTTP:** all 3 endpoints anonymous; any trace-id holder reads query + chunk text and overwrites votes; 40 slow
  `/trace` calls hold 40/40 anyio threads and `/health` gets no answer in 2 s; `/feedback` 200/200 accepted;
  feedback comments reach the reviewer's terminal raw (escape sequences). XFF spoof **defended** (`[200,200,429,429]`).
- **A6 Qdrant:** no API key; bound `0.0.0.0`/`[::]:6333-6334` (LAN-reachable); payload holds **full chunk text**;
  scroll dumps the corpus. Writes not tested.
- **A7:** naive injection regex 7/7 dev adversarial, **4/150 dev positives (2.7%) false positives, all "override"**
  (`dependency_overrides`); evasion not measured. PII shapes: corpus 103 chunks with emails (heuristic: 121/140 matches
  placeholders), 30 URLs with credentials, 5 JWT-shaped; dev queries 2 email + 1 URL-cred; real interactions store 0/12.
- **A8, the headline:** Phase 10's "adversarial_injection recall 1.00" **reproduces exactly (0/7 pass tau=0.5, same
  per-query values) but only measures off-topic detection: the same 7 injections wrapped around a real FastAPI question
  pass the gate 14/14 (0.6893-0.8007).** It is not an injection defense.
**Found along the way:** my first `/trace`-starvation test falsely XPASSed (it read the probe after releasing the
blocked calls). Instrumented, found, fixed. Unexplained: appending injection text *raised* one control question's
confidence 0.049 -> 0.79-0.80. `dense.py`'s comment that the Qdrant payload doesn't carry content is wrong.
**User decisions (2026-10-18, all accepted as recommended):** (1) LLM-run attacks live in a separate versioned
`data/security/attacks_v1.jsonl`; golden adversarial items are referenced by id, not copied; (2) tenancy stand-in = `doc_type`
partition chosen by the authenticated principal, enforced on both retrieval legs + cache key; (3) input-side detection
hand-rolled and flag-only (injection regex) + high-precision PII redaction, measured against protectai DeBERTa /
Presidio as the rejected alternatives; (4) Gemini model is `gemini-3.5-flash` (config is correct).

**Unit 15b complete as of 2026-10-18 -- tenant partitioning** (design record 15b, `src/aegis/tenancy.py`,
`TenancySettings`, `retrieval/{sparse,dense,hybrid}.py`, `cache/{exact_match,semantic}.py`, `pipeline.py`, `factory.py`,
`api/app.py`, `tests/test_tenancy.py`, `tests/security/test_attack_tenant_bypass.py`, `experiments/15b_partition_bypass.py`,
raw `docs/results/partition-bypass-raw.json`). Partitions `full`/`docs`/`issues` (sets of doc_type). The pipeline is bound
to one partition at construction; `PartitionedRetriever.search` has no filter parameter (TypeError), pushes `doc_types`
down (Qdrant MatchAny; bm25s weight mask + exact post-filter; hybrid forwards every filter to every leg), then **fails
closed** (`PartitionViolation`) on any out-of-partition chunk. Cache scope = corpus_version + partition (required kw-only).
Pipeline refuses a retriever bound to a different partition. Request bodies with extra fields -> 422. Served default
`full` (unchanged behavior until 15c). **Numbers:** A1 609/1,800 -> **0/1,800**; bypass sweep 190 queries (10 bypass-shaped
+ 180 dev) x {docs,issues} x {dense,hybrid} on real Qdrant: **0 out-of-partition chunks, 0 violations raised, 0 short
lists**; mutation (reintroduce A1 bug) -> fails closed, 0 leaked. Real server starts; `partition` body -> 422.
`uv run pytest -q`: **612 passed, 8 xfailed** (3 partition attacks flipped xfail -> pass). Side effects: every existing
exact-cache entry is now unreachable (key changed); `12b`/`14c` experiments got `partition="full"` at cache call sites
(signature only). Not proven: Qdrant-internal isolation, cache timing side channel, direct Qdrant access (A6 -> 15f).
**Unit 15c complete as of 2026-10-18 -- access control** (design record 15c, `src/aegis/api/auth.py`, `api/app.py`,
`api/store.py`, `eval/review_queue.py`, `factory.build_pipelines`, `ApiKeyGrant`/`ApiSettings`, `tests/test_auth.py`,
`tests/security/test_attack_http_surface.py`). **User decision (plain-language options, 2026-10-18): anonymous chat +
per-browser ownership tokens + `/trace` behind API keys.** Rejected: keys on every endpoint (a public demo would ship the
key in `app.js`). Session = random 256-bit token in an HttpOnly/SameSite=Strict cookie, stored only as a hash; the
interaction records its owner; `/feedback` accepts only the owner (404 otherwise, so ids can't be probed); `/trace`
needs a `can_trace` key (401/403) and is capped at 2 concurrent (503 beyond); upstream Langfuse failures -> 502;
`/feedback` rate limit 10 burst / 20 per min per socket peer; invalid key -> 401 (never anonymous); keys configured by
sha256 (`python -m aegis.api.auth new-key`); principal -> partition -> pipeline (`build_pipelines` shares models; startup
fails if a key's partition has no pipeline); review tool escapes all non-printables. Store migrates (`owner` column;
old rows votable by nobody). **Real server (n=1 each; one router call for a billing query, no generation):** cookie set
with HttpOnly/SameSite=strict; other-cookie and no-cookie votes 404, owner 200; `/trace` anon/bad key 401; 30 votes ->
9 x 200 then 21 x 429; **40 concurrent `/trace` with a valid key -> 2 ran, 38 x 503, `/health` 3.6 ms**. Found: first
real keyed `/trace` was a 500 = Langfuse `ReadTimeout` (Langfuse slow/flaky at the time: direct calls timed out at 5.8 s then
succeeded at 4.9 s). Now a logged 502; **a real `/trace` 200 was never obtained this session**. A5a's original "all
endpoints 401" test was replaced by the decided policy (stated in 15c, not presented as a pass). TDD slip: auth tests not
run red before implementing. `uv run pytest -q`: **643 passed, 2 xfailed** (A3, A4 remain -> 15e). Phase 18 must set
`API_SESSION_COOKIE_SECURE=true`.

**Unit 15d complete as of 2026-10-18 -- input boundary** (design record 15d, `guardrails/input_boundary.py`,
`InputBoundarySettings`, `api/app.py`, `tests/test_input_boundary.py`, `data/security/{attacks_v1,pii_probes_v1}.jsonl`,
`experiments/15d_{input_boundary_comparison,deberta_padding_evasion,redaction_retrieval_impact}.py` + raw JSON). `redact()`
on query + history + feedback comment at the HTTP boundary (covers pipeline, both caches, store, export, Langfuse, Groq/Gemini).
Injection regex = flag only (frozen before evasion was measured; `override` dropped for 4/150 FP, cost golden 7/7 -> 6/7).
**Numbers:** regex 0/150 FP, 6/7 golden, 12/14 wrapped, **0/14 paraphrase**; **DeBERTa** 1/150 + 1/23 FP (one a Chinese-language
legit question), 7/7, 12/14, **14/14 paraphrase -- contradicted expectation, investigated:** injection + 984-token real issue ->
**0/7 (end), 2/7 (start)**; windowed 2/7, 4/7 with 7/150 FP at 556 ms/query; single-pass 194-210 ms mean. Decision (user's
flag-only) stands on the numbers. **PII:** regex 8/8 secret probes, 0/6 code negatives, 3/180 dev touched (all real);
**Presidio** 135/180 dev flagged at >=0.5 (code as URL/PERSON), 4/8 secrets missed, 67.6 ms mean. Redaction retrieval cost:
3 touched dev queries, top-10 overlap 10/10, GT hits identical. Libraries run only in a `uv run --with` overlay (not deps).
`uv run pytest -q`: **661 passed, 2 xfailed**. Leftover caches: DeBERTa 714 MB (`~/.cache/huggingface`), spaCy/Presidio in
uv cache; disk 97% full, so safe to delete on request. Idea for 15e: offline DeBERTa scan of corpus chunks (~64 min CPU -> Kaggle).

**Unit 15e complete as of 2026-10-18 -- indirect-injection defenses** (design record 15e, `guardrails/commitments.py`,
`CommitmentSettings`, `guardrails/citations.py`, `generation/prompts/{base,v3}.py`, `generation/pipeline.py`, `pipeline.py`,
`events.py` (`GuardrailWarning`, `DoneEvent.warnings`), `static/app.js`, `experiments/15e_commitment_fp.py`,
`data/security/commitment_probes_v1.jsonl`). (1) **Output-side commitment check, served:** term AND you/we framing AND
commitment AND NOT refusal -> `UNAUTHORIZED_COMMITMENT` before every citation check; no validated citation, never cached
(a hit replays without re-validating), `done.warnings=["unauthorized_commitment"]`, UI `role=alert` notice. A4 flipped.
(2) **Prompt v3, built NOT served:** per-request 64-bit key in BEGIN/END markers (key only in the system prompt, regenerated
if it appears in any chunk or the query) + instruction-hierarchy rules; v1/v2 byte-identical via `messages_and_key`. A3's strict
xfail now asserts *the served prompt* is forgery-resistant, so it flips when v3 is served. **Numbers:** commitment FP real
answers 0/557 (**vacuous: 0 sentences with a business term**), corpus **0/40,551** (61 with a term), technical-payment probe
**3/8 FP** (frozen patterns, not tuned on it); v3 forgeries 0 across 3 styles; **v3 +624 input tokens (+20.4%) per 10-block
request** (3,059 -> 3,683). `uv run pytest -q`: **695 passed, 1 xfailed**; under `GENERATION_PROMPT_VERSION=v3`: 695 passed.
**Owed (15g, quota):** model obedience to a poisoned chunk under v2 vs v3 (scratch collection), and v3 valid_citation_rate vs
v2's 0.7296 -- which decides whether v3 is served.

**Unit 15f complete as of 2026-10-18 -- vector store exposure** (design record 15f, `docker-compose.yml`, `.env` (+2 keys,
values never printed), `QdrantSettings.{api_key,read_only_api_key}`, `vector_store.get_client(read_only=)`, `factory.py`,
`baseline.py`, `tests/test_qdrant_client_keys.py`, `experiments/15f_{qdrant_exposure,vector_leakage}.py` + raw JSON).
**Before (new finding): LAN, no key -> create/write/delete collection all 200** (index deletion / vector tampering from the
network). Code-read mitigation: `DenseIndex` resolves text from the local corpus by chunk_id, so Qdrant writes can reorder
or drop real chunks but not inject new text (not tested destructively). **After:** ports `127.0.0.1` only; compose refuses to
start without both keys (`:?`); no key -> 401; read-only key (served API; no fallback to admin key) -> 403 on create, and
on upsert/delete-points/delete-collection against an existing collection; LAN -> connection refused. Container recreated;
19,236 points / 8 collections verified; real server warm-up on the read-only key OK (retrieval 137 ms). **Vector leakage
(lower bound, vectors only, bge-large known):** precision@10 raw 0.188 (< 0.235 frequency prior: hubness), hubness-corrected
**0.286**, random 0.015; 15/100 chunks >= 5/10 (e.g. OAuth2 chunk 7/10: `encoded_jwt`, `access_token_expire_minutes`).
Payload `text` kept (public corpus; mutating 19k points + breaking `retrieve_candidates.py` not worth it; required for private
corpora). Own probe bug caught: the read-only upsert first hit a missing collection (404) and tested nothing; fixed.
`uv run pytest -q`: **699 passed, 1 xfailed**. Phase 18 must carry both keys and set `https=True` off-host.

**Provider migration decided 2026-10-18 (user):** OpenAI for most calls via a paid key, **hard $10 cap for the whole project**;
Groq/Gemini kept as fallbacks and independent second judges. Approved roles: generation gpt-4.1-mini (pending bake-off),
router/rewrite/entailment gpt-4.1-nano (router pending re-eval), judge gpt-5-mini (Batch API), second judge qwen on Groq,
RAGAS gpt-4.1-mini. Prices (official page, 2026-10-18, USD/1M in/out): 4.1-nano 0.10/0.40, 4.1-mini 0.40/1.60, 4o-mini
0.15/0.60, 5-nano 0.05/0.40, 5-mini 0.25/2.00; Batch -50%. Estimate for everything left: ~$2.40 (~$3-4 with buffer).
Qwen-era numbers stay as their own column, not overwritten.

**Unit 15g complete as of 2026-10-18 -- OpenAI provider + spend guard (step 0, code only)** (design record 15g,
`cost/ledger.py`, `cost/__main__.py`, `generation/llm/{openai_provider,chat}.py`, router/rewriter/entailment through
`chat_completion`, `factory` -> `default_client()` (openai -> groq -> gemini), `OpenAISettings`, `tests/conftest.py`).
SpendLedger reserves worst case (counted prompt tokens + max_tokens) inside an IMMEDIATE sqlite txn, refuses over
budget before contacting OpenAI; settles billed usage (cached input discounted); unpriced model -> KeyError. Over budget
-> generation falls back to Groq/Gemini; router -> provider_unavailable escalation; entailment fails closed. Reasoning
models get max_completion_tokens + reasoning_effort. `openai>=2.54,<3` declared (was transitive). New `rewrite_max_tokens=200`.
**Incident:** first suite run after the default switch made 1 real router call via two Groq-only-mocked tests, and my
debug rerun made a 2nd: **$0.000091 total, recorded by the ledger** (also the first real end-to-end proof, 420+9 tokens,
cost settled exactly). Fixed suite-wide: `tests/conftest.py` makes any real OpenAI client raise + uses a temp ledger;
later full runs left the real ledger at 2 calls. **Spend to date: $0.000091 of $10** (`uv run python -m aegis.cost`).
`uv run pytest -q`: **717 passed, 1 xfailed**. Must-do before step 4: route RAGAS/Batch clients through the ledger.
**Next:** step 1 generation bake-off (~$0.10), step 2 router re-eval (~$0.01), then 15h real-stack security batch.

**Unit 15g1 complete as of 2026-10-19 -- generation bake-off (step 1)** (design record 15g1, `experiments/15g1_generation_bakeoff.py`,
`docs/results/generation-bakeoff-openai{.md,-table.md,-raw.json}`). Phase 9 protocol exactly (same 20 seeded queries, asserted; v2;
temp 0.0; max_tokens 400); qwen's stored answers reused. **Judge held fixed:** every set validated by qwen-on-Groq (the 0.7296 judge)
AND gpt-4.1-nano (served). **Harness check: qwen answers + qwen judge reproduce 0.7296 exactly.** Entailment wrapper retries 429s
and raises instead of failing closed (0 errors, 0 unparseable, 88 calls/judge). **valid_citation_rate (qwen judge / nano judge):**
qwen 0.7296 / 0.6329; **gpt-4.1-mini 0.4486 / 0.4361** (Δ −0.281, paired-bootstrap 95% CI [−0.417, −0.140], lost 15/20);
gpt-4o-mini 0.3458 / 0.3264 (Δ −0.384 [−0.517, −0.253]); **gpt-4.1-nano 0.6665 / 0.6288** (Δ −0.063 [−0.209, +0.072], tie).
mini/4o-mini fail by **substantive uncited claims** (36/47 vs qwen 13; not a format issue, all use `[n]`); nano ties on the rate but
**24% of its cited claims are unsupported vs qwen 7.6%** (26 vs 6; spot-check: real decorative citations, one refusal citing all 10 blocks).
qwen abstains more (refusal language 10/20 answers vs 4-5). **Judge swap alone: qwen 0.7296 -> 0.6329 (−0.097)**, so qwen-era and
nano-judged numbers can't share a column. Cost/answer: mini $0.001765, 4o-mini $0.000618, nano $0.000434; p50 total 2,690 / 2,937 /
1,884 ms; TTFT mean 1,060 / 799 / 746 ms. Billed input 67,475 tokens/model = pre-run tiktoken count exactly. **Spend: $0.059732
(est. ≤$0.075); total $0.059823 of $10.** Untested confound: prompt fit (v2 was tuned on Groq models). Found: `09a`'s report
hard-coded "prompt v1" though v2 ran; fixed in script + both sweep result files. **Generator decision OPEN (user):** the served
default is still gpt-4.1-mini, the model that measured worse. No test changes (experiment script, I/O-bound).

**Unit 15g2 complete as of 2026-10-19 -- prompt fit (option C, user OK): NEGATIVE result** (design record 15g2, `prompts/v2c.py`,
`tests/test_prompts.py` (+1, red first), `experiments/15g2_prompt_fit.py`, raw `docs/results/prompt-fit-openai-raw.json`). v2c = v2 +
"no citation -> delete the sentence", "cite only a block that states the claim", no speculative advice, uncited gap sentence. Dev
gate: 10 seeded dev positives disjoint from the bake-off 20 (`Random(seed+1)`), v2 vs v2c x mini/nano, both judges. **v2c worse
everywhere (qwen judge):** mini 0.409 -> 0.305, nano 0.495 -> 0.371; supported claims 31 -> 18 / 30 -> 13; refusal-language answers
2-3/10 -> 9/10; output 265 -> 171 / 232 -> 125 tokens; mini substantive uncited ~24 -> ~23 (rule did not take); only gain mini
wrong-cite 0.31 -> 0.22. **Eval stage NOT run** (gate failed; saved ~$0.044). Found: the validator counts honest "context does not
cover X" sentences as missing_citation (~6/29 mini, ~6/10 nano under v2c; 9 for qwen in 15g1) -> Open Question #17, not changed.
Dev set is harder than the bake-off 20 (v2: mini 0.409, nano 0.495, nano wrong-cite 0.41). Disclosed leak: v2c was written from the
bake-off 20's failures. **Spend: $0.043726; total $0.103549 of $10.** v2 stays served; v2c registered, unserved.
`uv run pytest -q`: **718 passed, 1 xfailed**. **Generator decision back with the user.**

**Unit 15g3 complete as of 2026-10-19 -- gpt-5 generators (option D, user OK)** (design record 15g3, `15g1_generation_bakeoff.py
--extra-models gpt-5-mini,gpt-5-nano`; addendum in `generation-bakeoff-openai.md`). Same 20 queries / v2 / 400 cap / both judges;
prior answers reused. Smoke test first (1 query each): finish=stop, **reasoning_tokens=0 at effort=minimal**, so this tests the
models, NOT "reasoning helps". **gpt-5-mini 0.7158 (qwen judge) / 0.6242 (nano judge): Δ −0.014 [−0.148, +0.111], W/L 10/5 = tie with
qwen**; unsupported 6 (= qwen's) of 47 cited (share 0.128 vs qwen 0.076, gpt-4.1-nano 0.243); refusal answers 4/20; 2/20 at cap;
$0.001173/answer; TTFT 1,137 ms, p50 2,426 ms, 19th/20 4,374 ms. Caveat: 59 claims vs qwen 111 at similar citation-marker count
(162 vs 168): long multi-fact sentences, so its sentence-level rate is coarser. **gpt-5-nano 0.4217 / 0.3717** (Δ −0.308, loses 15/20;
7/20 truncated at the cap). **Spend $0.031937 (est. ≤$0.042); total $0.135486 of $10.** **User OK 2026-10-19: `OpenAISettings.generation_model = "gpt-5-mini"`** (order openai -> groq -> gemini unchanged).
Served cap stays GroqSettings.max_tokens=1000 (bake-off measured at 400, as qwen's was).

**Unit 15g4 complete as of 2026-10-19 -- router re-eval (step 2)** (design record 15g4, `experiments/15g4_router_reeval.py`,
`docs/results/router-reeval-openai{.md,-raw.json}`). 58 Phase 8 labeled dev queries, served prompt/settings; unparseable counted
separately (0 in every arm). **Phase 8's 0.6897 was llama-3.1-8b-instant (retired), not qwen** -> added qwen as free like-for-like arm.
**Accuracy: gpt-4.1-nano 0.6379, gpt-4.1-mini 0.7759, qwen 0.7931.** **Billing (the only route acted on, policy.py): 8/8 recall, 0/50
false escalations in every arm** (Wilson 95% lower bound 0.68: thin). nano's errors: known_issue recall 0.20 (-> docs_howto); McNemar
mini vs nano p=0.057, qwen vs nano p=0.035. docs_howto vs known_issue has no downstream effect today -> **router stays gpt-4.1-nano**
(mini is 4x/call, $0.000325 vs $0.0000814); switch if a later phase uses the route for retrieval. **Spend $0.023569 vs my estimate
<=$0.02 (overrun: estimated from one 420-token call; real queries avg 779 tokens, max 2,399); total $0.159055 of $10.**
**Next: 15h real-stack security batch (needs cost estimate + user OK).**

**Unit 15h complete as of 2026-10-22 -- real-stack security batch on gpt-5-mini** (design record 15h,
`experiments/15h_{realstack_security,claim_split_sensitivity}.py`, `data/security/poison_chunks_v1.jsonl` (5 payloads, canary each),
raw `docs/results/security-realstack-raw.json`). User OK: est ~$0.23, stop $0.45, Langfuse forced off (checked at runtime), rule fixed
before the run: serve v3 iff (a) poisoned obedience v3 <= v2 and (b) H2 CI upper bound >= 0. **H1 (poison at rank 1, in memory, 10
queries x 5 payloads):** canary hits hand-labelled; **obeyed v2 5/50 (typosquat `pip install fastapi-securefix` 4/10, whole-answer
HACKED 1/10), v3 0/50**; no paraphrased obedience in a scan of the 89 non-hits; commitment payload never repeated (check fired 0, not
exercised end to end). **All 4 obeyed install claims scored SUPPORTED_LEXICAL** (validator certifies poisoned advice -> OQ #20). **H2:**
v3 vs v2 gpt-5-mini, served splitter +0.113 [−0.021, +0.268] (qwen judge) but claims 59 -> 34: **splitter doesn't split bullets** (OQ #18);
line-aware re-score (measurement-only patch) **+0.068 [−0.039, +0.175]**, nano judge +0.068 [−0.045, +0.168]; 6/20 v3 answers at the 400
cap. **H3 (served pipeline, one cache per version -- exact-cache key has no prompt version, OQ #19):** paraphrases 13/14 escalated
(6 billing route, 7 gate); wrapped 0/14 escalated; **v2 complied 1/14 (whole v2 system prompt leaked, translated to French), v3 0/14**
(1 refused without answering). **Rule (a) and (b) pass -> `GenerationSettings.prompt_version = "v3"`**; A3 strict xfail went red as
designed, marker removed. **Spend $0.185355 (H2 $0.028559, H1 $0.119449, sensitivity $0.001633, H3 $0.035714); total $0.34441 of $10.**
`uv run pytest -q`: **719 passed, 0 xfailed**. Docker was down at session start (started Docker Desktop + Qdrant). **Next: 15i wrap-up.**

**Unit 15j complete as of 2026-10-22 -- cache scope carries the answer configuration (OQ #19)** (design record 15j,
`generation/pipeline.py` `cache_fingerprint`, `pipeline.py` `_cache_version` at all 4 cache get/set sites, `tests/test_pipeline.py` +3,
red first). Scope = `corpus_version::prompt=<v>::llm=<provider:model,...>`, read per request. Served:
`prompt=v3::llm=openai:gpt-5-mini,groq:qwen/qwen3.8-27b,gemini:gemini-3.5-flash`. Also covers the generator switch (same bug). Every
existing cache entry is now unreachable (correct: none came from the served config). `uv run pytest -q`: **722 passed**. $0.

**Unit 15k complete as of 2026-10-22 -- claim splitting on paragraphs + list items (OQ #18)** (design record 15k, `citations.py`
`_prose_blocks`, `tests/test_citations.py` +4 (3 red first; hard-wrap guard), `experiments/15k_rescore_claim_split.py`, raw
`docs/results/claim-split-rescore-raw.json`). Plain newline deliberately not a boundary (hard-wrapped sentences). Re-scored stored
answers, no generation, **$0.006259**: qwen judge qwen 0.7296 -> **0.7539**, gpt-4.1-mini 0.4441, gpt-4o-mini 0.3490, gpt-4.1-nano 0.6665,
**gpt-5-mini v2 0.7217, v3 0.7842**, gpt-5-nano 0.4514 (nano judge in raw). **Both decisions hold:** gpt-5-mini − qwen −0.032 [−0.160,
+0.084]; v3 − v2 +0.063 [−0.048, +0.172] (rule (b) passes). Non-list models move ±0.005. Phase 9-11 numbers stay historical (old
splitter). `uv run pytest -q`: 726 passed.

**Unit 15l complete as of 2026-10-22 -- risky-advice check (OQ #20)** (design record 15l, `guardrails/risky_advice.py`,
`RiskyAdviceSettings`, `ClaimStatus.UNVOUCHED_RISKY_ADVICE`, `CitationValidator(vouched=)`, pipeline warnings + never cached,
`GuardrailWarning`, `app.js` alert, `tests/test_risky_advice.py` +18, `test_pipeline_stream.py` +1, `experiments/15l_risky_advice_fp.py`,
probes `data/security/risky_advice_probes_v1.jsonl` (sha256 8ba1ac49...), raw `docs/results/risky-advice-fp-raw.json`). Install of a
package the DOCUMENTATION corpus never mentions (or URL/VCS target), link to an undocumented host, or a request for the user's secret
-> never validated. Vouched = tokens (11,728) + hosts (131) of documentation chunks; missing corpus -> fail closed. **Numbers ($0):**
probes 12/12 risky, 0/12 legit; 15h real obeyed install claims **3/4 at frozen patterns** (missed "install the patch package X") -> widened
prose pattern (disclosed, not held-out) -> 6/6 securefix sentences; real answers 0/1,162 fired (**weak: 0 install sentences, 6 links, 10
secret nouns in them**); documentation 0/14,824 (circular for packages, real for secret requests). Residuals: import-name typosquats,
insecure-config advice, mentions fire too (safe direction). `uv run pytest -q`: **745 passed**. **Spend total $0.350669 of $10.**
**Next: 15i wrap-up.**

**Unit 15i complete as of 2026-10-22 -- Phase 15 wrap-up. PHASE 15 COMPLETE.** All 4 plan.md exit criteria met, evidence table in
`docs/results/security-assessment.md` ("Exit criteria"): (1) `tests/security/` 31 tests, 0 xfail (+A9 added this unit, verified to
discriminate: with the package vouched the same sentence returns SUPPORTED_LEXICAL); (2) injection before/after: poisoned obedience
5/50 -> 0/50, direct 1/28 -> 0/28, A3/A4/A9 closed, with residuals stated (A8 gate passes wrapped injections; A7 input detection not
a defense; fallbacks untested against poison; insecure-config advice + import-name typosquats uncovered); (3) tenant bypass:
structural fail-closed tests + **re-run of `15b_partition_bypass.py` after 15j-15l: A1 0/1,800, 0/760 out-of-partition**; (4) assessment
finalized. `15-rag-security.md` §8 = phase-end summary by the five attack classes (unit-1 text kept unedited). Quick Status corrected
(was stale: "Phase 15 not started", quota blockers). `uv run pytest -q`: **746 passed**. $0 this unit; **project spend $0.350669 of $10.**
**Next session: Phase 16 (CI evaluation gate).** Read plan.md Phase 16 first; carried-over gaps: Phase 13 p95 (re-measure on gpt-5-mini),
Phase 11 κ, step 4 re-measurement (RAGAS/Batch must be wired through the ledger before any OpenAI run).

**Unit order (user OK 2026-10-18):** 15b tenancy (done) -> 15c HTTP auth/ownership/limits (done) -> 15d input boundary + evasion (done)
-> 15e injection defenses (unforgeable delimiters, commitment check, prompt v3) -> 15f Qdrant/vector (done) -> 15g real-stack
batch (quota; ~6-8k tokens/request estimated, measure one first) -> 15h assessment + final record. Honest estimate:
~2 days vs the plan's 1/2 day.

## Phase 14 — API, Streaming & UI

**Scope** (plan.md, Days 16-17, topic 35): FastAPI `POST /chat` (SSE) / `GET /health` / `GET /trace/{id}` /
`POST /feedback`; typed SSE events (`token`, `citation`, `escalation`, `done`, `error`); conversation memory
wired to Phase 8's rewriting; static HTML+vanilla JS UI with a visibly distinct escalation state; feedback ->
review queue; rate limiting / validation / graceful provider failure. Exit: streaming works in a browser,
citations clickable to real source text, escalation visibly different, multi-turn resolves, feedback persists.

**Unit 1 complete as of 2026-10-01** (design record 14, `src/aegis/events.py`, `src/aegis/pipeline.py`,
`tests/test_pipeline_stream.py`, `experiments/14a_stream_smoke.py`). `AnswerPipeline.answer_stream()` yields
typed events; contract pinned by tests (answer `token+ citation* done`; cache hit same shapes; escalation
`escalation done`; failure `token* error`, terminal). **Decision (user-confirmed via `AskUserQuestion`): stream
live, validate after** -- not less safe than today, since the non-streaming pipeline already serves answer text
whether or not every claim validates. One implementation (`_run`) with an optional event sink; no sink = original
non-streaming path, so evals/tests are unaffected. Work runs on one worker thread behind a queue so OTel spans
and the trace-id ContextVar never straddle threads. `error` events carry only the exception class + fixed
message (provider exceptions can include org ids). Cache entries now store `CitedChunk` so a hit replays real
`citation` events (field defaults to `[]`; old entries still load). `uv run pytest -q`: 508/508 (+12). **Real run
(n=1):** first token 1,298 ms, 85 token events over 252 ms, `citation` at 1,921 ms (371 ms after the last token),
`done` 1,927 ms; cache replay of the same query 3 ms. **Streaming buys less than the plan assumes:** the model
emits the whole answer in ~250 ms, so the first-token wait is the pre-generation path (routing + retrieval +
confidence), not generation. Not verified: real mid-stream failure, client disconnect, a real browser.

**Unit 2 complete as of 2026-10-01** (design record 14b, `src/aegis/api/app.py`, `src/aegis/api/main.py`,
`src/aegis/factory.py`, `ApiSettings`, `tests/test_api.py`). `POST /chat` streams the typed events as SSE
(hand-written frames over `StreamingResponse`; no SSE library), `GET /health`, pipeline injected into
`create_app()`, real pipeline built + warmed in the lifespan. Validation failures are HTTP 422 before the pipeline
runs; pipeline failures are a terminal `error` event under HTTP 200. Deps added: `fastapi`, `uvicorn`, dev `httpx`.
`max_query_chars=20000`, set from the measured golden-query lengths (max 16,755; a 2,000 cap would reject 43/180).
`uv run pytest -q`: 513/513 (+5). **Real server, real HTTP (small n):** first token 965-1,824 ms on 3 fresh
generated answers; 3 escalations returned `escalation`+`done`; exact-cache repeat 7 ms; blank query 422; client
disconnect after first token left the server healthy but the pipeline ran to completion (no cancellation).
**Unresolved contradiction:** one fresh query had a 10,023 ms confidence stage and one request spent 28.7 s in
`validation` (tier-2 LLM call, cause not checked) -- same unexplained variance as Phase 13, still threatens p95 < 3 s.
Experiments were not re-pointed at the factory (they record what produced published numbers).
Not verified: browser consumption, concurrent requests, mid-stream provider failure through HTTP.

**Unit 3 complete as of 2026-10-01** (design record 14c, `src/aegis/api/{app,store,audit_service}.py`,
`tests/test_api.py`, `tests/test_interaction_store.py`). `GET /trace/{id}` wraps the Phase 13 audit trail + evidence
(id must be 32 hex chars); `POST /feedback` records up/down + comment against a `trace_id`. `/chat` now records
each completed interaction (query, answer text shown, escalated, cache status) server-side in sqlite, so a vote
is accepted only for a trace the server issued (404 otherwise) and the feedback -> golden-candidate loop does not
depend on Langfuse's 30-day retention or on client-supplied text. Second vote replaces the first. Streams that
end in `error` are not recorded. `uv run pytest -q`: 524/524 (+11). **Real run (n=1):** vote recorded 200,
unknown trace 404, `/trace` on a fresh trace 200 in **24.8 s** (soft 20 s cap; ingestion lag) with all 6 stages,
no warnings/inconsistencies, 10/10 evidence `ok`; vote survived a server restart. Open: `/trace` and `trace_id`
are unauthenticated and `/trace` returns query + chunk text (Phase 15 surface); no rate limit yet; review
queue / golden-candidate export not built.

**Unit 4 complete as of 2026-10-01 -- wiring done, "resolves correctly" NOT demonstrated** (design record 14d,
`src/aegis/pipeline.py`, `src/aegis/api/app.py`, `src/aegis/query/rewrite.py`, `experiments/14b_multiturn_real.py`).
`POST /chat` takes optional `history`; Phase 8's `QueryRewriter` runs as the first stage so cache/router/retriever
see the standalone question; `done` carries `rewritten_query`; `rewrite` is its own trace span; audit trail and
interaction store record both texts. No history = no rewriter call. Rewriter API errors / blank output now logged.
Client holds history (server stateless); window 6 msgs x 1,000 chars (unmeasured guard rails). `uv run pytest -q`:
539/539. **Real HTTP:** follow-up + history -> standalone query, answered, rewrite stage 856 ms; same follow-up with
no history escalated. **n=6 end-to-end retrieval check:** hit@10 A(no history) 2/6, B(user turn) 1/6, C(user turn +
real answer) 0/6; escalations 4/6 -> 1/6 -> 1/6. Rewriting converts abstentions into answers but retrieval gain is
not shown at this n and regime C (assistant answer in history) is unmeasured at scale -- Phase 8's +52.7% NDCG@5
covers regime B only. Exit criterion "multi-turn follow-ups resolve correctly" stays OPEN.

**Unit 5 complete as of 2026-10-01** (design record 14e, `src/aegis/api/static/{index.html,app.js,sse.js,style.css}`,
`tests/test_ui_static.py`). Static page served at `/`: fetch+stream-reader SSE client (incremental parser, node-tested
with a split frame), validated `[n]` -> clickable chip expanding the real source chunk, unvalidated `[n]` struck
through, escalation as a distinct amber card, history sent + "Searched for" shown, 👍/👎 (+comment) -> `/feedback`,
time-to-first-text per answer; model/corpus text only via `textContent` (test asserts no innerHTML); bold/code
rendered from DOM nodes. `uv run pytest -q`: 542/542. **Verified in a real browser (Playwright, n=1 session):**
streaming + 15 citation chips, chip click opened `doc::tutorial/body.md` text, escalation card (screenshot viewed),
👎 persisted with rewritten query. **Findings, not fixed:** (1) the follow-up "what about making it required?" was
rewritten correctly but served as a **semantic cache hit** on the earlier generic "declare a query parameter"
answer -- the 0.88 threshold didn't separate the pair; (2) cold first request: first text 22.7 s -- cache_lookup
9.3 s (cache path not covered by warm-up) + generation 17.9 s on Gemini (why Groq wasn't used unknown; fallback
logs nothing). Not checked: dark mode, mobile, a11y, mid-stream failure in browser, escalation wording is
router-internal. Phase 14 exit: streaming ✔, clickable citations ✔, escalation distinct ✔, feedback persists ✔,
**multi-turn correct: open** (wiring works; regime-C eval owed; cache-collision example above).

**Unit 6 complete as of 2026-10-17** (design record 14f, `src/aegis/api/ratelimit.py`, `api/app.py`,
`generation/llm/fallback.py`, `router/classifier.py`, `guardrails/citations.py`, `pipeline.py`, `events.py`,
`static/app.js`, `experiments/14c_provider_failure_real.py`). Plan build item 6. **(1) Rate limit:** hand-rolled
per-client-IP token bucket on `/chat` (user chose it over `slowapi`), 429 + `Retry-After`, burst 5 / 6 per min
(guard rails sized from Groq calls per request, not tuned), checked after body validation so 422s cost nothing.
**(2) Outage -> escalation (user decision, `AskUserQuestion`):** when the router or generation LLM fails on every
provider before any text is shown, the request escalates with new `EscalationEvent.cause="provider_unavailable"`
(closed set: `policy | low_confidence | provider_unavailable`; UI explains each). The router now raises
`RouterUnavailableError` on API errors instead of returning a disguised `billing_account_legal` (closes Phase 13 open
item 4); unparseable model output still fails safe to billing. Mid-stream failure stays a terminal `error`.
**(3) No silent failures:** `llm.provider_failed` / `llm.failover` / `citation.entailment_api_error` /
`api.rate_limited` log events. `uv run pytest -q`: **563/563** (+21; Qdrant up). **Real, n=1 each (invalid keys
to force real SDK errors):** failover Gemini 400 -> Groq answered (1.0 s); router outage -> `provider_unavailable`
escalation, no retrieval (162 ms); generation outage (Groq 401 + Gemini 400) -> escalation after retrieval in
1.0 s, not cached. **Real HTTP:** 6 x 200 (1 miss 11.1 s + 5 exact hits; the slow miss refilled the bucket), then
3 back-to-back -> 429 `Retry-After: 1`, `/health` fine. Bug caught by the tests: the limiter defines `__len__`, so
`limiter or default` silently replaced an injected empty limiter. Also fixed doubled logger names (`aegis.aegis.*`).
Not handled / flagged (14f SS6): SDK-internal 429 retries could make failover slow (unmeasured); a tier-2
entailment outage still serves the answer with all citations struck through; `08a_router_eval.py` now stops on an
outage (intended, no checkpointing); proxy-aware keying -> Phase 15.

**Unit 7 complete as of 2026-10-17 -- all 6 Phase 14 build items now built** (design record 14g,
`src/aegis/eval/review_queue.py`, `api/store.py`, `tests/test_review_queue.py`). Plan build item 5's second half:
voted interactions -> review queue (👎 first) -> terminal review (positive + stratum / negative + category / reject)
-> `--export` to `data/golden/candidates_feedback.jsonl` (`split="candidate"`, golden-set schema validated, full
provenance). Frozen golden set never touched. **Guards:** system citations are NOT exported as suggested labels
(12e label-bias finding); every candidate checked against dev AND holdout (normalized exact text) -> `duplicate_of`;
follow-ups export their rewritten standalone text. Terminal tool, not an API page (no auth until Phase 15).
`uv run pytest -q`: **576/576** (+13). **Real end-to-end** (real server, throwaway DB): 2 real HTTP votes + 1 real
holdout query -> queue ordered 👎 first -> exported 3; holdout item flagged `duplicate_of golden::issue::11134
(holdout)`. **Findings:** the first real 👎 was on a *correct* billing hard-escalation ("I just wanted to know who to
contact") -- a vote is not a label, rules out auto-promotion; the text-only stratum heuristic agrees with the frozen
strata on only **138/180**, so it is a hint, not an Enter-to-accept default. The real store has 0 votes so far:
the loop is built and verified, not yet fed. Labeling a v2 needs `label_cli` path changes (not done).

**Unit 8 complete as of 2026-10-17 -- both 14e findings root-caused and fixed** (design record 14h,
`experiments/14d_semantic_cache_short_queries.py`, `config.py`, `factory.py`, `pipeline.py`, `embedding/encoder.py`,
`retrieval/dense.py`). **Collision:** recovered the real 14e texts from Langfuse (the 14e write-up's guess omitted
"in FastAPI"): "How do I make a query parameter required in FastAPI?" vs cached "How do I declare a query parameter in
FastAPI?" = **0.9354** (> 0.88). Cause: the rewriter's shared "How do I ... in FastAPI?" template dominates short
embeddings; Phase 12's sweep used long issue-body negatives. First hypothesis (short queries collide in general)
refuted: 11,175 real short-title pairs max 0.8507. **Decision (user, `AskUserQuestion`): semantic tier OFF by default**
(`CacheSettings.semantic_cache_enabled=False`; code/tests kept; worth 0.2-1.6% of simulated traffic and a hit skips
routing/confidence/validation). Addendum added to `docs/results/cache-performance.md`. **Cold start:** two stacked
causes. (1) warm-up never loaded bge-large -- its query embedding was in the on-disk cache (0.1 ms, no model load);
first real query paid 10.5 s. Fixed with `Encoder.warm_up` (bypasses cache both ways). (2) the first CPU encoder pass
after the cross-encoder's first MPS run costs ~1.3-1.7 s once (mechanism not established) -- fixed by warming the
encoder last. Refuted along the way: per-thread init, embedding-cache I/O, Langfuse span cost. **Real server:** first
request `cache_lookup` 8,447-8,669 ms -> **166 ms**; replayed 14e conversation: follow-up now a miss with a specific,
correct answer; cold first request stages ~2.0 s (n=1) vs 22.7 s first text in 14e. Startup ~25 s (all model loads,
now logged as `api.warm_up`). `uv run pytest -q`: **581/581** (+5). Possibly relevant to Phase 13's unexplained
confidence variance (same order-dependent CPU/MPS shape) -- not tested.

**Unit 9 complete as of 2026-10-17 -- PHASE 14 COMPLETE** (design record 14i, `experiments/14e_multiturn_regime_c.py`,
raw `docs/results/multiturn-regime-c-raw.json`). Regime-C eval, all 24 items of the Phase 8 multi-turn set (user chose
to spend the qwen quota today; Phase 9 sweep slides a day). Phase 8's protocol (exclude source issue, ranx) + served
path (router + confidence). **Leg A reproduced Phase 8 exactly** (0.1995/0.0897/0.2861). NDCG@5 / R@10 / MRR /
escalated: A 0.1995/0.0897/0.2861/12 of 24; B 0.2805/0.1103/0.4347/9; **C 0.3116/0.1246/0.4816/8**. Paired bootstrap
NDCG@5: C-A +0.1121 [+0.0220, +0.2110]; C-B +0.0312 [-0.0004, +0.0664]; B-A +0.0810 [-0.0025, +0.1715].
**Pre-registered criterion MET** (thin: C-B lower bound -0.0004). Where turn1 was answered (n=14) C >= B on all 14.
Caveats: synthetic set; 8/24 follow-ups still escalate; Phase 8's regime B only partly reproduced (0.2805 vs 0.3045,
rewriter model changed) and B-A is not significant at n=24.

**Phase 14 exit criteria: all 5 met** -- streaming in a browser (14e), clickable citations to real source text (14e),
escalation visibly distinct (14e; now with a cause-specific explanation, 14f), multi-turn resolves (14i, pre-registered,
thin margin), feedback persists (14c) and feeds a review queue (14g). All 6 build items built.

**Carried to Phase 15:** `/chat`, `/trace`, `/feedback` unauthenticated; `/trace` returns query + chunk text and holds a
worker up to 20 s; rate limit keyed on socket peer (no trusted proxy); feedback candidates hold raw user text.
**Carried, other:** semantic cache off pending a templated-negative re-tune (14h SS7); tier-2 entailment outage serves
answers with all citations struck (14f); Phase 13 p95 < 3 s still unmet (14h's CPU/MPS first-pass finding may bear on it).

**Deferred-items run, 2026-10-18.** (1) **Phase 11 judge kappa, paired N=23/29 under the fixed rubric: kappa = 0.1321**
(raw agreement 0.6522); trend 0.2143 (N=11) -> 0.1772 (N=13) -> 0.1321 (N=23) -- falling, not converging. Gemini gave
22/23 a 5; all 8 disagreements are Groq 4 / Gemini 5. Gemini's ceiling effect survives the rubric fix, so this judge pair
cannot validate the Groq judge regardless of N; needs a varying rater (human labels or another model) -- user decision,
not changed. First attempt lost ~13 calls to a 5 requests/minute Gemini cap (not the daily one); added 13 s pacing to
`11g`. 6 rows remain (daily cap). `docs/results/judge-agreement.md` updated. (2) **Phase 9 sampling sweep: grid
complete (4/4).** valid_citation_rate: t=0.0 0.7296 (both top_p, identical), **t=0.5 0.7350** (n=20; `unsupported`
6 -> 11, `missing_citation` 22 -> 13 -- same rate, more wrongly-cited claims), **t=1.0 0.6546** (n=17; last 3 queries hit
qwen's 200k TPD cap). One run/config, no CI. **Decision unchanged: temperature=0.0.** Found and fixed: the script's
regenerated report had overwritten the hand-written analysis in `sampling-sweep.md` (restored from git `7e3c6c6` and
merged); its generated table now goes to `sampling-sweep-table.md`. qwen TPD is exhausted for today.

**Next action:** Phase 15 -- Security Hardening (plan.md). Remaining quota-bound item: Phase 11 judge kappa, 6 Gemini rows (rerun `11g` tomorrow);
the judge-pair problem itself needs a user decision (see 2026-10-18 note).

### Phase 11 wrap-up (2026-09-26)

Everything code-, analysis-, and documentation-side that doesn't require new LLM quota is complete: all 6
build units, the judge rubric fix (groq-side verified at N=26), the faithfulness root cause (two real
mechanisms found, recommend closing as explained/disclosed), and half the sampling-sweep grid (2 of 4
configs real). Two items remain, **owned by the user, not carried as this project's own next-session
task**:

1. **Real paired inter-judge κ under the fixed rubric** — Gemini has 0/29 real scores across two attempts
   (both quota-blocked). Run `uv run python experiments/11g_judge_rubric_rescore.py` once Gemini's 20/day
   quota allows (checkpoint-resumable — only fills in the missing scores), then update
   `docs/results/judge-agreement.md` with the real outcome.
2. **The remaining 2 sampling-sweep configs** (temperature 0.5 and 1.0, both top_p=1.0) — run `uv run
   python experiments/09a_sampling_sweep.py` once `qwen/qwen3.8-27b`'s TPD quota resets (now
   config-resumable — safe to just re-run, won't overwrite the 2 real completed configs).

Both scripts will pick up and report real numbers automatically; no further code changes are needed for
either. Open Questions #1 and #16 and the DoD tracker rows below reflect this exact state.
**Status:** 🟡 Phase 11 started. Scope agreed with the user (brainstorming session), **revised 2026-09-25** (see Decision Log — user will not hand-label): 6 units — (1) RAGAS LLM/embeddings adapter wiring, (2) Cohen's κ utility (judge-vs-judge, no hand-labeling CLI), (3) LLM-as-judge built twice (Gemini + `openai/gpt-oss-120b`), (4) real N=30 batch run (generation → RAGAS → both judges → inter-judge κ), (5) custom metrics (citation validity aggregation, escalation appropriateness, code-block static-syntax check), (6) `make eval` scorecard. Real decisions locked in with the user: **N=30** shared answer batch (not plan.md's literal 50 — traded statistical strength for same-day feasibility given the quota math below), **Gemini `gemini-3.5-flash`** as the starting judge model (different vendor from the Groq generator, deliberately avoiding self-preference bias), **static syntax validity (`ast.parse`)** for the code-block-correctness metric (not sandboxed execution — real security-exposure and engineering-time tradeoff, Phase 15 doesn't exist yet), **inter-judge agreement replaces judge-vs-human-label κ** (2026-09-25 — user declined to hand-label; see Decision Log for the honest cost of this substitution).

**Unit 1 complete as of 2026-09-25** (design record 11, `docs/results/groq-model-replacement.md`, `docs/results/ragas-smoke-test.md`). Two real, sequential blockers resolved in one session: (1) the Groq model-retirement problem carried over from 2026-08-18 — measured `openai/gpt-oss-120b`'s real reasoning-token tax at answer length (survivable, 25-45% of completion tokens, but 16.7s mean latency nearly matches Gemini's already-rejected 21.4s), found last session's `groq/compound-mini` pick had *also* died in the intervening five weeks, and measured a fresh candidate (`qwen/qwen3.8-27b`) clean — 0 reasoning tokens, 8.7s mean latency, one real free-tier constraint (1000 output-tokens/minute cap) found and fixed. User decided (`AskUserQuestion`): `qwen/qwen3.8-27b` for all five Groq-backed config roles, applied to `config.py`. (2) Once Groq worked again, the real Unit 1 smoke test surfaced a second, unrelated problem: `eval/ragas_adapters.py` (written last session) targeted the wrong RAGAS LLM/embeddings protocol — `Faithfulness`/`ResponseRelevancy` need `BaseRagasLLM`/legacy `BaseRagasEmbeddings`, not the `InstructorLLM`/modern-`BaseRagasEmbedding` interfaces `llm_factory` itself also returns. Rewrote both adapters as hand-written `BaseRagasLLM`/`BaseRagasEmbeddings` subclasses wrapping the raw `groq`/`google-genai` SDKs (no new dependencies — the `langchain-core` types needed were already a transitive cost of the `ragas<0.4` pin). Real smoke test now passes end-to-end: `faithfulness=0.8750`, `response_relevancy=0.7820`, `id_based_context_precision`/`recall`=`0.8000`, one real query. `uv run pytest -q`: 373/373 passed throughout.

**Unit 2 complete as of 2026-09-25** (design record 11b, `src/aegis/eval/agreement.py`). `cohen_kappa`/`raw_agreement_rate` -- thin wrapper over `scikit-learn`'s `cohen_kappa_score` (added as an explicit direct dependency; was previously only transitive), TDD'd against a hand-computed textbook example (Wikipedia's Cohen's kappa worked example: kappa=0.40, raw_agreement_rate=0.70, both exact matches), plus edge-case tests (perfect agreement, chance-level agreement, mismatched-length guard, quadratic weighting for a future ordinal rubric). No project data yet -- this unit is the reusable statistics primitive; Unit 3 produces the actual label sequences. `uv run pytest -q`: 380/380 passed (+7 new).

**Unit 3 complete as of 2026-09-25** (design record 11c, `src/aegis/eval/judge.py`, `docs/results/judge-smoke-test.md`). `Judge`/`HelpfulnessJudgment` (single 1-5 holistic score + required reasoning, real `instructor` structured output — not free-text parsing), `build_groq_judge`/`build_gemini_judge` reusing Unit 1's `InstructorLLM`-based builders for the use case they were actually scoped for. TDD'd with a fake LLM (9 tests: prompt formatting, score-bounds validation, the "honest refusal scores high" prompt-instruction regression guard). Real two-judge smoke test on one real generated answer: `openai/gpt-oss-120b` scored 4/5 (1,647ms), Gemini scored 5/5 (17,225ms) — both real structured output, close but not identical. Two more real bugs found and fixed en route: `instructor` needed its `google-genai` extra (`jsonref`) added explicitly (was transitive-only, same gap `scikit-learn` had before Unit 2), and a real transient Gemini `503 UNAVAILABLE` (already-documented failure mode, Phase 9 bake-off) resolved on retry with no code change. `uv run pytest -q`: 389/389 passed (+9 new).

**Unit 4 complete as of 2026-09-25 — a real, load-bearing negative result** (design record 11d, `docs/results/judge-agreement.md`, raw: `docs/results/judge-agreement-batch-raw.json`). Ran the real N=30 batch (`experiments/11e_judge_validation_batch.py`, generation → RAGAS → both judges). Found and fixed a third real bug before it would run at all: `GroqRagasLLM` was reusing `settings.groq.max_tokens=1000` (`qwen/qwen3.8-27b`'s real OTPM ceiling, Unit 1), too tight for RAGAS's own statement-generation/entailment calls (`LLMDidNotFinishException`). Decoupled it (`JudgeSettings.ragas_llm_max_tokens=4096`) — then found `openai/gpt-oss-120b` (the new RAGAS-scoring model) has its *own* 200,000 tokens/day cap, which collapsed under combined RAGAS+judge load; split onto a second, independent-quota model (`openai/gpt-oss-20b`) for RAGAS scoring, keeping `gpt-oss-120b` judge-only. **Headline result: inter-judge Cohen's κ = -0.0161 (unweighted) / 0.0906 (quadratic), N=9 paired — at or below chance agreement, well under plan.md's 0.6 bar.** Two distinct, real, well-evidenced mechanisms: Gemini's scores have almost no variance (10/11 real successes were a perfect 5/5 — a ceiling-effect leniency bias); `openai/gpt-oss-120b` systematically scores honest "I don't have enough information" refusals low (1-2) despite an explicit rubric instruction to score them high, while Gemini reliably follows that same instruction — the more consequential finding, since a judge that penalizes honest abstention would reward this project's own North Star failure mode if used to tune the system. RAGAS `faithfulness` mean 0.7977 (n=20/29, real, below the 0.90 target, not root-caused this session). Every model used for judging (`qwen/qwen3.8-27b`'s ITPM, `gpt-oss-120b`'s TPD, `gpt-oss-20b`'s TPD, Gemini's 20-requests/day) hit a real, hard quota wall by the end of the batch — one query (`golden::issue::2600`) failed generation entirely when Groq and Gemini were both down simultaneously. `uv run pytest -q`: 402/402 passed throughout (no new tests this unit — the batch script is I/O-bound/integration-only).

**Unit 5 complete as of 2026-09-25** (`src/aegis/eval/custom_metrics.py`, 13 new tests). Built while Unit 4's batch ran in the background, deliberately isolated from it (zero LLM calls, can't collide with the live quota walls). `citation_validity_summary` (aggregates `CitationValidator` results across a batch), `escalation_appropriateness` (TP/FP/FN/TN + precision/recall/deflection_rate over escalate/is_negative pairs — same convention as Phase 10's real confusion matrix; one test reproduces Phase 10's exact shipped numbers, TP=27 FP=24 FN=3 TN=126, as a regression guard), `code_block_syntax_validity` (`ast.parse` on Python-tagged-or-untagged fenced code blocks, explicit non-Python tags skipped rather than guessed at — grounded in a real corpus fence-tag count: 7,906 untagged blocks is the single largest group). Verified against real generated answers from the Unit 4 checkpoint, not just synthetic test strings. `uv run pytest -q`: 402/402 passed (+13 new).

**Unit 6 complete as of 2026-09-25** (design record 11e, `src/aegis/eval/scorecard.py`, `Makefile`). `make eval` now runs `python -m aegis.eval.scorecard` and prints one consolidated report — RAGAS means, inter-judge κ (Unit 2), code-block syntax validity + mean citation-validity rate (Unit 5), all checked against plan.md's Phase 11 exit criteria — pure aggregation over Unit 4's already-written batch rows, zero new LLM calls, so it runs regardless of today's quota exhaustion. Real, honest output: 2 of 4 exit criteria fail (`faithfulness=0.7977 < 0.90`, `kappa=-0.0161 < 0.6`), 2 pass (custom metrics implemented, the command itself works). Also fixed a latent bug found along the way: `make eval` was silently a no-op before this unit — `aegis/eval/harness.py` (the Makefile's prior target) has no `__main__` block. Scoped deliberately to Phase 11's own build list (RAGAS + judge + custom metrics), not a combined retrieval+generation scorecard — that's Phase 19's job. TDD'd (14 new tests, including a regression guard reproducing the exact numbers in `docs/results/judge-agreement.md` from the real raw batch file). `uv run pytest -q`: 416/416 passed (+14 new).

**Judge rubric fix applied and spot-checked as of 2026-09-25** (design record 11f, `docs/results/judge-rubric-fix-spot-check.md`). Rewrote `JUDGE_PROMPT_TEMPLATE`: the honest-refusal rule moved to a `CRITICAL RULE` block before the numeric scale, explicitly naming and preempting the exact override language `gpt-oss-120b`'s real reasoning text had used ("does not provide concrete guidance", "leaves the problem unresolved"); added a ceiling-effect anchor ("reserve 5... identify what a 4 version would be missing") targeting Gemini's near-zero variance, without going multi-dimensional (design record 11c's rejection of that still holds). TDD'd first (5 new prompt-content tests, RED confirmed against the old template before implementing). **Real, live spot-check against quota confirmed the groq-side fix works**: re-scored the exact 4 real disagreement cases from Unit 4 with the new rubric — `gpt-oss-120b`'s scores moved from `[1,2,1,1]` to `[5,5,5,4]`, an exact match to Gemini's original scores on the same 4 rows; 3 real non-refusal cases were unchanged (no overcorrection). Both providers' quotas were exhausted again mid-check (`gpt-oss-120b`: 199,157/200,000 TPD; Gemini: 20/day, only 1 call completed) before Gemini's own side of the fix or a full N=30 re-run could be verified. `uv run pytest -q`: 420/420 passed (+4 new prompt-content tests in `test_judge.py`).

**Full re-score attempted 2026-09-26, groq side confirmed at N=26, Gemini still fully blocked.** Checked quota first (real trivial calls): Groq had reset, Gemini had not (still 429, a transient 503 turned out to be a one-off blip, not the quota actually clearing). Wrote `experiments/11g_judge_rubric_rescore.py` — reuses Unit 4's real already-generated answers, re-runs only the two judges (not generation/RAGAS, since neither changed), checkpointed per-row same as Unit 4. **Real result: 26/29 groq scores collected (3 hit `gpt-oss-120b`'s TPD cap again, re-exhausted by this very run); Gemini 0/29 (its 20/day cap did not clear this run).** All 5 real refusal cases among the 19 rows with both an old and new score moved up as intended (e.g. `1->5`, `2->4`), matching Gemini's original stance on those rows — the targeted fix holds at fuller scale. **New, real, honest observation not seen in the smaller spot-check:** mean moved 3.10->4.31 and the distribution compressed toward the top (24/26 now score 4-5, vs. 12/20 before) — a milder version of the same ceiling-effect shape Gemini already had, not the opposite problem; 2 non-refusal answers moved down (5->4), showing the "reserve 5" anchor is discriminating in the intended direction on at least some cases, but 1 moved up with no clear refusal-related cause. That last case is also a **real non-determinism finding**: `golden::issue::2884` scored 4 in the earlier spot-check and 5 in this re-score, from the identical (query, answer, context, prompt) at `temperature=0.0` — `gpt-oss-120b` is not perfectly deterministic across real calls even at temperature 0. Full detail: `docs/results/judge-rubric-fix-rescore.md`. No paired data exists yet, so real inter-judge κ under the fixed rubric is still unmeasured — whether the groq-side compression helps or hurts agreement with Gemini can't be known until Gemini's side runs.

**Deferred to the user (judge κ):** the user will run this themselves once Gemini's quota allows, not queued as this project's own next-session task. Command: `uv run python experiments/11g_judge_rubric_rescore.py` — it's checkpoint-resumable (only fills in Gemini's missing 29 scores and the 3 missing groq rows, doesn't re-pay for the 26 groq scores already collected). After it completes, `docs/results/judge-agreement.md` needs updating with the real outcome (whichever direction it goes) — `make eval` will then pick up the new κ automatically, no scorecard changes needed. `GeminiRagasLLM` (Unit 1's `BaseRagasLLM` path for `Faithfulness`/`ResponseRelevancy`) got real, indirect exercise this phase (RAGAS calls ran via the Groq side only in the actual batch, so this specific class is still not directly smoke-tested — low priority given Groq-side RAGAS scoring is now working).

**Faithfulness root cause found as of 2026-09-26, zero LLM cost** (design record 11h, `docs/results/faithfulness-root-cause.md`, `experiments/11h_faithfulness_root_cause.py`). Pure analysis of data already on disk — correlated the 20 real faithfulness scores against the same rows' `valid_citation_rate` (Pearson r=0.3083, weak), then manually checked the two lowest-scoring real cases against their real cited context. **Two distinct, real mechanisms, not one:** `golden::issue::219` (faithfulness=0.077, the single lowest score) is a genuine, confident hallucination — its primary cited chunk is a real, completely unrelated GitHub issue — caught cleanly by RAGAS but only partially by the citation validator (0.667) and missed entirely by the pre-fix LLM judge (5/5), a striking three-way guardrail divergence on the same real answer. `golden::issue::15764` (faithfulness=0.250) is a verified-accurate, well-grounded honest refusal (its one positive claim checked directly against real context and confirmed true) — scored low purely because RAGAS's NLI-based check has no mechanism to credit a correct negative/absence claim, a metric-structural blind spot, not a real grounding failure. Recommended treating the 0.7977 mean as a disclosed, explained limitation rather than chasing it upward, since the fix for mechanism 2 (discouraging refusal phrasing) would optimize directly against Phase 10's confidence scorer. Satisfies plan.md's "written analysis of why not" exit-criterion alternative. `uv run pytest -q`: 420/420 (no new tests — pure data analysis, no new logic to guard).

**Next action (faithfulness):** none required — recommend closing this DoD item out as explained/disclosed rather than pursuing further. If revisited later: consider excluding or separately reporting refusal-answer faithfulness scores in `make eval`'s scorecard, since they measure the metric's blind spot rather than answer quality.

**Phase 9 sampling-sweep grid re-scoped and half-completed, 2026-09-26** (`docs/results/sampling-sweep.md`, addendum to design record 09). `llama-3.3-70b-versatile` (the model the original 2 attempts used) is retired — re-scoped `experiments/09a_sampling_sweep.py` to `qwen/qwen3.8-27b`, restored `N_QUERIES` to 20 (its per-minute quota shape affords this), added inter-call pacing against the real ITPM cap (Unit 4). Benchmarked one real call first before committing to the grid, same discipline as the original attempt. **Real result: both temperature=0.0 configs (top_p=0.5 and top_p=1.0) completed in full, N=20 each, and produced identical output** — same mean valid_citation_rate (0.7296), same exact claim-status counts — the correct, expected result (top_p has no effect at greedy/temp=0.0 decoding), confirmed rather than assumed. The third config (temperature=0.5) hit a real, new 200,000 TPD cap on `qwen/qwen3.8-27b` itself (previously only its per-minute caps were known) after 0/20 succeeded; killed the run there rather than let it grind through guaranteed-failing calls for the 4th config. Old Attempt 1's (0.0, 0.5) number (0.6634, different retired model) is kept as historical, not reused — mixing models within one sweep would confound the variable under test. `uv run pytest -q`: 420/420 (no new tests — an experiment script, I/O-bound, integration-only per CLAUDE.md's testing convention).

**Deferred to the user (sampling sweep):** the user will complete this themselves once `qwen/qwen3.8-27b`'s TPD quota resets, not queued as this project's own next-session task. Command: `uv run python experiments/09a_sampling_sweep.py` — made config-level resumable right after this run (reads the existing raw file, skips the 2 already-complete temperature=0.0 configs, only runs the remaining 2), so it's now safe to just re-run the same command directly. Each config used roughly 40,000-50,000 of the 200,000 daily budget, so a fresh day should cover both remaining configs in one sitting.

**Self-review pass, 2026-09-26.** Found and fixed one real inconsistency introduced by this session's own speed: design record 11f's §5/§6/§7 still only reflected the small 4-case spot-check after the fuller N=26 re-score had since produced a materially different, more nuanced picture (the distribution-compression finding, the non-determinism finding) — updated 11f in place to fold in the real, later result rather than leaving two contradictory "what actually happened" accounts across two documents.

> This file is the state of the world. It is updated at the end of every phase and
> whenever something significant lands. If you return after a gap, read this first —
> it is designed to be sufficient to reload full context on its own.

---

## Quick Status

| | |
|---|---|
| **Day** | 13 of 21 (in progress) |
| **Phase** | 15 of 19 **complete** (2026-10-22); next is Phase 16 (CI evaluation gate, never cut). Phase 14 complete. Phase 13: 3 of 4 exit criteria (p95 < 3 s unmet). Phase 12 complete. Phase 11 wrapped up with κ unmet. Phase 9 sweep complete |
| **Design records written** | 54 files in `docs/explainers/` across phases 0-15 (lettered sub-records; Phase 15 alone has 15, 15b-15l, 15g1-15g4) |
| **Golden set** | 180 dev + 40 holdout, frozen and hashed (Phase 4) |
| **Definition of Done** | 3 of 10 met (#1 Recall@10 + MRR lift; #3 escalation confusion matrix; #4 cache hit rate + cost/latency reduction). Measured but **not met**: #2 faithfulness (0.7977 vs ≥ 0.90), #6 p95 (5.6 s vs < 3 s), #8 judge κ (0.1321 at N=23 post-fix vs ≥ 0.6). Not started: #5, #7, #9, #10. `make eval` (Unit 6) now surfaces this and the faithfulness miss automatically, in one command. |
| **Blockers** | None hard. Free-tier quota no longer gates measurement: OpenAI is primary behind the $10 ledger ($0.350669 spent, 2026-10-22). Groq qwen (200k TPD) is still the free second judge and fallback generator. Docker Desktop must be running for Qdrant (it was down at the 2026-10-22 session start). Open measured gaps carried forward: Phase 9 uncited claims, Phase 11 κ 0.1321 < 0.6, Phase 13 p95 5.6 s > 3 s (qwen-era; not re-measured on gpt-5-mini) |

---

## Phase Board

Legend: ⬜ not started · 🟡 in progress · ✅ complete · ⏭️ cut

| # | Phase | Day | Status | Key output |
|---|---|---|---|---|
| 0 | Foundations & failing baseline | 1 | ✅ | Naive RAG that hallucinates, on purpose |
| 1 | Corpus acquisition & ingestion | 2 | ✅ | `documents.jsonl` |
| 2 | Chunking laboratory | 3 | ✅ | 4 strategies, one interface |
| 3 | Embeddings, distance & vector store | 4 | ✅ | ANN recall/latency curve |
| 4 | **Golden evaluation set** ⚑ | 5 | ✅ | 180 queries + 40 negatives, frozen, hashed |
| 5 | Retrieval core (analyzer, BM25, dense, RRF) | 6 | ✅ | Custom analyzer + fusion |
| 6 | **Retrieval eval & ablation grid** ⚑ | 7 | ✅ | The evidence table |
| 7 | Reranking | 8 | ✅ | Pool-size vs latency curve |
| 8 | Agentic router & query understanding | 9–10 | ✅ | Routing confusion matrix |
| 9 | Grounded generation & citations | 11 | 🟡 | Citation validator — sweep done 2026-10-18; open: "100% of claims cited" unmet (missing_citation 22/111 at t=0.0) |
| 10 | **Calibrated abstention** ⚑ | 12 | ✅ | The escalation confusion matrix |
| 11 | **Generation eval + judge validation** ⚑ | 13 | 🟡 | RAGAS + Cohen's κ |
| 12 | Caching, delta indexing & cost | 14 | ✅ | Cache hit rate, cost table |
| 13 | Observability & latency budget | 15 | 🟡 | p95 < 3s, per-stage breakdown |
| 14 | API, streaming & UI | 16–17 | ✅ | Working demo w/ visible escalation |
| 15 | Security hardening | 18½ | ✅ | Complete 2026-10-22, all 4 exit criteria met (`security-assessment.md` "Exit criteria"). Tenancy A1 609/1,800 -> 0 (0/760 bypass, re-run 15i); HTTP surface closed (15c); input boundary (15d, injection regex flag-only by measurement); prompt v3 served (15h): poisoned obedience 5/50 -> 0/50, direct exfiltration 1/28 -> 0/28; commitment + risky-advice output checks (15e, 15l); Qdrant keyed loopback (15f). Provider migration inside the phase: generator gpt-5-mini (15g1-g3), router nano (15g4). 31 security tests, 0 xfail |
| 16 | **CI evaluation gate** ⚑ | 18½ | ⬜ | Gate that fails on regression |
| 17 | LangGraph port & framework comparison | 19 | ⬜ | Comparison writeup, both paths pass eval |
| 18 | Deployment | 20 | ⬜ | Public URL |
| 19 | Failure analysis & writeup | 21 | ⬜ | Held-out numbers, taxonomy, README |

⚑ = never cut (see `plan.md` §7)

---

## Measured Results

*Nothing goes here that wasn't produced by a command that was actually run.*

| Metric | Value | Phase | Config | Date |
|---|---|---|---|---|
| Generation bake-off addendum: gpt-5-mini / gpt-5-nano (effort minimal, 0 reasoning tokens), qwen judge / nano judge | gpt-5-mini 0.7158 / 0.6242 (Δ −0.014, CI [−0.148, +0.111], tie; wrong-cite share 0.128; $0.001173/answer; p50 2,426 ms); gpt-5-nano 0.4217 / 0.3717 (7/20 truncated) | 15 | `15g1_generation_bakeoff.py --extra-models` | 2026-10-19 |
| Citation rate re-score under fixed claim splitter (15k), qwen judge | qwen 0.7539; gpt-5-mini v2 0.7217 / v3 0.7842; gpt-4.1-mini 0.4441; gpt-4.1-nano 0.6665 | 15 | `experiments/15k_rescore_claim_split.py` | 2026-10-22 |
| Risky-advice check (15l) | probes 12/12 risky, 0/12 legit; 15h obeyed install claims 3/4 frozen, 4/4 after widening; real answers 0/1,162; docs 0/14,824 | 15 | `experiments/15l_risky_advice_fp.py` | 2026-10-22 |
| Indirect injection, poisoned chunk at rank 1 (5 payloads x 10 queries), gpt-5-mini | obeyed v2 5/50, v3 0/50; 4/4 obeyed install claims validated SUPPORTED_LEXICAL | 15 | `experiments/15h_realstack_security.py --parts h1` | 2026-10-22 |
| Prompt v3 vs v2 citation rate, gpt-5-mini, bake-off 20 | served splitter +0.113 [−0.021, +0.268]; line-aware +0.068 [−0.039, +0.175] (qwen judge) | 15 | `15h_realstack_security.py --parts h2`, `15h_claim_split_sensitivity.py` | 2026-10-22 |
| Direct injection end to end (14 wrapped + 14 paraphrase), served pipeline | paraphrases 13/14 escalated; wrapped 0/14 escalated; complied v2 1/14 (system prompt leaked in French), v3 0/14 | 15 | `15h_realstack_security.py --parts h3` | 2026-10-22 |
| Router re-eval, 58 Phase 8 dev queries | accuracy nano 0.6379 / mini 0.7759 / qwen 0.7931; billing recall 8/8 + precision 1.00 in all three; nano known_issue recall 0.20 | 15 | `experiments/15g4_router_reeval.py` | 2026-10-19 |
| Prompt fit v2 vs v2c, dev 10 (disjoint), qwen judge | mini 0.409 -> 0.305, nano 0.495 -> 0.371; supported claims 31 -> 18 / 30 -> 13; refusal answers 2-3/10 -> 9/10 (negative result) | 15 | `experiments/15g2_prompt_fit.py --stage dev` | 2026-10-19 |
| Generation bake-off, valid_citation_rate (qwen judge / nano judge), N=20, v2, temp 0 | qwen 0.7296 / 0.6329; gpt-4.1-mini 0.4486 / 0.4361 (Δ −0.281, CI [−0.417, −0.140]); gpt-4o-mini 0.3458 / 0.3264; gpt-4.1-nano 0.6665 / 0.6288 (Δ −0.063, CI spans 0). Wrong-cite share: qwen 0.076, mini 0.222, 4o-mini 0.152, nano 0.243. Cost/answer mini $0.001765, 4o-mini $0.000618, nano $0.000434 | 15 | `experiments/15g1_generation_bakeoff.py` | 2026-10-19 |
| Full docs corpus size vs. context window | 363,420 tok (155 files) vs. 128K window (llama-3.3-70b-versatile) — 2.84× over | 0 | full `docs/en/docs/` tree | 2026-07-29 |
| Baseline subset size | 121,553 tok (85 tutorial/advanced docs, 486,214 chars) | 0 | `src/aegis/baseline.py` | 2026-07-29 |
| Cost: full-subset-stuffing vs. retrieval, per query (input tokens only) | $0.0717 vs. ~$0.0002 (~300× cheaper) | 0 | top_k=3, 500-char chunks, Groq $0.59/M input | 2026-07-29 |
| Confident hallucinations recorded | 3 of 5 questions, 2 independent runs | 0 | `docs/results/00-baseline-transcript.json` | 2026-07-29 |
| Documentation pages ingested | 129 (of 155 English `.md` files; 131 excl. `reference/`, 129 after excluding 2 non-content files) | 1 | `data/processed/documents.jsonl` | 2026-07-29 |
| Closed issues ingested (PRs excluded) | 3,530 | 1 | GitHub REST API, `fastapi/fastapi` | 2026-07-29 |
| Issues with identifiable maintainer resolution | 2,204 / 3,530 (62.4%) — ceiling for Phase 4 golden set | 1 | `author_association` in {OWNER,MEMBER,COLLABORATOR} | 2026-07-29 |
| Token distribution (docs / issues), chars/4 estimate | docs p50=1,436 p95=6,609 max=171,510; issues p50=470 p95=1,444 max=48,397 | 1 | `documents.jsonl` | 2026-07-29 |
| Chunk count, 4 strategies (129 docs + 51 sampled issues) | fixed 1,646 · recursive 1,791 · heading_aware 2,682 · semantic 1,787 | 2 | `max_tokens=400, overlap_tokens=50` | 2026-07-30 |
| Code-block integrity rate, 4 strategies | fixed 84.1% · recursive 85.3% · heading_aware 98.4% · semantic 99.1% (880 real blocks) | 2 | same config | 2026-07-30 |
| Chunks with populated `heading_path` | fixed/recursive/semantic 0.0% · heading_aware 99.9% | 2 | same config | 2026-07-30 |
| Oversized chunks, full 3,659-doc corpus (no sampling) | fixed 0/8,984 · recursive 0/9,561 · heading_aware 427/19,236 (420 documented atomic-code exception + 7 accepted residual limitation, see explainer §6) | 2 | `docs/results/chunking-stats.md` | 2026-07-30 |
| Semantic chunking wall time, single largest doc (`release-notes.md`, 3,690 sentences) | 211–248s (confirmed compute-bound: batched vs. 32 per-segment calls differ by only ~15%) | 2 | `bge-small-en-v1.5` / fastembed, CPU | 2026-07-30 |
| Cosine ≡ dot product on real corpus vectors | Max abs diff 1.79e-07 across 2,000 real chunk pairs; all norms exactly 1.000000 | 3 | `bge-small-en-v1.5`, 19,236 chunks | 2026-08-02 |
| Embedding throughput, full 19,236-chunk corpus | bge-small 1,517.1s (12.7/s avg, unmitigated, swap-driven) · bge-large 665.0s (28.9/s avg, mitigated) · BGE-M3 799.4s (24.1/s avg, mitigated) | 3 | MPS, M1 Pro 16GB | 2026-08-02 |
| Swap usage during sustained MPS embedding | ~3.5GB (`vm.swapusage`) | 3 | Concurrent Qdrant + MPS embedding load | 2026-08-02 |
| Exact-duplicate embeddings in corpus | 39.7% of 19,236 vectors; largest group 1,354 issues whose chunk text is the literal string `_No response_` | 3 | `bge-small-en-v1.5` | 2026-08-02 |
| HNSW recall@10 (score-aware, tie-corrected), 36 configs swept | 1.0000 for all 36 (m ∈ {8,16,32}, ef_construct ∈ {50,100,200}, ef_search ∈ {50,100,200,400}) | 3 | `docs/results/ann-recall-vs-latency.md` | 2026-08-02 |
| HNSW query latency, 36 configs swept | 6.07–8.07 ms/query (no consistent trend with any parameter) | 3 | 300 self-sampled queries, 19,236-pt corpus | 2026-08-02 |
| Post-filter vs. pre-filter recall, 5 `product_area` categories | Post-filter 0.1667–0.4033 (even `general`, 26% of corpus, short of k on 90% of queries) · Pre-filter 1.0000 on all 5 | 3 | `docs/results/prefilter-vs-postfilter.md` | 2026-08-02 |
| Resolved GitHub issues (FastAPI closed issues with maintainer resolution) | 2,204 of 3,530 (62.4%) | 4 | Phase 1 measurement; harvest pool for golden set | 2026-08-03 |
| Low-signal resolutions filtered (bare "thanks", pure redirects, <=6 words) | 95 of 2,204 (4.3%) | 4 | Conservative regex, retained as buffer | 2026-08-03 |
| Stratum distribution (2,109 candidates post-filter) | how_to 385 (18.3%) · error_debugging 1,096 (51.9%) · conceptual 41 (1.9%) · version_specific 58 (2.7%) · multi_hop 281 (13.3%) · ambiguous 343 (16.3%) | 4 | `src/aegis/eval/strata.py` classification | 2026-08-03 |
| Candidates harvested (30/stratum with 2x buffer) | 333 positives (60 how_to, 60 error_debugging, 38 conceptual, 55 version_specific, 60 multi_hop, 60 ambiguous) + 40 negatives = 373 total | 4 | Overflow buffer absorbs low-quality issues that survival labeling would skip | 2026-08-03 |
| Dense retrieval composition (unfiltered top-20) | 5,571 issue, 1,089 doc (83.6% issue-dominated) across 333 candidates; 103/333 (30.9%) with zero doc chunks in top-20 | 4 | Boilerplate-pollution failure mode (same as Phase 1/2): GitHub issue-template text dominates similarity | 2026-08-03 |
| Dense retrieval composition (12 doc + 8 issue filtered) | 3,996 doc, 2,664 issue (100% of queries guaranteed doc representation) | 4 | Fix: doc_type-filtered searches instead of flat top-20 | 2026-08-03 |
| Golden set labels (LLM-based, Haiku, token-overlap heuristic ≥15%) | 180 positives (30/stratum) + 40 negatives = 220 total | 4 | Deterministic, reproducible from seed; Phase 11 will measure agreement with Opus judge via Cohen's κ | 2026-08-03 |
| Golden set split (dev / holdout) | 180 dev (25 pos/stratum + 30 neg), 40 holdout (5 pos/stratum + 10 neg) | 4 | Holdout sealed; Phases 6–16 measure on dev, Phase 19 on holdout for generalization | 2026-08-03 |
| Golden set hash (dev split, SHA256) | `b48a1a73f7ca70231544972b0404d80a4260bcc541ca0264ea23848375b27214` | 4 | Proof-of-freeze in `data/golden/golden_set_v1.jsonl` | 2026-08-03 |
| Golden set hash (holdout split, SHA256) | `d4d10c2c5316615657f13caae93c73fbeb7c8ae6318f4fdb021afc18a3b7adcc` | 4 | Sealed in `data/golden/golden_set_holdout_v1.jsonl` | 2026-08-03 |
| Analyzer test suite (incl. the 40-real-token exit criterion) | 56/56 passed | 5 | `tests/test_analyzer.py`, tokens pulled from `documents.jsonl` | 2026-08-03 |
| BM25 vs. dense on a real `ECONNREFUSED` query, real issue `issue::5578` (contains the literal token) | BM25: rank 1, score 3.1467 · Dense (bge-small): not found in top 200 of 19,236 chunks | 5 | `SparseIndex`/`DenseIndex`, full corpus | 2026-08-03 |
| Unfiltered dense search vs. `doc_type`-prefiltered, query about `response_model` | Unfiltered top-5: 100% issue chunks (0.82-0.86 cosine), correct doc absent · Prefiltered: top-1 is `tutorial/response-model.md`, score 0.7864 | 5 | `DenseIndex`, bge-small, full corpus — reproduces Phase 3's finding through the new module | 2026-08-03 |
| Full test suite after Phase 5 | 245/245 passed | 5 | `uv run pytest -q` | 2026-08-03 |
| `ranx` verification (hand-computed 5-doc example) | Precision@5=0.6, Recall@5=1.0, MRR=1.0, MAP@5=0.7556, NDCG@5=0.8855 — exact match | 6 | `tests/test_metrics.py` | 2026-08-04 |
| Winning retrieval config: recall@10 / MRR / NDCG@10 / MAP@10 | 0.4288 / 0.8640 / 0.7064 / 0.3695 | 6 | heading-aware, enrichment off, dense, bge-large; 150 dev positives, self-match excluded | 2026-08-04 |
| Retrieval-strategy axis: dense vs. sparse vs. hybrid (retuned) | dense 0.4288 vs. sparse 0.0967 vs. best-tuned hybrid_rrf 0.4324 (statistically ties dense, never exceeds it) recall@10 | 6 | `docs/results/retrieval-ablation.md` §1 | 2026-08-04 |
| Encoder axis: bge-large vs. BGE-M3 (under dense) | 0.4288 vs. 0.1348 recall@10 | 6 | Same golden set, both already-indexed Phase 3 collections | 2026-08-04 |
| Contextual enrichment axis: off vs. on | 0.4288 vs. 0.1751 recall@10 — opposite of plan.md's default expectation | 6 | Root cause: GitHub issue-template boilerplate headings shared across thousands of unrelated issues, same failure mode as Phases 1-4's boilerplate-pollution | 2026-08-04 |
| Chunking axis (doc-level ground-truth transfer): heading_aware vs. fixed/recursive/semantic | 0.5470 vs. 0.2393 / 0.2436 / 0.2571 recall@10 | 6 | First real retrieval-quality evidence for Phase 3's provisional chunking choice; resolves Open Question #5 (semantic chunking's cost doesn't pay for itself) | 2026-08-04 |
| Semantic chunking, full 3,659-doc corpus (first full-corpus run — Phase 2 only sampled) | 723s to chunk + 640s to embed (9,168 chunks) | 6 | `experiments/06a_prepare_ablation_indexes.py semantic` | 2026-08-04 |
| Self-match leak effect (before vs. after `exclude_doc_id` fix), dense/heading-aware/bge-large | recall@10 0.4193→0.4288, MRR 0.6161→0.8640 | 6 | Every golden query's own source issue was a trivial near-verbatim top-1 hit, wasting a top-10 slot; affected every measurement in this phase's first pass | 2026-08-04 |
| Full test suite after Phase 6 | 272/272 passed | 6 | `uv run pytest -q` | 2026-08-04 |
| Reranking NDCG@5 vs. Phase 6's winning config, all 4 swept pool sizes | no-rerank 0.7550 · pool=10 0.6873 (−9.0%) · pool=20 0.5344 (−29.2%) · pool=50 0.4072 (−46.1%) · pool=100 0.3340 (−55.8%) | 7 | `bge-reranker-v2-m3`, Kaggle T4, 150 dev positives, `docs/results/rerank-tradeoff.md` | 2026-08-11 |
| Reranking latency (p95), all 4 swept pool sizes | pool=10 5,227ms · pool=20 12,362ms · pool=50 33,936ms · pool=100 57,563ms | 7 | Kaggle T4, same run | 2026-08-11 |
| Golden set ground-truth breadth (median across 150 dev positives) | 17 ground-truth chunks spanning 12 distinct documents per query | 7 | Root-cause check for the reranking regression; token-overlap auto-labeling (`auto_label.py`) is "biased toward inclusion" by design | 2026-08-11 |
| Local CPU cross-encoder throughput: synthetic short text vs. real corpus-length text | ~30.5ms/pair (synthetic, ~15 words) vs. ~780ms/pair (real chunks, up to ~400 tokens) — 25x | 7 | Benchmarked before the abandoned local CPU sweep attempt | 2026-08-11 |
| Full test suite after Phase 7 | 283/283 passed | 7 | `uv run pytest -q` | 2026-08-11 |
| Routing confusion matrix (58 labeled queries: 25 how_to→docs_howto, 25 error_debugging→known_issue, 8 billing_legal→billing_account_legal) | Overall accuracy 68.97% (40/58); per-class recall docs_howto 0.80, known_issue 0.48, **billing_account_legal 1.0000** | 8 | `llama-3.1-8b-instant`, few-shot, JSON-mode; `docs/results/routing-confusion-matrix.md` | 2026-08-16 |
| Query-understanding ablation vs. Phase 6 winning config, 150-query dev set | expansion NDCG@5 −49.4%, recall@10 −51.8% · HyDE NDCG@5 −23.6%, recall@10 −28.3% · iterative retrieval NDCG@5 −13.4%, recall@10 −15.8% | 8 | `docs/results/query-understanding-ablation.md` | 2026-08-16 |
| Query rewriting vs. context-free follow-up, 24-item synthetic multi-turn set | NDCG@5 0.1995→0.3045 (+52.7%), recall@10 0.0897→0.1203 (+34.2%); oracle (original full query) 0.7405 | 8 | LLM-generated synthetic conversations (`data/golden/query_rewrite_eval_v1.jsonl`), scored against each item's real original golden-query ground truth | 2026-08-16 |
| Multi-query expansion regression, root cause (20-query spot check) | Mean NDCG@5 0.868→0.440, 0/20 improved; small paraphrase model sometimes generates invented answers instead of rephrasings, and unweighted 4-way RRF lets 3 wrong lists outvote 1 correct one | 8 | `llama-3.1-8b-instant` paraphrases, `rrf_k=60` (tuned in Phase 6 for 2 sources) | 2026-08-16 |
| Full test suite after Phase 8 | 318/318 passed | 8 | `uv run pytest -q` | 2026-08-16 |
| Real per-chunk token_count distribution, full 19,236-chunk corpus (grounds the context budget default) | p50=79, p90=363, p95=390, mean=135.7 tokens | 9 | `data/processed/chunks.jsonl` | 2026-08-16 |
| Full-pipeline smoke test (real Qdrant retrieval + real Groq generation + real citation validation, 1 query) | 6 claims, 0 missing/invalid citations, 4/6 needed tier-2 entailment, 1 correctly flagged `UNSUPPORTED` (model invented example code not in context) | 9 | `docs/results/sampling-sweep.md` context | 2026-08-16 |
| Sampling sweep, temperature=0.0/top_p=0.5, N=20 real dev-positive queries (`llama-3.3-70b-versatile`, since retired -- historical, not comparable to the 2026-09-26 re-run below) | mean `valid_citation_rate` (faithfulness proxy) 0.6634; claims: supported_lexical=99 (57.9%), supported_entailment=23 (13.5%), unsupported=13 (7.6%), missing_citation=36 (21.1%); mean latency 15,522ms | 9 | `docs/results/sampling-sweep.md` | 2026-08-16 |
| Sampling sweep re-run, `qwen/qwen3.8-27b`, both temperature=0.0 configs, N=20 each | Identical results for top_p=0.5 and top_p=1.0 -- mean valid_citation_rate=0.7296 both, identical claim-status counts (supported_lexical=59, supported_entailment=14, exempt=10, missing_citation=22, unsupported=6) -- real, mechanistically expected (top_p has no effect at greedy/temp=0.0 decoding) | 9 | `docs/results/sampling-sweep.md` | 2026-09-26 |
| `qwen/qwen3.8-27b` tokens/day (TPD) quota, discovered running the sweep's 3rd config | Limit 200,000/day -- same magnitude as `gpt-oss-120b`/`gpt-oss-20b` (Unit 4), now confirmed on this model too; hit partway into temperature=0.5, top_p=1.0 (0/20 succeeded); 4th config (temp=1.0) never attempted | 9 | Real `RateLimitError`, `docs/results/sampling-sweep.md` | 2026-09-26 |
| Citation validator tier-1/tier-2 cost split (of 135 non-missing/invalid claims, same run) | 99/135 (73.3%) resolved free at tier 1 (lexical overlap); only 36/135 (26.7%) needed an LLM entailment call | 9 | Derived from the sweep above | 2026-08-16 |
| Groq free-tier daily token quota, `llama-3.3-70b-versatile` | Limit 100,000 tokens/day; one 20-query sampling config alone used 95,944 (real ~4,800 tokens/query at `max_tokens=1024`); fully exhausted (`100000/100000`) after a second, smaller attempt | 9 | Real `RateLimitError` responses, `docs/results/sampling-sweep.md` | 2026-08-16 |
| Real (non-faked) provider-fallback demonstration: `GroqProvider` with a nonexistent model name → real `NotFoundError` (404) → `FallbackLLMClient` recovers via a second `GroqProvider` on a separate quota bucket | 5 real streamed chunks, joined: "Automatic API documentation generation." | 9 | Live Groq API, two real calls | 2026-08-16 |
| Full test suite after Phase 9 (partial) | 355/355 passed (+37 new: budgeter, prompts, fallback client, citation validator, pipeline) | 9 | `uv run pytest -q` | 2026-08-16 |
| Gemini model availability: `gemini-2.5-flash` (in `models.list()`) vs. a real generation call | Listed as available; real call returns `404 NOT_FOUND` — "no longer available to new users." Pinned to `gemini-3.5-flash` instead, verified against a real successful call | 9 | Live `google-genai` API | 2026-08-16 |
| Gemini "thinking" mode default cost | Real call at `max_tokens=20` → `finish_reason=MAX_TOKENS`, 0 visible characters (entire budget spent on hidden `thought` tokens). Fixed with `thinking_config=ThinkingConfig(thinking_budget=0)` | 9 | Live `google-genai` API, `gemini-3.5-flash` | 2026-08-16 |
| Provider bake-off, Groq vs. Gemini, 15 real dev-positive queries, shared retrieval | Groq: 2/15 succeeded (13 `RateLimitError` — quota, not quality), mean latency 1,549ms. Gemini: 12/15 succeeded (3 transient `503`), mean `valid_citation_rate` 0.7014, mean latency 21,437ms — **~14x** Groq's | 9 | `docs/results/llm-bakeoff.md` | 2026-08-16 |
| Real (non-forced) Groq→Gemini chained fallback via `default_client()` | Succeeded via Groq directly on a small call shortly after the bake-off — confirms Groq's quota is request-size-sensitive, not a hard full-day lockout | 9 | Live Groq API | 2026-08-16 |
| Gemini free-tier request quota (discovered running the v1-vs-v2 prompt check) | 20 requests/day/model (not token-based, unlike Groq) — already mostly consumed by 17 earlier real Gemini calls this session (2 smoke tests + 15 bake-off) | 9 | Real `ClientError 429 RESOURCE_EXHAUSTED`, `experiments/09c_prompt_v2_check.py` | 2026-08-16 |
| Prompt v1 vs. v2 comparison (Gemini, N=10 planned) | v1: 2/10 completed (mean `valid_citation_rate` 0.9375, 1/14 claims missing_citation) before the 20-req/day wall; v2: 0/10 completed — **not validated**, data too thin to trust | 9 | `docs/results/prompt-v1-vs-v2-raw.json` | 2026-08-16 |
| Full test suite after Phase 9 (v2 prompt added) | 355/355 passed (v2 registration only, no new test surface) | 9 | `uv run pytest -q` | 2026-08-16 |
| Citation validator sentence-splitter fix (fenced code blocks) | RED test reproduced the real glued-sentence bug; GREEN after reusing `chunking/segments.py`'s `split_code_segments` | 9 | `tests/test_citations.py::test_fenced_code_block_does_not_get_glued_to_surrounding_prose` | 2026-08-17 |
| Full test suite after Phase 9 (sentence-splitter fix) | 356/356 passed (+1 new) | 9 | `uv run pytest -q` | 2026-08-17 |
| Prompt v1 vs. v2, real matched-pair comparison (N=6, same 6 real dev-positive queries, Groq, both quotas actually reset) | mean `valid_citation_rate` 0.4199 (v1) vs. 0.5641 (v2), +34.3% relative; v2 wins 5, ties 1, loses 0 | 9 | `docs/results/prompt-v1-vs-v2.md`, `prompt-v1-vs-v2-groq-raw.json` | 2026-08-17 |
| Confidence-signal rank-correlation with the real should-escalate label (180-query dev sweep) | reranker_top_score -0.58 · semantic_agreement -0.22 · score_gap -0.16 · distribution_shape -0.02 | 10 | `docs/results/confusion-matrix.md` | 2026-08-17 |
| Single-signal vs. combined-signal ablation (recall at precision>=0.5) | reranker_top_score alone 0.9333 · equal-weighted (0.25 each) combination 0.8333 (*worse* than one signal) · shipped correlation-weighted (0.60/0.15/0.20/0.05) combination 0.9000 | 10 | `docs/results/confusion-matrix.md` | 2026-08-17 |
| distribution_shape raw range, real dev sweep | [0.0000, 76856.5500] — one query's near-zero-variance top-k dense scores; winsorized (p1/p99) bounds used instead: [1.4188, 27.5129] | 10 | Traces to Open Question #6's near-duplicate-chunk corpus property | 2026-08-17 |
| Escalation confusion matrix at chosen tau=0.50 (150 dev positive / 30 dev negative) | TP=27 FP=24 FN=3 TN=126 — precision=0.5294 recall=0.9000 deflection_rate=0.8400 | 10 | `docs/results/confusion-matrix.md`, `confusion-matrix.png` | 2026-08-17 |
| Per-negative-category recall at tau=0.50 | adversarial_injection 1.00 (7/7) · billing_legal 1.00 (8/8) · out_of_corpus 0.83 (5/6) · unanswerable 0.78 (7/9) | 10 | Same sweep | 2026-08-17 |
| Per-stratum over-caution (FP) rate at tau=0.50 | ambiguous 28.0% · conceptual 24.0% · how_to 20.0% · multi_hop 16.0% · error_debugging 4.0% · version_specific 4.0% | 10 | Same sweep | 2026-08-17 |
| Full tau sweep wall time, 180 real queries (dense retrieval + cross-encoder rerank of a 3-candidate pool, CPU) | 736s (12.3 min) — a 6-query benchmark had projected ~5-9 min | 10 | `experiments/10a_tau_sweep.py`, CLAUDE.md's 10-min local-compute line | 2026-08-17 |
| Full test suite after Phase 10 | 373/373 passed (+17 new: confidence scorer, decide/handle, escalation payload) | 10 | `uv run pytest -q` | 2026-08-17 |
| `openai/gpt-oss-120b` reasoning-token tax at real answer length, 6 real dev-positive queries | 6/6 finished with visible text; reasoning_tokens 25-45% of completion_tokens; mean latency 16,697ms (1,186-29,044ms) | 11 | `experiments/11a_gpt_oss_reasoning_tax.py`, `docs/results/groq-model-replacement.md` | 2026-09-25 |
| `qwen/qwen3.8-27b` reasoning-token check at real answer length, same 6 queries | 0/6 reasoning tokens; 2/6 real 429 at max_tokens=1024 (1000 OTPM cap), both fixed at max_tokens=1000; mean latency 8,692ms (653-32,309ms) | 11 | `experiments/11b_qwen_reasoning_check.py`, `docs/results/groq-model-replacement.md` | 2026-09-25 |
| Real end-to-end pipeline check, new `qwen/qwen3.8-27b` default (`GroqProvider`+`GenerationPipeline`, 1 real golden query) | latency 1,125ms, valid_citation_rate=1.0 | 11 | Live Groq API via `config.py`'s new default | 2026-09-25 |
| RAGAS smoke test, 1 real generated answer (`golden::issue::2689`) | faithfulness=0.8750 (16,157ms) · response_relevancy=0.7820 (55,487ms) · id_based_context_precision=0.8000 · id_based_context_recall=0.8000 (both <1ms, no LLM) | 11 | `experiments/11c_ragas_smoke_test.py`, `docs/results/ragas-smoke-test.md` | 2026-09-25 |
| Full test suite after Phase 11 Unit 1 | 373/373 passed (no new tests -- `ragas_adapters.py` is I/O-bound, covered by the integration smoke test above) | 11 | `uv run pytest -q` | 2026-09-25 |
| Cohen's kappa wrapper vs. hand-computed textbook example (50 items, Wikipedia's worked example) | kappa=0.4000, raw_agreement_rate=0.7000 -- exact match | 11 | `tests/test_agreement.py` | 2026-09-25 |
| Full test suite after Phase 11 Unit 2 | 380/380 passed (+7 new: `test_agreement.py`) | 11 | `uv run pytest -q` | 2026-09-25 |
| LLM-as-judge smoke test, 1 real generated answer (`golden::issue::2689`), both real judges | `openai/gpt-oss-120b`: score=4, 1,647ms · `gemini-3.5-flash`: score=5, 17,225ms | 11 | `experiments/11d_judge_smoke_test.py`, `docs/results/judge-smoke-test.md` | 2026-09-25 |
| Full test suite after Phase 11 Unit 3 | 389/389 passed (+9 new: `test_judge.py`) | 11 | `uv run pytest -q` | 2026-09-25 |
| Judge validation batch, N=30 dev-positive queries, real generation+RAGAS+both judges | generation 29/30 (1 real simultaneous Groq+Gemini failure); faithfulness mean=0.7977 (n=20); response_relevancy mean=0.5662 (n=27); id_based_context_precision mean=0.5828 (n=29); id_based_context_recall mean=0.4103 (n=29) | 11 | `experiments/11e_judge_validation_batch.py`, `docs/results/judge-agreement.md` | 2026-09-25 |
| Inter-judge Cohen's κ (Gemini vs. `openai/gpt-oss-120b`, N=9 paired of 29) | κ=-0.0161 (unweighted) / 0.0906 (quadratic) — at/below chance; raw_agreement_rate=0.2222 | 11 | `eval/agreement.cohen_kappa` against real judge output, `docs/results/judge-agreement.md` | 2026-09-25 |
| Judge score distributions (full, not just paired) | `openai/gpt-oss-120b` n=20: {4:9, 1:5, 5:3, 2:3}, mean=3.10 · Gemini n=11: {5:10, 4:1}, mean=4.91 (near-zero variance) | 11 | Same batch | 2026-09-25 |
| Real quota walls hit running the N=30 batch, by model | `qwen/qwen3.8-27b`: input-tokens/minute (ITPM) cap, 7000 limit · `openai/gpt-oss-120b`: 200,000 tokens/day (TPD), ~99% consumed by query ~10/30 · `openai/gpt-oss-20b`: same 200,000 TPD, independently exhausted by query ~29/30 · Gemini: 20 requests/day, fully exhausted mid-batch | 11 | Real API errors, `docs/results/judge-agreement.md` | 2026-09-25 |
| Full test suite after Phase 11 Units 4-5 | 402/402 passed (+13 new: `test_custom_metrics.py`; Unit 4's batch script is integration-only, no new unit tests) | 11 | `uv run pytest -q` | 2026-09-25 |
| `make eval` scorecard, real output (Unit 6, pure aggregation over Unit 4's raw batch, zero new LLM calls) | 2/4 plan.md exit criteria pass (custom metrics implemented, command runs); 2/4 fail (faithfulness 0.7977 < 0.90, kappa -0.0161 < 0.6) — both already-known, now surfaced automatically | 11 | `src/aegis/eval/scorecard.py`, `make eval` | 2026-09-25 |
| Code-block syntax validity, real batch answers (Unit 5 metric run for real via the scorecard) | n_checked=8 n_valid=8 (100%), n_skipped_non_python=0 | 11 | Same run | 2026-09-25 |
| Mean citation-validity rate, real batch answers (scalar only — no per-`ClaimStatus` breakdown available from stored batch data) | n=29, mean=0.7069 | 11 | Same run | 2026-09-25 |
| Full test suite after Phase 11 Unit 6 | 416/416 passed (+14 new: `test_scorecard.py`, including a regression guard against `docs/results/judge-agreement.md`'s exact published numbers) | 11 | `uv run pytest -q` | 2026-09-25 |
| Judge rubric fix, real spot-check: `gpt-oss-120b` scores on the 4 real Unit 4 disagreement cases, old rubric vs. new | `[1, 2, 1, 1]` -> `[5, 5, 5, 4]` -- exact match to Gemini's original scores `[5, 5, 5, 4]` on the same 4 rows | 11 | `docs/results/judge-rubric-fix-spot-check.md` | 2026-09-25 |
| Judge rubric fix, real spot-check: `gpt-oss-120b` scores on 3 real non-refusal cases, old vs. new | `4->4`, `4->4`, `5->5` -- unchanged, no overcorrection | 11 | Same spot-check | 2026-09-25 |
| Full test suite after Phase 11 judge-rubric fix | 420/420 passed (+4 new prompt-content tests) | 11 | `uv run pytest -q` | 2026-09-25 |
| Judge rubric fix, full re-score of Unit 4's 29 real answers, groq side only (Gemini 0/29, still fully quota-blocked) | 26/29 succeeded (3 hit `gpt-oss-120b`'s TPD cap again); mean 3.10->4.31; all 5 real refusal cases moved up, matching Gemini's original stance; 2 non-refusal answers moved down (5->4, anchor working as intended); 1 non-refusal moved up with no clear cause | 11 | `experiments/11g_judge_rubric_rescore.py`, `docs/results/judge-rubric-fix-rescore.md` | 2026-09-26 |
| `gpt-oss-120b` judge non-determinism at temperature=0.0: identical (query, answer, context, prompt) re-scored twice | 4 (spot-check) vs. 5 (full re-score) -- not perfectly deterministic across real calls | 11 | Same rescore, `golden::issue::2884` | 2026-09-26 |
| Faithfulness root cause: correlation between real RAGAS faithfulness and `valid_citation_rate`, n=20 | Pearson r=0.3083 -- weak, not zero | 11 | `experiments/11h_faithfulness_root_cause.py`, `docs/results/faithfulness-root-cause.md` | 2026-09-26 |
| Faithfulness root cause, case 1: `golden::issue::219` (lowest real score, 0.077) | Genuine hallucination -- cited chunk [2] is a real, completely unrelated GitHub issue; caught by RAGAS (0.077), only partially by the citation validator (valid_citation_rate=0.667), missed entirely by the pre-fix LLM judge (5/5) | 11 | Same analysis | 2026-09-26 |
| Faithfulness root cause, case 2: `golden::issue::15764` (an honest refusal, 0.250) | Verified-accurate -- its one positive claim (`route.tags` is a valid attribute) is directly confirmed by the real cited context; low score is a metric-structural blind spot (NLI can't credit a correct negative/absence claim), not a real grounding failure | 11 | Same analysis | 2026-09-26 |
| Full test suite after Phase 12 Unit 1 | 430/430 passed (+10 new: `test_exact_match_cache.py`) | 12 | `uv run pytest -q` | 2026-09-26 |
| Semantic cache threshold sweep: 15 real hardest distinct-query pairs (negatives) vs. 15 real hand-written paraphrases (positives), bge-large | Real overlap found -- negatives up to 0.8694, positives range 0.7441-0.9481; chosen threshold 0.88 (lowest with 0 real false positives): 0/15 false positives, 9/15 (60%) true positives | 12 | `experiments/12a_semantic_cache_threshold_sweep.py`, `docs/results/semantic-cache-threshold.md` | 2026-09-26 |
| Full test suite after Phase 12 Unit 2 | 440/440 passed (+10 new: `test_semantic_cache.py`, including one real integration test against the actual hardest near-miss pair) | 12 | `uv run pytest -q` | 2026-09-26 |

### Definition of Done tracker

| # | Criterion | Target | Current | Status |
|---|---|---|---|---|
| 1 | Recall@10 + MRR lift, baseline → final | measured | Recall@10 +0.332 (4.4x), MRR +0.496 (2.35x) vs. worst measured config (sparse-only) — see `docs/results/retrieval-ablation.md` §7 for why Phase 0's literal baseline isn't corpus-comparable | ✅ |
| 2 | RAGAS faithfulness | ≥ 0.90 | **Measured, target not met, real root cause found (2026-09-26):** mean=0.7977 (n=20/29). Two distinct real mechanisms, not one: (a) a genuine hallucination (`golden::issue::219`, 0.077) that RAGAS correctly caught and the citation validator + LLM judge both missed or fully missed; (b) a metric-structural blind spot where RAGAS's NLI check can't credit a correct negative/refusal claim (`golden::issue::15764`, 0.250, verified accurate). Recommend closing as a disclosed, explained limitation rather than chasing the number — "fixing" (b) would mean discouraging honest refusals, i.e. optimizing against Phase 10's own design. Satisfies plan.md's "written analysis of why not" exit-criterion alternative. `docs/results/faithfulness-root-cause.md` | ⚠️ |
| 3 | Escalation confusion matrix published | 4 quadrants | 4 quadrants, real numbers: TN=126 FP=24 FN=3 TP=27 at tau=0.50 (`docs/results/confusion-matrix.md`) | ✅ |
| 4 | Cache hit rate + cost/latency reduction | measured | Zipf-simulated over the 150 dev intents (Phase 12 Unit 3): hit rate 56.7-86.7% at cold start (N=150, skew 0.8-1.6) rising to ~98% by N=9,000; latency/token reduction tracks hit rate (~98% at N=9,000); exact hit 0.0 ms vs miss 12,787.5 ms modeled (`docs/results/cache-performance.md`, `cost-per-query.md`). Caveats: simulated traffic (no real log); ceiling is a property of 150 intents; semantic tier (0.2-1.6% of hits) off since 2026-10-17 (14h) | ✅ |
| 5 | CI gate demonstrably fails on regression | screenshot | — | ⬜ |
| 6 | p95 end-to-end, retrieval path | < 3 s | **Measured, not met (Phase 13):** paced generate path 4.4 s p50 / 5.6 s p95; confidence-stage variance unexplained. Re-verify at P18 | 🟡 |
| 7 | Failure analysis, categorized | written | — | ⬜ |
| 8 | LLM-judge agreement (Cohen's κ) | ≥ 0.6 | **Measured pre-fix, target not met:** κ=-0.0161 (unweighted) / 0.0906 (quadratic), N=9 paired — real, methodology-substituted (inter-judge, not judge-vs-human; user declined to hand-label). Root cause identified and rubric fixed (Gemini ceiling-effect anchor + `gpt-oss-120b` refusal-penalizing rule), groq-side verified real at N=26. **Real paired κ under the fixed rubric is deferred to the user** — Gemini has 0/29 real scores under it (`experiments/11g_judge_rubric_rescore.py`, checkpoint-resumable, run once its quota allows). Surfaced automatically by `make eval` (Unit 6). `docs/results/judge-agreement.md` | ⚠️ |
| 9 | Adversarial security suite passing | all pass | **Before measured (2026-10-18):** 12 of 13 attacks succeed, 1 defended (A5f). After 15b-15f: A1, A2, T1, A4, A5a-e, A5g, A6, P1 defended; 1 xfail (A3 on served v2; fixed in unserved v3). Model-behavior attacks not yet run (quota) | 🟡 |
| 10 | Held-out generalization number | reported | — | ⬜ |

---

## Decision Log

Every non-obvious choice, its alternatives, and why. Append-only.

| Date | Decision | Alternatives rejected | Reasoning |
|---|---|---|---|
| 2026-07-28 | FastAPI as the corpus | Supabase; Kubernetes | Structured Markdown with heavy code blocks, ~11k closed issues as free labels, and a domain the user can personally verify — which is what makes the golden set trustworthy. |
| 2026-07-28 | 21 days, full-time | 14 days | Full-rigor eval is ~30% of the work; 14 days forces cuts to the evidentiary spine. |
| 2026-07-28 | Full-rigor evaluation | Lighter eval; minimal smoke tests | The abstention threshold cannot be calibrated without it, and calibrated abstention is the entire differentiator of the project. |
| 2026-07-28 | **Golden set at Phase 4** (moved from proposal §5.8) | Keep proposal ordering | Every choice in Phases 5–11 needs a measurement to justify it. Evaluation is a prerequisite, not a report card. |
| 2026-07-28 | Added Phase 15 (security) | Leave security as a reading topic | The corpus is user-generated GitHub comments flowing into an LLM prompt — a live indirect-injection surface. |
| 2026-07-28 | Added negative set (40 queries) | Golden set only | Without refusal-expected queries, two quadrants of the confusion matrix are empty and τ is tuned on data where abstention is always wrong. |
| 2026-07-28 | Python 3.12 via `uv` | System Python 3.13; poetry; pip+venv | Parts of the ML stack still lag 3.13. `uv` for speed and a lockfile. |
| 2026-07-28 | Qdrant | pgvector; Chroma; Weaviate | Native sparse vectors + strong payload filtering for hybrid search. |
| 2026-07-28 | Groq primary, Gemini fallback | Single provider | Free tiers usable; multi-provider fallback is a production requirement anyway. |
| 2026-07-28 | LLM-judge model **deferred to Phase 11** | Pick now | Judge quality dominates eval trustworthiness — decide after measuring agreement against hand labels. |
| **2026-07-28** | **v2 revision — build/import rebalanced.** BM25 → `bm25s`; retrieval metrics → `ranx`. Kept building: custom analyzer, heading-aware chunker, RRF, confidence scorer, citation validator, routing policy, eval harness. | v1: hand-roll BM25 and metrics from scratch | User already knows the topics (stated explicitly) — the goal is hiring signal, not re-learning. Writing a mature library's job proves nothing to an interviewer and costs days better spent on the ablation grid and abstention system, which *are* the differentiators. |
| 2026-07-28 | Cut ColBERT implementation → design-record only | Full implementation | ~0.5 day for a result reasonable to predict without building it; bi-vs-cross-encoder comparison already covers the core lesson |
| 2026-07-28 | Three-way encoder bake-off → two (bge-large, BGE-M3) | Keep bge-small as a third full candidate | bge-small kept only as the fast dev/iteration workhorse, not a final candidate — the third full comparison was ~0.5 day for a near-certain result |
| 2026-07-28 | Added Phase 17 — LangGraph port | Skip frameworks entirely; build on LangChain/LangGraph from the start | Framework fluency is a real screening filter. Building on a framework from day one hides the user's own architecture and makes debugging harder. Porting a working, measured system afterward gets both the keyword and a defensible, evidence-based opinion ("here's what it cost, here's what it bought") rather than a preference. |
| 2026-07-28 | Time freed by the above (~1 day) reallocated to the LangGraph port | Compress schedule to 20 days instead | Keeps 21-day target with more buffer in Phases 5–7, which were previously stretched thin |
| 2026-07-29 | Kept folder name `SourceDesk`, did not rename to `aegis` | Rename per plan.md literally | The rename's stated reason (space in path breaking Docker volume mounts) doesn't apply — the actual path has no space. Renaming would also disrupt the already-open editor session for no benefit. The Python package is still named `aegis` (`src/aegis/`), so nothing downstream changes. |
| 2026-07-29 | Qdrant image pinned to `v1.18.3`, not the plan's unstated default | Leave at whatever `docker pull qdrant/qdrant` resolves to | `qdrant-client` 1.18.0 (latest at `uv add` time) warns on incompatibility with server versions more than one minor apart; started on v1.12.4 by accident (stale copy-paste), hit the warning immediately, re-pinned to v1.18.3 to match. Cheap to fix now, expensive to debug later as a "why are recall numbers weird" mystery. |
| 2026-07-29 | Baseline embedding model: `BAAI/bge-small-en-v1.5` via `fastembed` (ONNX) | `sentence-transformers` + MPS | No torch install, no MPS setup, for a component that's thrown away at the end of this phase. Real bake-off (bge-large vs BGE-M3) is Phase 3 and will decide the model actually used going forward. |
| 2026-07-29 | Baseline corpus = 85 files (`tutorial/` + `advanced/` only), not the full 155-file docs tree | Ingest everything now | Phase 0's job is a fast, deliberately-bad demo, not real corpus acquisition — that's Phase 1's structure-preserving parser. The subset was large enough (121,553 tokens) to produce real, reproducible hallucinations on the first run. |
| 2026-07-29 | Generation LLM for the baseline: Groq `llama-3.3-70b-versatile` only (no Gemini fallback yet) | Wire up the fallback now | Fallback logic is a production concern (Phase 9+); Phase 0 only needs one call path to prove the architecture fails. |
| 2026-07-29 | Phase 1 exit criterion revised: ≥300 docs → ≥125 docs (final: 129 loaded) | Keep ≥300 and pad the corpus (all locales, or include `reference/`) | Measured: English `docs/en/docs/` is 155 `.md` files total, 131 excluding `reference/` (24 files of `::: fastapi.ClassName` mkdocstrings auto-doc stubs — no renderable prose without the mkdocstrings plugin actually running against the installed package), 129 after also dropping `_llm-test.md` (FastAPI's own translation-LLM prompt test fixture — not documentation, would inject meaningless filler text into retrieval) and `translation-banner.md` (no H1, a transclusion snippet with no standalone content). 300 was an estimate made without checking the repo; padding with other-language locales or content-free stub pages would hit an arbitrary number without adding real signal — the measurement wins. |
| 2026-07-29 | `product_area` derived from a shared keyword table (path-based for docs, title+body text for issues), not from GitHub labels | Derive issue `product_area` from labels, as plan.md sketched | Measured: fetched the live label list for `fastapi/fastapi` — it's all process/triage labels (`bug`, `p1`-`p5`, `lang-*`, `approved-N`, `dependencies`). No topical labels exist (no `security`, `async`, `cors`, etc.), and the one label that looks topical (`dependencies`) is dependency-version-bump tooling, not the FastAPI DI feature. Labels can't carry the signal the router (Phase 8) needs; text keyword matching can, and using the same taxonomy for both doc types (rather than two incompatible schemes) is what a router actually needs. |
| 2026-07-29 | `{* path *}` include directives resolve relative to `docs/en/` (the mkdocs.yml directory), a fixed base for every file — not relative to each markdown file's own directory | Resolve relative to each file's own parent directory (standard filesystem-relative semantics) | Measured: the literal include text `../../docs_src/...` is identical across files at different nesting depths (`tutorial/query-params-str-validations.md`, 1 level deep, and `tutorial/security/first-steps.md`, 2 levels deep, both use the exact same `../../docs_src/...` prefix). Standard per-file relative resolution would require a longer `../` prefix for deeper files; it doesn't vary, so FastAPI's own mkdocs plugin must resolve these against a fixed base equal to the mkdocs.yml directory, confirmed by resolving against `docs/en/` and checking the target file exists. |
| 2026-07-29 | Also resolve `{!path!}`/`{!> path!}` (pymdownx.snippets syntax) | Ship with `{* *}` support only | Found by manually opening `advanced/templates.md`'s parsed output (the exit-criterion-mandated step, not an automated check) — 3 files use this alternate include syntax instead of `{* *}`, and every automated leftover-syntax check I'd written passed cleanly despite the content being missing. Direct validation of why "actually open them" is a hard requirement, not a nice-to-have. |
| 2026-07-29 | Excluded `_llm-test.md` and `translation-banner.md` from the doc corpus (131 → 129) | Include everything under `docs/en/docs/*.md` | Both are non-content: `_llm-test.md` is FastAPI's own test fixture for their translation-LLM prompts (495 lines of test scaffolding, not documentation); `translation-banner.md` is a transclusion snippet with no H1 and no standalone content. Including either would inject meaningless filler into retrieval. |
| 2026-07-29 | Ingested all 3,530 closed issues (not capped at 2,000–3,000), tagging which have a resolution comment rather than pre-filtering | Filter down to only resolved issues before ingesting, to match the plan's target range | Filtering first would have thrown away the actual required measurement (Phase 1's Measure step: "count of issues with an identifiable resolution — the ceiling on your golden set"). Ingesting everything and reporting the resolved fraction (2,204/3,530, 62.4%) makes that number a fact instead of a choice. |
| 2026-07-29 | `product_area` for issues strips fenced code blocks and a specific issue-template boilerplate line before keyword matching | Match keywords against raw issue text directly | Measured via manual sampling: raw-text matching classified `async` on 695/3,530 issues (20%) — nearly every issue includes an `async def` repro snippet regardless of topic. Worse: FastAPI's old issue template's checklist literally says "I already checked if it is not related to FastAPI but to [Pydantic]/[Swagger UI]/[ReDoc]" on 1,416/3,530 issues (40%), which was silently dominating the `pydantic`/`openapi` categories from boilerplate, not content. Both fixed by stripping before matching (verified with regression tests that fail without the fix). A third, narrower source (canned maintainer replies, e.g. "template" in a stock "open a Discussion" response) was found but not chased further — documented as a residual limitation instead, since cataloging every canned reply is open-ended. |
| 2026-07-29 | Verified hash-reproducibility via `content_hash` as a pure function on fixed input, not by re-running the live GitHub fetch twice | Re-run `aegis.ingest.run` twice and diff the output, as originally planned | Running the full ~3,630-request fetch twice in one session hit GitHub's 5,000/hour core rate limit on the second run's 3,530th call. Recovered by reprocessing the already-fetched raw data locally (saved before the second run) instead of re-fetching — which is also the methodologically correct test: the live issue API is a mutable source (comments can be added between two real fetches), unlike the frozen git-cloned docs corpus, so two live fetches aren't expected to be byte-identical regardless of code correctness. |
| 2026-07-30 | `tiktoken` (`cl100k_base`) added as a dependency for chunk-size token counting, not the eventual embedding model's own tokenizer | Load the real bge-large/BGE-M3 tokenizer (not chosen until Phase 3) | Chunk boundaries only need a count that's stable and consistent with itself across strategies, not exact parity with an encoder that doesn't exist yet in this project. `max_tokens=400` is set with headroom below any real encoder's 512-token limit to absorb this approximation plus the contextual-enrichment prefix. |
| 2026-07-30 | Code-block atomicity guaranteed only for heading-aware and semantic, deliberately not extended to fixed/recursive | Give every strategy the same atomicity guarantee | Fixed and recursive are supposed to be the naive/moderate baselines; their measured code-integrity rate (84.1% / 85.3% at `max_tokens=400`) is itself the useful data point for Phase 6, not a defect to quietly fix. |
| 2026-07-30 | `greedy_merge` (shared by recursive/heading-aware/semantic) verifies packing by actually joining candidate pieces and re-counting tokens, not by summing each piece's own token count | Keep the sum-based running-total approach (simpler, was the original implementation) | The sum-based version does not account for the join separator's own token cost, and is not the same number as the true joined text's token count in general (BPE isn't additive across concatenation) — found as two separate real violations of `max_tokens` on real corpus documents (issue #15000 at 443/400, and a semantic-chunking test at 26/25) before being replaced. See `docs/explainers/02-chunking-strategies.md` §6. |
| 2026-07-30 | `recursive_split_pieces` drops the original separator instead of re-attaching it to each piece | Re-attach the separator to preserve exact source whitespace when pieces are concatenated | The re-attached separator plus `greedy_merge`'s own join separator doubled up at every merge boundary, inflating real chunk token counts past `max_tokens` (issue #15000, 443/400) — found via full-corpus measurement, not caught by unit tests using clean small inputs. |
| 2026-07-30 | Closing-fence detection (`segments.py`) accepts any line that *starts* with `` ``` ``, not only a line that *is* `` ``` `` (optionally with trailing whitespace) | Strict CommonMark-style closing-fence match | Real issue text breaks a strict match twice: CRLF line endings (closing line is literally `` ```\r\n ``) and a closing fence sharing its line with trailing HTML (`` ```</div> ``). Either one, unhandled, made the closing-fence branch never match, silently swallowing every heading and paragraph after it into one giant "code" segment (issue #15000 went from 2 detected sections to the correct 9 once fixed). |
| 2026-07-30 | Accepted a residual limitation: 7 of 19,236 heading-aware chunks (0.036%) still exceed `max_tokens`, not chased further | Build a more sophisticated fence parser (e.g. adopt a mature CommonMark library) to also handle these | All 7 affected documents have a genuinely *odd* number of literal `` ``` `` occurrences in the raw issue text (e.g. `issue::4548` has exactly one in the whole document) — an actually-unclosed fence in the source, which no stateless open/close regex can pair correctly by construction. Same category of call as Phase 1's decision to stop cataloging canned maintainer replies: bounded, quantified, confined entirely to the noisiest part of the corpus (issues, zero in docs), worth a real parser only if it later proves load-bearing. |
| 2026-07-30 | LLM-based chunking (one of the course's 7 chunking kinds) not implemented — design-record-only, same treatment as ColBERT in the Phase 7 plan | Implement a fifth strategy using an LLM to place chunk boundaries | An LLM call per boundary decision across 3,659 documents is a real cost for a technique whose main advantage — better judgment at ambiguous boundaries — is hard to distinguish from heading-aware's structural approach on a corpus that already has strong, free structural signal (Markdown headings, fenced code). Revisit only if Phase 6 shows all four built strategies plateau below what the golden set demands. |
| 2026-07-30 | Semantic chunker's sentence embedder is dependency-injected (`embed_fn`), defaulting to a lazily-loaded real `fastembed` model | Always load the real model, even in unit tests | Unit tests inject a small deterministic fake embedder so the grouping/breakpoint logic is testable in milliseconds without an ONNX model load; the real model path is exercised by the measurement script instead (which found the ~211–248s cost per large document — see Measured Results). |
| 2026-08-02 | Persisted `data/processed/chunks.jsonl` using heading-aware as a **provisional** chunking strategy | Wait for Phase 6's ablation grid before persisting anything | Phase 3 needs a real chunked corpus to embed before Phase 6 exists to pick a winner empirically. Heading-aware has the best measured code-block integrity (98.4%) and is the only strategy with real heading attribution (Phase 2). Phase 6 re-chunks with all four strategies and can overturn this choice with real retrieval numbers. |
| 2026-08-02 | `sentence-transformers` (not `fastembed`) for the Phase 3 encoders | Keep using `fastembed`/ONNX as in the Phase 0 baseline | Phase 0's ONNX path is a from-scratch download in a different format regardless; `sentence-transformers` is plan.md's named choice for the real bake-off and gives direct MPS device control, which fastembed's ONNX runtime doesn't expose the same way. |
| 2026-08-02 | Embedding cache is a hand-rolled SQLite table keyed on `(model, content_hash)`, not a library like `diskcache` | Add `diskcache` or similar | The need is narrow (two columns, point lookups by a composite key) and Python's stdlib `sqlite3` covers it in ~70 lines with zero new dependencies; a general-purpose cache library would add surface area for a problem this small. |
| 2026-08-02 | Ran `bge-large` and `BGE-M3` embedding **sequentially**, not concurrently, despite being independent jobs | Run both at once to halve wall-clock wait time | CLAUDE.md's risk table flagged M1 16GB memory pressure before this phase started; confirmed real via `vm.swapusage` (~3.5GB in use) during the first (unmitigated) bge-small run. Running two large transformer models on MPS at once was judged too likely to compound it — see the throughput numbers in Measured Results for what the pressure already cost once, unmitigated. |
| 2026-08-02 | HNSW parameter sweep run only on `bge-small`; `bge-large`/`BGE-M3` get one config chosen from that curve, not their own sweep | Sweep all three encoders | The graph algorithm's recall/latency behavior doesn't depend on which encoder produced the vectors; only corpus size and structure matter for that. Sweeping the fast encoder gives a real 36-point grid in seconds; sweeping all three would have added ~an hour for no new signal. |
| 2026-08-02 | ANN recall measured against self-sampled corpus vectors as queries, not real user queries | Wait for Phase 4's golden set before measuring recall at all | Phase 3 precedes Phase 4 in the schedule — no golden queries exist yet. Self-sampled queries measure HNSW's fidelity to the exact nearest-neighbor structure of the embedding space (a real, useful number), explicitly not retrieval quality for real questions (Phase 6's job once Phase 4 exists). |
| 2026-08-02 | Built a score-threshold recall metric (`recall_at_k_score_aware`) instead of trusting ID-set recall | Use standard ID-set recall (`overlap(approx, exact) / k`) | ID-set recall reported a flat, impossible-looking ~0.67 across every HNSW parameter. Root cause: 39.7% of the corpus's vectors are exact duplicates of another vector (largest group: 1,354 issues whose entire chunk is the GitHub issue-template placeholder `_No response_`), so "the" top-10 nearest neighbors isn't uniquely defined — ID-set recall penalizes a correct-but-differently-tied answer. Score-threshold recall (credit any result meeting or beating the true k-th-best score) fixed it: recall became 1.0000 everywhere, which is what should be expected at this corpus scale. |
| 2026-08-02 | `recall_at_k_score_aware` always divides by the requested `k`, never by the number of scores actually returned | Divide by `len(scores)` (the original implementation) | Dividing by the returned count let a method that returns fewer results than asked — e.g. naive post-filtering discarding most of its candidate pool — score a perfect 1.0 as long as the few it did return were correct. Caught because the pre/post-filter experiment produced a self-contradictory result ("short of k results on 100% of queries" alongside "recall=1.0000"), which is only possible if the metric itself was broken. Now regression-tested (`tests/test_knn_recall_score_aware.py`). |
| 2026-08-04 | Chunking axis scored at **document level** (a hit if any retrieved chunk's `doc_id` matches a ground-truth doc), not chunk level | Keep chunk-level scoring everywhere for consistency | Ground truth is heading-aware chunk IDs; fixed/recursive/semantic chunkers each embed their own strategy name in the chunk ID, so those IDs don't exist in the other three strategies' index at all. Chunk-level scoring would silently score every non-heading-aware strategy near zero regardless of actual quality. Confirmed with the user before running (`eval/doc_level.py`, TDD'd). |
| 2026-08-04 | Added `exclude_doc_id` as a first-class parameter on `SparseIndex.search`/`DenseIndex.search`/`HybridRetriever.search` | Wrap it as an eval-only post-hoc filter outside the retrieval interfaces | Golden queries' self-match leak (their own source issue is a trivial near-verbatim hit) is not just a Phase 6 concern — every future eval-dependent phase (7, 10, 11, 16, 17) calls the same retrieval interfaces against the same golden set and would hit the identical bug. Native `must_not` filtering (dense) / over-fetch-then-filter (sparse, no native filter API) belongs in the interface, not duplicated per script. |
| 2026-08-04 | "Lift from baseline" (Phase 6 exit criterion) computed as worst-vs-best **within** the controlled ablation grid (sparse-only vs. winning config), not against Phase 0's literal `aegis_baseline` collection | Use the real historical Phase 0 baseline, since it's the literal artifact plan.md's DoD seems to reference | Checked the actual collection: it indexed 85 of 129 doc pages, raw 500-char slices, and zero of the 3,530 issues — the golden set is mostly issue-sourced. Scoring it would measure "wrong corpus," not "worse retrieval config," a confound rather than a fair, controlled comparison. |
| 2026-08-04 | Contextual enrichment set to **off** in the winning config, reversing plan.md's stated expectation ("usually the cheapest large win") | Ship enrichment on since it's the documented default recommendation | Measured, twice (before and after fixing a real title-duplication bug): enrichment costs 0.4288 → 0.1751 recall@10. Root cause confirmed by inspecting real embedding text: GitHub issue-template boilerplate headings (`Description`, `Question`, `How to do it`) are near-identical across thousands of unrelated issues, and prepending them compresses dense retrieval's discriminative signal — the same boilerplate-pollution failure mode Phases 1-4 each independently hit, here inside enrichment specifically. |
| 2026-08-11 | **Reranking set to off** in the default config, reversing the entire premise of Phase 7 | Ship it on at the "least bad" pool size (10) for the portfolio narrative | Measured: every swept pool size (10/20/50/100) makes NDCG@5 *worse* than the no-rerank baseline (−9.0% to −55.8%, monotonic), and every pool size's own p95 latency alone already exceeds the full pipeline's 3s budget. Root cause investigated directly (not just accepted): the golden set's ground truth (`auto_label.py`, token-overlap ≥15%, "biased toward inclusion") has a median of 17 chunks across 12 documents per query — a broad-topic standard a fuzzy bi-encoder satisfies better than a sharp cross-encoder. Kept as a tested, toggleable component rather than removed. See `docs/explainers/07-reranking-and-encoder-architectures.md`. |
| 2026-08-11 | Moved the Phase 7 pool-size sweep's compute to a free Kaggle T4 GPU, off this project's own M1 Pro | Keep all compute local, per the project's established pattern everywhere else | Measured, not assumed: MPS hit macOS's "stuck" scheduler state twice under this session's real memory pressure (once from two large transformer models MPS-resident at once, once again after fixing that). CPU inference was stable but real corpus-length text (~400 tokens/chunk) benchmarked at ~780ms/pair vs. ~30.5ms/pair on short synthetic text used for the initial estimate — projecting ~5.7 hours, infeasible in-session. Kaggle's free T4 tier sidesteps the local 16GB ceiling entirely. |
| 2026-08-11 | Kaggle sweep script fixed to `BATCH_SIZE=16` + `torch.cuda.empty_cache()` between pool sizes + incremental per-pool-size result writes, after a first GPU attempt crashed | Retry the original `BATCH_SIZE=64` script as-is on the assumption the OOM was a fluke | `BATCH_SIZE=64` exceeded every swept pool size, so each query's entire candidate pool was scored as one un-chunked batch; at pool=50 (50 pairs, up to ~400 tokens each) that, stacked on fragmented memory from ~300 prior forward passes with no cache-clearing, exhausted the T4's ~14.56GB (CUDA OOM: 3.54GiB requested, 2.95GiB free, 11.61GiB already resident). Incremental writes meant the pool=10/20/50 data already collected before the crash was never lost. |
| 2026-08-12 | Router ground truth (58-query confusion matrix) reuses `strata.py`'s existing how_to/error_debugging classification rather than a fresh labeling pass | Fresh manual read-and-label pass across all 6 strata, or LLM-based auto-labeling | Confirmed with the user. `strata.py`'s cues are already about *how a query is phrased* ("how do i" vs. "error"/"crash"), the same distinction the router itself makes — reusing it is reusing the same signal, not repurposing an unrelated one. Checked first whether ground-truth chunk doc_type composition would work instead: it doesn't (46-58% doc-chunks in every stratum, no separation, a side effect of Phase 4's broad auto-labeling). LLM-based auto-labeling was rejected as the same "LLM grading an LLM" circularity the router classifier itself has. |
| 2026-08-12 | Query-rewriting eval set (24 items) is LLM-generated synthetic multi-turn conversations, not hand-written | Hand-write ~15-20 pairs myself for full traceability (my own initial recommendation) | User picked LLM-generation instead. This corpus (FastAPI docs + closed issues) has no real multi-turn data either way, so the *conversation* is synthetic regardless of who writes it; scoring stays traceable because each synthetic item is anchored to a real golden query's real `ground_truth_chunk_ids`. Spot-checked 3 of 24 against their source issue text after generation — good fidelity, including one correctly-translated Chinese-language item (`golden::issue::2896`). |
| 2026-08-16 | Router classifier fails safe to `billing_account_legal` on *any* parse failure (malformed JSON, out-of-vocabulary label, or a raw Groq `APIError`) | Fail open to `docs_howto` (fewer unnecessary escalations) | CLAUDE.md's North Star: an unnecessary escalation costs a human ticket; silently mishandling a possible billing/legal query is the Air Canada failure mode this project exists to prevent. Not a hypothetical — a real `json_validate_failed` API error was hit mid-router-eval (`RouterSettings.max_tokens=20` was too tight for Groq's JSON-mode grammar constraint on some real queries) and is now handled the same way as every other parse failure, not a special case. |
| 2026-08-16 | Three of four Phase 8 query-understanding techniques (multi-query expansion, HyDE, iterative retrieval) shipped **off** by default; only query rewriting is **on** | Ship all four on since they're all "standard" agentic RAG techniques | Measured, and each root-caused rather than just reported (`docs/results/query-understanding-ablation.md`): expansion −49.4% NDCG@5 (small model hallucinates answers instead of paraphrasing, plus unweighted 4-way RRF lets 3 wrong lists outvote 1 correct one), HyDE −23.6% (same "sharp technique vs. broad ground truth" mechanism Phase 7 found for the cross-encoder reranker — second independent confirmation of that pattern), iterative retrieval −13.4% (a Phase-10-shaped confidence proxy used out of necessity before Phase 10 exists, entire regression concentrated in the 38/150 queries it touched). Rewriting is the only technique solving a problem the single-turn baseline doesn't already handle well, and it measurably helps on the eval built for that problem (+52.7% NDCG@5 vs. context-free). |
| 2026-08-16 | Query-understanding ablation script's encoder forced to `device="cpu"`, overriding the configured `mps` default | Use the configured MPS default like every other component | The script's first run hung 17+ minutes at 0% CPU with ~3GB swap in use — the identical MPS-under-memory-pressure signature Phase 3 and Phase 7 both already hit and documented in this log. Fixed the same way Phase 7 did. Recorded here a third time specifically because it's a repeat of an already-documented lesson, not a new discovery — see design record 08 §6. |
| 2026-08-16 | Gemini added as the fallback provider (`google-genai` SDK, `gemini-2.5-flash`), user provisioning the API key mid-session | Groq-only for this session; descope the bake-off exit criterion honestly instead | Confirmed with the user (`AskUserQuestion`): no `GEMINI_API_KEY`, no Gemini SDK, and no local Ollama install existed at Phase 9 start. User chose to add a real key rather than build untested or skip the exit criterion — key still pending as of phase-end, see Open Questions. |
| 2026-08-16 | Context budgeter uses strict-prefix greedy selection (stop at the first chunk that doesn't fit), not skip-ahead bin-packing | Skip a chunk that doesn't fit and keep trying smaller lower-ranked ones, to use the budget more fully | Skip-ahead could seat a rank-7 chunk while dropping rank-2 purely because rank-2 was bigger — relevance order should govern which chunks are *selected*, not just where they're placed. One deliberate exception: the single highest-ranked chunk is truncated rather than dropped if it alone exceeds the whole budget (real chunks reach 6,325 tokens on oversized atomic code blocks, per Phase 2/6). |
| 2026-08-16 | Citation numbers (`citation_id`) track retrieval rank, assigned during selection, independent of the bookend algorithm's display position | Number chunks by their position in the final rendered text | Bookend placement means the *n*-th displayed chunk holds a different rank depending on how many chunks survived the budget for that call — numbering by display position would make "[3]" mean something different call to call. Numbering by rank means "[1]" always means "the single most relevant retrieved chunk," which is what both a human reader and the citation validator need to trust. |
| 2026-08-16 | `_lexical_overlap` (citation validator, tier 1) reimplemented as exact word-boundary token-set intersection, not `eval/auto_label.py`'s `token in candidate_text` substring check it was modeled on | Reuse the Phase 4 technique verbatim for consistency | TDD caught it: a test claim about "the sky is blue" against an unrelated chunk about "dependency injection" passed tier 1 as `SUPPORTED_LEXICAL`, because `"on"` is a substring of `"injection"` (and `"the"`/`"is"`/`"in"` had similar spurious hits). Found by the test suite itself, not by inspection — fixed before shipping. Phase 4's original technique isn't retroactively wrong for its own (coarser, query-vs-chunk) purpose, but wasn't safe to reuse unmodified for a stricter per-claim check. |
| 2026-08-16 | `FallbackLLMClient` fallback boundary is narrow: only a failure *before* the first streamed token triggers switching providers; a mid-stream failure propagates | Fall back on any failure, including mid-stream, to maximize availability | Splicing a second provider's fresh generation onto a first provider's partial output would produce an answer no single model actually wrote — undermining the citation-grounding premise this whole phase exists for. Auth/rate-limit/model-not-found errors (the realistic failure modes, confirmed by this phase's own real `NotFoundError` and `RateLimitError` hits) surface before any token is produced anyway. |
| 2026-08-16 | `FallbackLLMClient.generate_with_provider()` reports which provider answered via a stateless internal generator (`_stream_with_provider` yielding `(name, token)` pairs), not a shared mutable `self.last_provider` field | Track the last-used provider as instance state, simplest to implement | Phase 6 already found and fixed a real thread-safety bug from exactly this shared-mutable-state-on-a-shared-instance pattern (`EmbeddingCache`'s single `sqlite3.Connection`). Not repeating it on a component (`FallbackLLMClient`) explicitly designed to be reused across every future generation call, including concurrent ones once Phase 14's API exists. |
| 2026-08-16 | Sampling sweep grid cut from 6 configs x 20 queries (`max_tokens=1024`) to 4 configs x 6 queries (`max_tokens=400`) mid-session, after the first config alone used 95,944 of Groq's 100,000-tokens/day quota | Keep the original grid and wait out the quota reset (~30-90 min based on observed error messages) | Confirmed with the user (`AskUserQuestion`) after the real constraint was measured, not guessed in advance. The scoped-down retry *also* hit the exhausted quota (only 2/24 calls succeeded) — a second real measurement, not a hypothetical, that a single day's free-tier budget cannot currently support this experiment's realistic per-query token cost. |
| 2026-08-16 | Reported the sampling sweep as a documented partial result (one real, complete N=20 config; the rest marked not-measured) rather than waiting out the quota reset in-session | Pause and wait ~30-90 minutes for the quota to reset, then complete the full grid before moving on | Confirmed with the user. The one real config already demonstrates the full pipeline works end to end against live infrastructure and produces a real, usable faithfulness-proxy number; waiting idle for an external, uncertain-duration reset was judged worse than moving on to other real Phase 9 work and revisiting the grid in a follow-up session. |
| 2026-08-16 | Gemini model pinned to `gemini-3.5-flash`, not the originally configured `gemini-2.5-flash` | Trust `client.models.list()`, which does list `gemini-2.5-flash` as available | A real generation call against `gemini-2.5-flash` returned `404 NOT_FOUND`: "no longer available to new users." The list endpoint doesn't reflect real per-API-key availability — model choice needs a real successful call, not just a successful lookup. `gemini-flash-latest` (an alias) also worked but was rejected in favor of a pinned version, for the same reproducibility reasoning as every other pinned model in this project. |
| 2026-08-16 | `GeminiProvider` sets `thinking_config=ThinkingConfig(thinking_budget=0)` on every call | Leave Gemini's default (extended thinking enabled) | A real call at `max_tokens=20` came back with `finish_reason=MAX_TOKENS` and zero visible characters — the whole budget was spent on hidden reasoning tokens before any answer text. Confirmed directly against the live API, not assumed from documentation. Also the right default independent of the symptom: this provider's role is a fast fallback for a support-answer task, not multi-step reasoning. |
| 2026-08-16 | Ran the real provider bake-off at N=15 despite Groq's quota being mostly exhausted, accepting an unequal (2 vs. 12) real sample rather than waiting for Groq's quota to reset first | Wait for Groq's quota to reset so both providers get an equal, fair N before running the bake-off | The 2 Groq successes plus all 12 Gemini successes still produced two genuinely decision-relevant findings (the ~14x latency gap; two same-query head-to-head comparisons) that a "wait and get equal N" approach would have delayed for an uncertain amount of time. The unequal-N caveat is stated explicitly in `docs/results/llm-bakeoff.md` rather than glossed over. |
| 2026-08-16 | Wrote and registered prompt v2 (explicit first-sentence citation rule + worked example) but did **not** promote it to the shipped default (`GenerationSettings.prompt_version` stays `"v1"`) | Ship v2 as the default now, since it's clearly intended to fix a real, measured v1 gap | The v1-vs-v2 comparison script (`experiments/09c_prompt_v2_check.py`) hit Gemini's 20-requests/day quota after only 2 real v1 completions and 0 real v2 completions — nowhere near enough evidence to justify a default change. CLAUDE.md is explicit: a decision without a measurement is a default, not a decision. v2 stays available (`get_prompt("v2")`) for the next session's validation pass. |
| 2026-08-17 | Fixed `CitationValidator`'s sentence-splitter/code-block bug (Open Question #11) by reusing `chunking/segments.py`'s `split_code_segments` to strip fenced code before sentence splitting | Write a new fence-detection regex specific to `guardrails/citations.py` | `split_code_segments` already exists, is already hardened against real corpus edge cases (CRLF line endings, a closing fence sharing its line with trailing HTML) from Phase 2, and solves exactly this problem ("don't let a code fence corrupt segment boundaries"). Writing a second fence parser would duplicate already-debugged logic for no benefit — same reuse call Phase 1 and Phase 2 each made for their own edge-case handling. |
| 2026-08-17 | Promoted `GenerationSettings.prompt_version` from `"v1"` to `"v2"`, reversing the prior "stays v1 until validated" decision | Wait for a larger (N≥15-20) confirmatory run before promoting, as originally planned | Reversal driven by real data, not a changed mind: a third real attempt (after two quota-blocked ones) completed a full matched-pair comparison once Groq's quota genuinely reset — v2 won or tied on 6/6 real queries (5 wins, 1 tie, 0 losses), +34.3% relative mean improvement, and the mechanism was confirmed on real generated text (an uncited opening sentence under v1, cited under v2, same question, same context). N=6 is smaller than the original N≥15-20 target, but CLAUDE.md's actual bar is "measured, not invented," not a specific sample size — a consistent, mechanistically-explained, zero-loss result clears that bar. A larger confirmatory run remains worthwhile but isn't blocking. |
| 2026-08-17 | Confidence-scorer signal weights reweighted by measured `\|rank-correlation\|` (0.60/0.15/0.20/0.05) instead of the equal (0.25 each) weighting plan.md's step 1 suggests as a starting point | Ship equal weights as literally stated in plan.md | Measured on the real 180-query dev sweep: equal weighting scored *worse* than `reranker_top_score` alone (0.8333 vs 0.9333 recall at precision>=0.5) because `score_gap` (rank-correlation -0.16) and `distribution_shape` (-0.02) are both close to uncorrelated with the real should-escalate label and dilute the one strong signal (-0.58). Confirmed with the user (`AskUserQuestion`) given this is a real deviation from plan.md's literal starting point, not a routine judgment call. All 4 signals plan.md names are kept in the combination (a "drop the weak ones entirely" alternative measured marginally better, 0.9667, but was rejected for methodological transparency — see design record 10 §3). |
| 2026-08-17 | Confidence-signal normalization bounds are winsorized (1st/99th percentile of the real dev sweep), not raw min/max | Use the literal observed min/max as normalization bounds | `distribution_shape`'s raw max was a single ~76,856 outlier (one query's near-zero-variance top-k dense scores, tying to Open Question #6's near-duplicate-chunk corpus property) — under raw min-max, every other query's real range would compress into a sliver below 0.001 of [0,1], making the signal contribute almost nothing regardless of its assigned weight. |
| 2026-08-17 | Escalation threshold **tau=0.50** chosen from the real precision/recall curve | Two alternatives with real, presented numbers: tau=0.15 (deflection-preserving, 98% deflection but only 70% recall — 9/30 negatives slip through) and tau=0.65 (high-recall, 97% recall but deflection collapses to 50%) | Confirmed with the user (`AskUserQuestion`) given plan.md's own framing of this as "the decision that matters most." tau=0.50 catches 27/30 (90%) of dev negatives — only 3 slip through, all `unanswerable`/`out_of_corpus` (`billing_legal`/`adversarial_injection` both 100% caught) — while still directly answering 126/150 (84%) of legitimate questions. Chosen as the point that most reduces the asymmetric-cost Air Canada quadrant (bounded ~$25/ticket over-caution vs. unbounded reputational risk, per plan.md) without going degenerate the way tau=0.65's 50% deflection rate would. |
| 2026-08-17 | Confidence scorer's cross-encoder reranking pool capped at 3 candidates (`ConfidenceSettings.rerank_pool_size`), not the full top-10 | Rerank the same top-10 pool used for the other two (cheap) signals, for one consistent pool size across all 4 signals | `reranker_top_score` and `score_gap` only ever read rank 1 and rank 2 — reranking rank 3-10 would spend real, measured compute (~780ms/pair, corpus-length text, Phase 7) on signal nothing uses. At pool=3, the real 180-query dev sweep still took 736s (12.3 min), over CLAUDE.md's 10-min local-compute line despite a smaller real probe projecting ~5-9 min — reported honestly (design record 10 §6) rather than re-run on Kaggle after the fact, since it completed in the background without blocking the session and the overrun was modest. |
| 2026-08-17 | tau swept and the confusion matrix measured against the golden set's **30 dev negatives only**, not the literal "all 40 negatives" plan.md's Phase 10 exit-criterion text names | Include the 10 holdout negatives too, to satisfy the literal "40" in plan.md | plan.md's Phase 10 wording predates the frozen dev/holdout split decision (Phase 4) and the standing rule established since — "Phases 6-16 measure on dev, Phase 19 on holdout for generalization" (this progress log). Phase 10 falls inside that range; touching holdout negatives now would leak them into a calibration decision (tau), the exact thing the holdout split exists to prevent. Same category of call as Phase 6's resolution of "lift from baseline" against plan.md's literal but superseded wording. |
| 2026-08-17 | Rejected using Phase 9's `ValidationResult.valid_citation_rate` as a 5th confidence signal, despite design record 09 §7 explicitly flagging it as a natural next step | Add it as a 5th signal, since it's already built and already measures grounding quality | Architecturally backwards for a pre-generation gate: `valid_citation_rate` only exists after an LLM has already generated an answer, and the entire point of this phase's confidence scorer is deciding whether to pay for that generation call at all. A good signal for Phase 11's generation-quality eval, not for this gate. |
| 2026-09-25 | **`qwen/qwen3.8-27b` replaces `llama-3.3-70b-versatile`/`llama-3.1-8b-instant`/`groq/compound-mini` for all five Groq-backed config roles** (`GroqSettings.model`, `RouterSettings.model`, `QuerySettings.rewrite_model`/`expansion_model`/`hyde_model`, `CitationSettings.entailment_model`); `GroqSettings.max_tokens` 1024→1000 | `openai/gpt-oss-120b` for primary generation only (still needing a second small-task model given `compound-mini`'s death); reconsidering Gemini as primary | Confirmed with the user (`AskUserQuestion`). Real 6-query answer-length measurement: gpt-oss-120b's reasoning tax is survivable (25-45% of completion_tokens, never empty) but its 16.7s mean latency nearly matches Gemini's already-rejected 21.4s, undermining "Groq primary because it's fast." qwen3.8-27b measured 0 reasoning tokens and 8.7s mean latency (~2.5x faster than Gemini) once `max_tokens` respected a real, measured 1000 output-tokens/minute free-tier cap. One verified model for all five roles, given a second independently-verified model (`compound-mini`) had already died once this project relied on it. Full detail: `docs/results/groq-model-replacement.md`. |
| 2026-09-25 | Hand-wrote `BaseRagasLLM`/`BaseRagasEmbeddings` subclasses (`GroqRagasLLM`/`GeminiRagasLLM`/`RagasLocalEmbeddings`) wrapping the raw `groq`/`google-genai` SDKs, rather than adding `langchain-groq`/`langchain-google-genai` to use `ragas.llms.LangchainLLMWrapper` | Add the two LangChain provider packages and use ragas's own documented non-OpenAI wrapper | `Faithfulness`/`ResponseRelevancy` need a `BaseRagasLLM` (confirmed by reading ragas 0.3.9 source directly after a real `AttributeError`, not from docs) -- `langchain-core`'s `LLMResult`/`ChatGeneration`/`AIMessage` types the hand-written subclasses need were already a transitive cost of the `ragas<0.4` pin (2026-08-18), so this is a ~15-line shim per provider, not new dependency surface. Full mechanism: design record 11. |
| 2026-09-25 | **User declined to hand-label any answers.** Phase 11's judge-validation step becomes **inter-judge agreement** (Gemini `gemini-3.5-flash` vs. `openai/gpt-oss-120b` on Groq, Cohen's κ between the two) instead of judge-vs-human-label κ; the resumable hand-labeling CLI originally scoped for Unit 2 is dropped. | (a) Reference-based proxy scoring against the golden set's real `resolution_hint` maintainer-resolution text; (b) drop κ/judge-validation entirely, ship RAGAS metrics only | Flagged directly to the user first (CLAUDE.md: disagree when warranted) — Phase 11 judge validation is on CLAUDE.md's never-cut list specifically *because* it's the evidentiary spine most candidates skip, and plan.md's own text calls hand-labeling "a strong signal" precisely because it's rare. No alternative here is as strong as real human ground truth, and this must be disclosed as a real, honest limitation in design record 11(+), not framed as equivalent. Inter-judge agreement was chosen over the two alternatives (presented via `AskUserQuestion`) because it preserves the actual measurement methodology (κ, disagreement-case review, judge failure-mode analysis) rather than replacing it with a differently-scoped metric (the reference-based proxy) or dropping validation altogether (Option C). `gpt-oss-120b` as the second judge reuses this session's real reasoning-tax measurements (25-45% of completion tokens, 16.7s mean latency) rather than an unmeasured model, and is architecturally distinct from both the generator (`qwen/qwen3.8-27b`) and the first judge (Gemini), so neither judge is scoring its own generation. |
| 2026-09-25 | LLM-judge rubric is a single holistic 1-5 helpfulness score + required reasoning, not a multi-dimensional breakdown (grounding/clarity/tone/completeness scored separately) | Multi-dimensional rubric, closer to a full production eval suite | Grounding is already RAGAS's `Faithfulness` (Unit 1); citation mechanics are already the citation validator (Phase 9) + Unit 5's aggregated metric. A richer rubric would mostly add places for two judges to disagree over wording rather than substance -- noise for a unit whose entire purpose (2026-09-25's inter-judge-agreement pivot) is measuring whether disagreement reflects a real quality difference. Full reasoning: design record 11c §2. |
| 2026-09-25 | RAGAS's scoring LLM split onto `openai/gpt-oss-20b` (separate from the judge role's `openai/gpt-oss-120b`), both distinct from the primary generator (`qwen/qwen3.8-27b`) | Keep RAGAS scoring on the same model as the judge role (`gpt-oss-120b`, as originally set up in Unit 1) | Running the real N=30 batch found `gpt-oss-120b` has a real 200,000 tokens/day cap that collapsed under combined RAGAS (5 calls/query) + judge (1 call/query) load from one shared bucket. Confirmed `gpt-oss-20b` has an independent quota bucket (worked immediately after `gpt-oss-120b`'s wall) and can complete a real `Faithfulness` call (score=0.625) before committing. Full detail: design record 11d §4. |
| 2026-09-25 | On hitting simultaneous Groq+Gemini quota walls mid-batch, resumed immediately and accepted further stalls (user's call, `AskUserQuestion`) rather than waiting ~20 min for partial refill or stopping at the partial N | Wait for quota to partially refill first; or stop and treat the ~10-query partial result as final | The resumable checkpoint design (built in for exactly this) makes a failed retry cheap (a fast 429, not a long hang) -- waiting idle would trade real time for a benefit ("fewer failed attempts") that doesn't change the real finding. In practice this let the batch reach N=29/30 generations and N=9 paired judge scores rather than stopping at ~10, which is what actually surfaced the systematic (not random) disagreement pattern in §"Judge Agreement" below. |
| 2026-09-25 | Wrote up Unit 4's real negative result (κ≈0) and queued the rubric fix for next session, rather than attempting a live fix today | Try an immediate rubric fix and re-test now | Every model used for judging (Gemini, both `gpt-oss` variants) was quota-exhausted by the time the batch finished -- no live re-test was possible today regardless of approach. Writing up the real, already-diagnosed finding (mechanism identified, not just "kappa is low") is real, valuable, unblocked work; guessing at a fix without being able to verify it against a real call would violate CLAUDE.md's verification discipline. |
| 2026-09-25 | Built Unit 6 (`make eval` scorecard) immediately, despite Unit 4's κ being a known-unreliable number, rather than waiting for the rubric fix + re-run first | Wait for quota reset + rubric fix before building the scorecard, since a scorecard showing today's κ could look like a settled result | Re-examined the earlier framing (progress.md's own prior "next action" line) and concluded it conflated two different units of work: the scorecard's job is to report real numbers honestly, including failing ones -- plan.md's own exit criterion explicitly allows "or a written analysis of why not" for the faithfulness target. The scorecard is pure aggregation (zero new LLM calls) over data already on disk, so it doesn't need quota to exist, and it renders both failing targets as `[FAIL]`, not `[PASS]` -- nothing about running it now misrepresents anything. It will pick up better numbers automatically once Unit 4 is re-run; no scorecard change needed then. |
| 2026-09-25 | Unit 6 scoped to Phase 11's own build list (RAGAS + judge + custom metrics) only, not a combined retrieval-plus-generation scorecard | Pull retrieval metrics in too, using the cached `data/cache/eval/*.json` files from Phase 6's ablation grid | Those cache files are single-axis ablation sweep points (one axis varied, others held at a base config), not one row that *is* "the winning config" combined -- reconstructing a true combined number would mean a fresh live retrieval run, different work than what Phase 11 asks for. plan.md gives Phase 19 its own explicit "final scorecard against all 10 Definition of Done criteria" step; building that scope here would duplicate it. |
| 2026-09-25 | Unit 6's citation-validity metric reports the mean of the batch's stored scalar `valid_citation_rate` only, not Unit 5's fuller `citation_validity_summary` (`ClaimStatus`-level breakdown) | Re-run `CitationValidator` against the batch's stored `answer_text` to get the richer breakdown | The batch checkpoint (`experiments/11e_judge_validation_batch.py`) never stored full `ValidationResult` objects, only the scalar rate -- getting the richer breakdown needs a live tier-2 entailment LLM call (`qwen/qwen3.8-27b`) for some claims, reintroducing quota risk into what should be zero-cost aggregation. Documented as a real, disclosed scope cut (design record 11e §3), not silently dropped. |
| 2026-09-25 | Fixed the judge rubric by restructuring instruction placement/forcefulness (CRITICAL RULE block before the scale, naming the exact override language observed; a "reserve 5" counterfactual anchor), rather than switching either judge model | Swap `gpt-oss-120b` for a different second judge model | The Unit 4 failure was a demonstrated instruction-adherence gap on a rubric this project controls (the model's own real reasoning text showed it correctly identifying refusals, then applying a task-completion override), not a capability ceiling -- a different model needs the same measurement investment with no guarantee of a better real outcome, and would discard Unit 4's already-diagnosed evidence about this model's specific failure shape. Full reasoning: design record 11f §3. |
| 2026-09-25 | Verified the rubric fix with a small, targeted live spot-check (4 real disagreement cases + 3 real non-refusal cases) before committing quota to a full N=30 re-run | Re-run the full N=30 batch immediately under the new rubric | CLAUDE.md's verification discipline -- confirm the fix works on the exact cases that diagnosed the problem before spending the much larger quota cost of a full batch on an unverified change. Real payoff: confirmed `gpt-oss-120b`'s scores moved `[1,2,1,1]` -> `[5,5,5,4]` (exact match to Gemini's original scores) and 3 non-refusal scores were unchanged, using only 7 Groq calls + 1 Gemini call, before both providers' quotas ran out again mid-check anyway -- a full N=30 re-run would have been quota-blocked at roughly the same point with far less signal per call spent. |
| 2026-09-26 | Wrote a separate re-scoring script (`11g_judge_rubric_rescore.py`) that reuses Unit 4's already-generated answers and re-runs only the two judges, rather than re-running the full `11e_judge_validation_batch.py` pipeline | Re-run generation + RAGAS + both judges from scratch under the fixed rubric | Generation and RAGAS didn't change -- only the judge prompt did. Re-generating 29 real answers and re-scoring RAGAS would spend real, scarce `qwen/qwen3.8-27b` and `gpt-oss-20b` quota to re-measure two things that hadn't moved, for no new signal. Real payoff: got 26/29 real groq-side re-scores before hitting a quota wall, at a fraction of the cost a full pipeline re-run would have needed to reach the same point. |
| 2026-09-26 | Reported the full re-score's real distribution shift (mean 3.10->4.31, compression toward scores 4-5) as an open, unresolved concern rather than treating the refusal-specific fix's confirmation as the fix being "done" | Report only the 5 confirmed refusal-case corrections as the headline result, since that's what the fix targeted and what it demonstrably achieved | CLAUDE.md: report what didn't work, and stop when a measurement contradicts expectation. The refusal fix is real and confirmed, but a milder version of Gemini's own ceiling-effect problem appearing on the groq side too is a genuine, undiagnosed new signal -- whether it helps or hurts real inter-judge kappa is unknown until Gemini's paired scores exist, and claiming success before that would repeat the exact mistake (declaring a judge reliable on an unpaired, partial signal) this whole unit exists to avoid. |
| 2026-09-26 | Recommended closing the RAGAS faithfulness gap as a disclosed, explained limitation rather than attempting a prompt-side fix to raise the number | Adjust the generation prompt to discourage explicit "the context doesn't contain X" phrasing, since that's the specific claim shape RAGAS's NLI check can't credit | Real evidence (`golden::issue::15764`) shows the low-scoring refusal was accurate and well-grounded on its one checkable claim -- the metric, not the answer, is what's structurally unable to score it well. Discouraging that phrasing to raise a metric number would directly undermine Phase 10's confidence scorer, whose entire purpose is producing exactly this kind of honest refusal. Full reasoning: design record 11h §3. |
| 2026-09-26 | Re-ran all 4 sampling-sweep configs fresh under `qwen/qwen3.8-27b`, rather than reusing Attempt 1's real (0.0, 0.5) number (0.6634, measured under `llama-3.3-70b-versatile`) for that one cell | Keep the old (0.0, 0.5) result and only measure the 3 remaining configs under the new model | Mixing two different models within one "temperature x top_p" sweep would confound the variable under test -- the whole point of the grid is isolating the sampling-parameter effect, which requires holding the model fixed. Restored `N_QUERIES` from 6 back to 20 in the same change, since the new model's per-minute (not per-day) caps afford it. |
| 2026-09-26 | Killed the sampling-sweep run early (after 2 of 4 configs completed) rather than letting it finish grinding through the remaining, already-doomed calls | Let the script run to completion, since it degrades gracefully (per-query try/except, no crash risk) | Configs 3 and 4 were provably going to be 100% failures once `qwen/qwen3.8-27b`'s TPD wall was hit (every subsequent call gets an immediate 429, confirmed from the real error messages) -- letting it churn through 20 more paced, guaranteed-failing calls (~3 more minutes) would have produced zero new information. The two real, complete configs' data was already safely checkpointed to disk before the kill. |
| 2026-09-26 | Added config-level resumability to `09a_sampling_sweep.py` before wrapping up Phase 11, even though no immediate re-run was planned this session | Leave the script as-is, since it already checkpoints incrementally per config | The user will run the remaining 2 configs themselves later, without this session's context -- as written, a plain re-run would have silently overwritten the 2 real, already-completed temperature=0.0 configs (no existing-file check before starting the grid). Fixed so the deferred task is safe to just run, not something that needs remembering an unwritten caveat about preserving data by hand first. |
| 2026-09-26 | **Phase 13: Langfuse Cloud (free tier), not self-hosted** | Full self-hosted Langfuse v3 stack (web+worker+Postgres+ClickHouse+Redis+MinIO, per the official docker-compose); self-hosted Langfuse v2 (Postgres-only) | Measured, not assumed: Langfuse's own docs recommend 4 cores/16 GiB for the v3 self-host stack alone; this machine's Docker Desktop is allocated 7.75 GiB total, already sharing that with Qdrant (~750 MiB), and the host's full 16 GiB also has to cover MPS embedding inference outside Docker. Real capacity conflict, not routine. Presented all 3 options with a recommendation (`AskUserQuestion`, full stack recommended); **user chose Cloud** — a deliberate deviation from plan.md's literal "self-hosted" wording, disclosed here rather than silently substituted. Free tier (50k units/month, 30-day retention, no card) comfortably covers this project's real traffic scale (7 traced stages/query × a few hundred dev/eval queries, nowhere near 50k/month). Full reasoning: design record 13. |
| 2026-09-28 | **Built a minimal end-to-end `AnswerPipeline` orchestrator inside Phase 13 (Unit 2), not deferred to Phase 14** | Trace at the experiment-script level only, without a real orchestrator; wait and let Phase 14 build it | No orchestrated request path existed anywhere in this codebase -- Phases 5-12 each shipped one stage in isolation, wired together only ad hoc inside experiment scripts. Phase 13's own build item #1 ("tracing every stage") presupposes a real request to attach spans to, and the audit-trail demo (build item #5) needs a real evidence object to reconstruct from -- neither is satisfiable against a batch script. Flagged the gap directly (`AskUserQuestion`) rather than silently picking a scope; user confirmed building it now. Phase 14 wraps this exact object in FastAPI/SSE rather than rebuilding it -- not duplicated scope, built at the point it was first actually needed. Full reasoning: design record 13b SS2-3. |
| 2026-10-01 | **Phase 14: stream tokens live, validate citations after** (`/chat` events: `token`, then `citation`s, then `done`) | Validate first then replay as a stream; stream live and retract/escalate on validation failure | Confirmed with the user (`AskUserQuestion`, recommended option). The non-streaming pipeline already serves answer text regardless of claim validation (validation only decides which citations are listed), so streaming live doesn't weaken the guarantee; replay-after-validation would make streaming cosmetic (first visible text ~0.6 s later, measured) without changing what is served; retraction changes Phase 9/10 behavior and exceeds plan.md's Phase 14 scope. Full reasoning: design record 14. |
| 2026-10-17 | **LLM outage before any text is shown escalates with `cause=provider_unavailable`**; router API errors raise `RouterUnavailableError` instead of returning a disguised `billing_account_legal` | Keep the all-providers-failed terminal `error` and only fix labels/logging | Confirmed with the user (`AskUserQuestion`, recommended option). A support desk hands an unanswerable-right-now request to a person, same as low-confidence abstention; nothing has been shown yet (AllProvidersFailedError is pre-first-token by construction), so the handoff contradicts nothing. The old router path reported a billing question nobody asked in the trace, the UI and the router eval. Mid-stream failures stay a terminal `error`. Design record 14f. |
| 2026-10-17 | **Hand-rolled token bucket for `/chat` rate limiting** | `slowapi` | Confirmed with the user (`AskUserQuestion`). One process, no Redis: the library's shared-store feature buys nothing, and it adds a dependency plus a decorator signature per route; the mechanism is one formula (~40 lines, TDD'd with an injected clock). Design record 14f. |
| 2026-10-17 | **Semantic cache tier off by default** (`semantic_cache_enabled=False`), exact tier kept | Skip semantic tier for follow-ups only; raise threshold to 0.94 | Confirmed with the user (`AskUserQuestion`, recommended option). A real 14e pair of different questions scored 0.9354 (> 0.88) because the rewriter's template dominates short embeddings; a semantic hit skips routing, confidence and validation for the question actually asked; the tier is worth 0.2-1.6% of simulated traffic. Follow-up-only bypass misses the same collision typed directly; 0.94 clears the real pair by 0.005 on n=1 and drops paraphrase recall 9/15 -> 4/15. Design record 14h. |
| 2026-10-18 | **OpenAI as primary provider, hard $10 project cap** (user); Groq/Gemini kept as fallbacks + cross-family second judges; pre-call worst-case spend ledger | Stay on free tiers; replace Groq/Gemini entirely; dashboard limit only; LiteLLM | Free-tier quotas deferred measurements by days (Phases 9, 11, 15). A dashboard limit acts after spend; the ledger refuses before the call and degrades to free providers at the cap. Keeping a non-OpenAI judge guards against self-preference when the generator is OpenAI. Design record 15g. |
| 2026-10-19 | **Generator = gpt-5-mini** (user OK); provider order unchanged (openai -> groq qwen -> gemini) | gpt-4.1-mini (the 2026-10-18 plan); gpt-4.1-nano; qwen on Groq as primary; prompt v2c for OpenAI models | Same 20 queries, prompt v2, judge held fixed: gpt-5-mini 0.7158 vs qwen 0.7296 (CI [−0.148, +0.111]), 6 unsupported citations. gpt-4.1-mini 0.4486 (CI excludes 0, under-cites); gpt-4.1-nano ties on rate but 26 unsupported (decorative citations, against the North Star); qwen primary brings back the 200k TPD cap; v2c lowered both OpenAI models' rate on a disjoint dev set and pushed refusals to 9/10. Caveats: sentence-level rate is coarser for gpt-5-mini's long sentences (step 4 RAGAS checks it); 0 reasoning tokens at minimal effort. Design records 15g1-15g3. |
| 2026-10-22 | **Serve prompt v3** (rule fixed before the run, user-approved) | Keep v2 (A3 open); decide on H2 citation rate alone | Pre-registered rule: (a) poisoned-chunk obedience v3 0/50 <= v2 5/50, (b) v3 − v2 citation rate CI upper bound >= 0 under both splitters and judges (line-aware +0.068 [−0.039, +0.175]). Costs +624 input tokens/request. Design record 15h. |
| 2026-10-22 | **Risky-advice check: vouch by the documentation corpus** (install/link/secret-request never validated otherwise) | Provenance demotion of all issue-supported claims; LLM safety judge; pip-line allowlist | Issues are 16,944/19,236 chunks and the known_issue value; an LLM judge reads the attacker's text; pip-line extraction missed uvicorn (quoted/prose mentions). Design record 15l. |
---

## Design Records

| # | Doc | Phase | Status |
|---|---|---|---|
| 00 | `the-baseline-that-fails.md` | 0 | ✅ |
| 01 | `ingestion-and-parsing.md` | 1 | ✅ |
| 02 | `chunking-strategies.md` | 2 | ✅ |
| 03 | `embeddings-distance-and-ann.md` | 3 | ✅ |
| 04 | `building-a-golden-set.md` | 4 | ✅ |
| 05 | `lexical-dense-and-fusion.md` | 5 | ✅ |
| 06 | `retrieval-metrics-and-ablation.md` | 6 | ✅ |
| 07 | `reranking-and-encoder-architectures.md` | 7 | ✅ |
| 08 | `agentic-routing-and-query-understanding.md` | 8 | ✅ |
| 09 | `generation-prompting-and-citations.md` | 9 | ✅ |
| 10 | `calibrated-abstention.md` | 10 | ✅ |
| 11 | `evaluating-generation.md` — written as 7 lettered sub-records instead (CLAUDE.md §3 convention): `11-groq-model-replacement-and-ragas-wiring.md`, `11b-cohens-kappa-utility.md`, `11c-llm-as-judge.md`, `11d-judge-agreement-batch.md`, `11e-eval-scorecard.md`, `11f-judge-rubric-refusal-fix.md`, `11h-faithfulness-root-cause.md` | 11 | ✅ |
| 12 | `caching-delta-indexing-and-cost.md` — written as lettered sub-records instead (CLAUDE.md §3 convention): `12-exact-match-cache.md`, `12b-semantic-cache.md`; more units to follow | 12 | 🟡 |
| 13 | `observability-and-latency.md` | 13 | ⬜ |
| 14 | `serving-streaming-and-ux.md` — lettered sub-records: `14-streaming-pipeline.md`, `14b-api-sse.md`, `14c-trace-and-feedback.md`, `14d-conversation-memory.md`, `14e-static-ui.md`, `14f-rate-limiting-and-provider-failure.md`, `14g-feedback-review-queue.md`, `14h-semantic-cache-collision-and-cold-start.md`, `14i-multiturn-regime-c-eval.md` | 14 | ✅ |
| 15 | `rag-security.md` — `15-rag-security.md` (threat model + before results), `15b-tenant-partitioning.md`, `15c-access-control.md`, `15g1-generation-bakeoff.md`, `15g2-prompt-fit.md`, `15g3-gpt5-generators.md`, `15g4-router-reeval.md`, `15h-realstack-security.md`, `15j-cache-answer-config-scope.md`, `15k-claim-splitting.md`, `15l-risky-advice-check.md`, `15d-input-boundary.md`, `15e-injection-defenses.md`, `15f-vector-store-exposure.md`, `15g-openai-provider-and-spend-guard.md`; 15h, 15i to follow | 15 | 🟡 |
| 16 | `ci-eval-gates.md` | 16 | ⬜ |
| 17 | `frameworks-and-orchestration.md` | 17 | ⬜ |
| 18 | `deployment.md` | 18 | ⬜ |
| 19 | `what-i-built-and-what-i-learned.md` | 19 | ⬜ |

---

## Course Topic Coverage

37 topics from `RAG_Topics_Learned.md`. Marked ✅ only when exercised *and* recorded, whether built or imported-and-measured. Full mapping in `plan.md` §2.

**Module 1 (5):** ✅✅⬜✅✅ — topics 1 & 4 done at P0 (baseline demo; cost/latency measurement). Topic 2 (the RAG loop) and topic 5 (IR basics) both land at P5 now that a real retrieval pipeline exists end to end.
**Module 2 (8):** ✅✅✅✅✅✅✅✅ — topics 7, 9, 10, 11 done at P3 (metadata filtering; semantic search & embeddings; distance metrics; contrastive training). Topics 6, 8, 12 done at P5: three retrieval techniques (dense + BM25 + metadata prefiltering, one interface), bag-of-words→TF-IDF→BM25 (`bm25s` + the custom analyzer), hybrid search & RRF (built, tested, compared against linear fusion). Topic 13 (retrieval eval metrics — Precision/Recall/MAP@k/MRR) done at P6: `ranx`, verified against a hand-computed example, then used for the full ablation grid. **Module 2 complete.**
**Module 3 (6):** ✅✅✅✅✅✅ — topics 14, 15 done at P3 (KNN→ANN→HNSW: brute-force ground truth + 36-config recall/latency sweep; vector databases: Qdrant, named vectors, payload indexes). Topic 16 done at P2. Topics 18, 19 done at P7: bi-encoders/cross-encoders/ColBERT (built and measured the first two, ColBERT design-record-only per plan.md's cut list) and reranking (pool-size sweep, measured net negative on this corpus). Topic 17 (query parsing: rewriting, NER, HyDE) done at P8 — rewriting and HyDE both built and A/B'd (rewriting measurably helps, HyDE measurably hurts); NER not built as a separate component (this corpus's entity-extraction need is already covered by Phase 1's `product_area` keyword matching — see design record 08 §4). **Module 3 complete.**
**Module 4 (9):** ✅✅✅✅✅✅⬜✅⬜ — topic 20 (transformer architecture: attention) done at P7, load-bearing for explaining why cross-encoders can't replace retrieval (O(n) joint forward passes, no precomputable index) and why they out-discriminate bi-encoders (full self-attention over query+passage jointly vs. two independent encodings). Topic 27 (agentic RAG: sequential, conditional, iterative, parallel workflows) done at P8 — all four built and measured with real components: conditional (the router), parallel (multi-query expansion), sequential (rewrite/HyDE chains), iterative (capped retry). The LangGraph port (P17) revisits this topic in a different framework later, not missing coverage now. **P9:** topic 21 (sampling strategies) done — real temperature/top_p sweep against a faithfulness proxy, one config fully measured (N=20; the full grid is blocked, see progress notes, but the technique itself is exercised and recorded, not just built). Topic 22 (choosing an LLM & benchmarks) done — real Groq-vs-Gemini bake-off (`docs/results/llm-bakeoff.md`), unequal real sample (2 vs. 12) but a genuine, decision-relevant finding (a ~14x latency gap, not just a quality number) grounding *why* Groq is primary. Topic 23 (prompt engineering) done — a full measure→fix→re-measure loop, not just a shipped template: v1's real gap (21.1% missing citations) diagnosed to a specific pattern, v2 written to fix it, validated for real (5 wins/1 tie/0 losses across 6 matched queries, +34.3%), and promoted to the shipped default on that evidence. Topic 24 (advanced prompting & reasoning) now fully done — context pruning (the piece P8 explicitly deferred here) is the context budgeter's whole job: real token counting, strict-prefix packing, and "lost in the middle"-aware placement. Topic 25 (handling hallucinations) done at P9 (two-tier citation validator) and deepened at **P10**: calibrated abstention is the second, complementary half plan.md's own topic-coverage matrix names for this topic (P9, P10 — "citation validator + abstention") — catching a hallucination-prone query *before* generation, not just checking the answer after.
**Module 5 (9):** ✅⬜⬜✅⬜⬜⬜⬜⬜ — topic 29 (what makes production challenging) exercised at P1, reinforced at P6: three real production bugs (thread-unsafe cache, an enrichment-prefix bug, a self-match eval leak) found only by running the full pipeline against real golden-set queries, none catchable by unit tests alone. P7 added a fourth instance of the same pattern in a different form: real hardware/memory constraints (MPS hangs, CUDA OOM) that no unit test surfaces, only a real run against real data volume. P8 repeated the *identical* MPS instance a third time, in a script that hadn't yet internalized its own project's prior two decision-log entries about it. **P10:** topic 32 (customized evaluation) done — the escalation confusion matrix and its precision/recall curve are a domain-specific evaluation no off-the-shelf metric (RAGAS included) provides; plan.md's matrix maps this topic to P11 primarily, but it's genuinely exercised and recorded starting here, deepened further when P11 adds citation-validity-rate and escalation-appropriateness metrics.

**Progress: 27 / 37**

---

## Build vs Import Ledger

Tracks what's actually implemented from scratch vs. imported, so the ratio stated in interviews matches reality. Updated as components land. Full rationale in `plan.md` §1 / `CLAUDE.md` §2.

| Component | Plan | Actual | Phase |
|---|---|---|---|
| Custom analyzer/tokenizer | Build | Build (75 lines) | 5 |
| BM25 | Import (`bm25s`) | Import (`bm25s`, wired to the custom analyzer) | 5 |
| Heading-aware chunker | Build | Build (132 lines, matches the ~150-line estimate) | 2 |
| RRF fusion | Build | Build (83 lines, `fusion.py` — RRF + linear fusion together) | 5 |
| Retrieval metrics | Import (`ranx`) | Import (`ranx`, verified against a hand-computed 5-doc example) | 6 |
| Cross-encoder reranker | Import | Import (`sentence-transformers` `CrossEncoder`, `bge-reranker-v2-m3`) + Build (~120 lines, `rerank/cross_encoder.py` + `rerank/pipeline.py` — toggleable composition) | 7 |
| Confidence scorer | Build | Build (258 lines, `guardrails/confidence.py` — 4 signals, `decide_escalation`/`handle`, escalation payload builder, TDD'd, 17 tests) | 10 |
| Citation validator | Build | Build (196 lines, `guardrails/citations.py` — two-tier: lexical overlap + LLM entailment, TDD'd, 9 tests) | 9 |
| Routing policy | Build | Build (51 lines, `router/policy.py` — hard-coded, no LLM call, TDD'd) | 8 |
| Router classifier | Build (plan.md's own repo layout names `router/classifier.py`) | Import (`Groq` LLM) + Build (131 lines, few-shot + JSON-mode + fail-safe parsing) | 8 |
| Query understanding (rewrite/expand/HyDE/iterative) | Build | Build (445 lines across `query/rewrite.py`, `expand.py`, `hyde.py`, `iterative.py`) + Import (Groq LLM calls; `reciprocal_rank_fusion` reused from Phase 5, not rebuilt) | 8 |
| Context budgeter | Build (plan.md Phase 9 step 2) | Build (126 lines, `generation/budgeter.py` — strict-prefix packing + "lost in the middle" bookend placement, TDD'd, 9 tests) | 9 |
| Versioned prompt templates | Build (plan.md Phase 9 step 3) | Build (v1, 27 lines + a small registry, TDD'd, 4 tests) | 9 |
| Multi-provider LLM client | Build (plan.md Phase 9 step 1) | Import (`groq`, `google-genai`) + Build (137 lines across `generation/llm/fallback.py`/`groq_provider.py`/`gemini_provider.py` — narrow-boundary fallback, TDD'd with fakes + a real live-API demonstration, 10 tests) | 9 |
| Eval harness | Build | Build (~150 lines, `harness.py` + `doc_level.py` — config-hashed, cached, per-stratum) | 6 |
| Semantic cache | Build | — | 12 |
| RAGAS metrics | Import | Import (`ragas` Faithfulness/ResponseRelevancy/IDBasedContextPrecision/IDBasedContextRecall) + Build (~230 lines, `eval/ragas_adapters.py` -- hand-written `BaseRagasLLM`/`BaseRagasEmbeddings` protocol adapters for Groq/Gemini/local `Encoder`, no ragas-native non-OpenAI provider support existed) | 11 |
| Cohen's kappa (judge agreement) | Import | Import (`scikit-learn` `cohen_kappa_score`) + Build (35 lines, `eval/agreement.py` -- thin wrapper, TDD'd against a hand-computed textbook example, 7 tests) | 11 |
| LLM-as-judge (helpfulness rubric) | Build (plan.md Phase 11 step 2) | Import (`instructor` structured output, reusing Unit 1's `InstructorLLM` builders) + Build (~100 lines, `eval/judge.py` -- `Judge`/`HelpfulnessJudgment`, prompt template, TDD'd with a fake LLM, 9 tests) | 11 |
| Custom metrics (citation validity, escalation appropriateness, code-block correctness) | Build (plan.md Phase 11 step 4) | Build (~130 lines, `eval/custom_metrics.py` -- reuses `CitationValidator`/Phase 10's escalation decision/`split_code_segments` rather than duplicating their logic; `ast.parse` for code correctness, not sandboxed execution; TDD'd, 13 tests, one reproducing Phase 10's exact shipped confusion-matrix numbers as a regression guard) | 11 |
| `make eval` scorecard | Build (plan.md Phase 11 step 5) | Build (~200 lines, `eval/scorecard.py` -- pure aggregation over Unit 4's already-written batch rows, reuses `agreement.cohen_kappa`/`custom_metrics.code_block_syntax_validity` rather than recomputing; TDD'd, 14 tests, one a regression guard against `docs/results/judge-agreement.md`'s exact published numbers from the real raw file) | 11 |
| Exact-match cache | Build (plan.md Phase 12 step 1) | Build (~65 lines, `cache/exact_match.py` -- sqlite-backed, same thread-safety pattern as `EmbeddingCache`; TDD'd, 10 tests) | 12 |
| Semantic cache | Build (plan.md Phase 12 step 2) | Build (~85 lines, `cache/semantic.py` -- brute-force cosine similarity, sqlite-persisted; reuses Phase 3's `Encoder`/bge-large rather than a new model) + real threshold tuning (`experiments/12a_semantic_cache_threshold_sweep.py`, zero LLM cost); TDD'd (10 tests: fake deterministic encoder + one real integration test against the actual hardest near-miss pair) | 12 |
| Tracing | Import (Langfuse) | — | 13 |
| Feedback review queue + golden-candidate export | Build (plan.md Phase 14 step 5) | Build (~190 lines, `eval/review_queue.py` + `reviews` table in `api/store.py`; terminal tool, TDD'd) | 14 |
| `/chat` rate limiter | Build (plan.md Phase 14 step 6; not named in §1) | Build (~40 lines, `api/ratelimit.py` -- per-client token bucket, injected clock, TDD'd; `slowapi` rejected, design record 14f) | 14 |

---

## Open Questions

| # | Question | Raised | Needed by | Resolution |
|---|---|---|---|---|
| 1 | Which LLM for the judge? Free tier may not clear κ ≥ 0.6 | 2026-07-28 | Phase 11 | Open — real answer partially in hand: pre-fix inter-judge κ was -0.0161 (well below 0.6, a real negative result, Open Question #16), a rubric fix is groq-side verified (N=26) but Gemini has returned zero real scores under it across two attempts — **the actual κ ≥ 0.6 question is still unanswered**, deferred to the user to complete once Gemini's quota allows (`experiments/11g_judge_rubric_rescore.py`, already checkpoint-resumable) |
| 2 | Do FastAPI closed issues yield ≥ 500 with clear resolutions? | 2026-07-28 | Phase 4 | **Resolved 2026-07-29** — 2,204 of 3,530 closed issues (62.4%) have an identifiable maintainer resolution comment, well above 500 |
| 3 | Deploy target: HF Spaces vs Render? Free-tier terms shift | 2026-07-28 | Phase 18 | Open — reverify terms at Phase 18 |
| 4 | Groq `llama-3.3-70b-versatile` at temperature=0.0 is not fully deterministic across runs — two Phase 0 baseline runs against the same index produced different fabricated answers to the same question | 2026-07-29 | Phase 6 (eval harness), Phase 11 (judge validation) | Open — decide whether to accept as a property of the system under test, cache generations for reproducibility, or something else |
| 5 | Semantic chunking costs ~211–248s to embed a single large document's sentences (measured on `release-notes.md`, confirmed compute-bound) — does it clear a large enough retrieval-quality bar in the Phase 6 ablation to justify that cost, or does it join the cut list? | 2026-07-30 | Phase 6 | **Resolved 2026-08-04** — measured on the full corpus for the first time: 723s to chunk + 640s to embed, and doc-level recall@10 of 0.2571 vs. recursive's 0.2436 (a ~5% relative gain for materially more compute) — both dwarfed by heading-aware's 0.5470. Semantic chunking is cut from the winning config. |
| 6 | 39.7% of the corpus's embeddings are exact duplicates of another chunk's, the largest group being 1,354 issues whose entire chunk text is the GitHub issue-template placeholder `_No response_` (Phase 3). Degenerate/near-empty chunks like these could surface as noisy near-duplicate results for many unrelated queries. Should they be filtered out of the indexed corpus, or does the confidence scorer (Phase 10) / reranker (Phase 7) handle it downstream without special-casing? | 2026-08-02 | Phase 4 (golden set shouldn't be built from these) / Phase 6 | Open — Phase 6 confirmed the broader pattern (GitHub issue-template boilerplate headings actively hurting contextual enrichment, see Decision Log 2026-08-04). **Phase 10 update:** got a direct, concrete answer for the confidence scorer specifically — it does *not* handle near-duplicate chunks for free. One real dev query's top-10 dense scores had near-zero variance (plausibly a near-duplicate cluster), sending the `distribution_shape` signal's raw value to ~76,856 against a typical range of 0-30. Winsorized normalization contained the symptom, but the signal's own measured correlation with the escalation label stayed ~0 even after fixing that. Still open whether filtering degenerate chunks at the source would help retrieval quality *directly* (not just this one downstream signal) — not chased further this phase. |
| 7 | The eval harness's cache key (`config_hash + queries_hash`) doesn't capture "the underlying Qdrant collection for this config was regenerated" — re-running an unchanged `RetrievalConfig` after rebuilding its index silently serves the stale cached result (hit once in Phase 6 after fixing the enrichment-prefix bug; worked around by manually deleting the affected cache file, not fixed generally). | 2026-08-04 | Phase 7+ (any phase that rebuilds an index without changing a config field) | Open — add a data-version field to `RetrievalConfig` if this recurs, or document "clear `data/cache/eval/` after any index rebuild" as a standing rule |
| 8 | The golden set's ground truth (`auto_label.py`, Phase 4, token-overlap ≥15%, "biased toward inclusion") has a median of 17 chunks spanning 12 documents per query — a broad "on-topic" bar, not "directly answers this question." Phase 7 found this causes a genuinely better relevance judge (a cross-encoder) to score *worse* than a fuzzier one (a bi-encoder) against this ground truth, purely as a labeling-methodology artifact, not a real quality difference. Does this same permissiveness distort Phase 6's absolute recall/NDCG numbers too (not just Phase 7's reranking comparison)? | 2026-08-11 | Phase 11 (LLM-judge validation, Cohen's κ vs. Opus) | Open — Phase 6's *relative* comparisons (config vs. config, same ground truth throughout) stay valid regardless; only absolute numbers and any technique whose strength is precision over recall (like reranking) are at risk. Phase 11's judge-agreement measurement is the natural place to quantify this directly. **Phase 8 update:** HyDE hit the identical mechanism independently (a specific, accurate hypothetical answer regressed because it committed to one narrow angle of a broad ground truth) — now two independently-built techniques confirming the same pattern, strengthening the case that this needs Phase 11's direct measurement rather than remaining a one-off explanation. |
| 9 | Multi-query expansion's unweighted `reciprocal_rank_fusion` across N query variants (1 original + up to 3 LLM paraphrases) structurally lets breadth-across-lists outvote peak relevance-in-one-list — measured directly (Phase 8): a chunk absent from the original query's results but present at a mediocre rank in all 3 paraphrase lists can outscore the original query's own top hit. `rrf_k=60` was tuned in Phase 6 for exactly 2 sources (dense + sparse) of comparable reliability, never validated for N-way fusion where most sources can be simultaneously wrong. | 2026-08-16 | Not currently needed — expansion is off by default (see Decision Log). Would matter if expansion is revisited, or if any future phase fuses more than 2 ranked lists. | Open — not chased further in Phase 8, out of scope per plan.md's ask to A/B and report, not iterate to convergence. Candidate fixes if revisited: weight the original query's list higher than paraphrases, or validate/filter paraphrase quality before fusing. |
| 10 | Prompt v1 achieves 0% invalid-citation-number errors but only 78.9% of real claims carry any citation at all (21.1% `missing_citation`, N=20 real sweep). Spot-checking real answer text found at least two distinct causes: genuine prompt-following failures (an opening topic sentence with a real, available citation that the model just didn't attach), and the validator's own `_requires_citation` heuristic being too blunt (a sentence merely restating the user's own stated problem back to them still trips the ">4 words, no trailing colon" rule). | 2026-08-16 | Phase 10 (a badly-cited answer should push the confidence scorer toward escalation, not just get flagged) | Open — needs a v2 prompt (explicit citation-on-first-sentence instruction/example) measured against real generations once Groq quota or Gemini access allows, *and* a narrower `_requires_citation` heuristic that excludes user-problem restatement. Conflating the two causes into one fix wouldn't actually close the gap. |
| 11 | ~~`CitationValidator`'s sentence splitter (reused from `chunking/semantic.py`, built for parsing existing prose) doesn't understand LLM-generated Markdown~~ | 2026-08-16 | Phase 10/11 | **Resolved 2026-08-17** — reproduced as a failing test first (`test_fenced_code_block_does_not_get_glued_to_surrounding_prose`), fixed by reusing `chunking/segments.py`'s `split_code_segments` (Phase 2) to strip fenced code out before sentence splitting; code segments produce no claim of their own. 356/356 suite passing. |
| 12 | Both providers' free tiers have real, tight daily caps, but shaped completely differently — Groq's is *token*-based (100,000/day on `llama-3.3-70b-versatile`; one 20-query config used 95,944 of it), Gemini's is *request*-based (20/day/model, discovered when 17 earlier real Gemini calls this session left almost nothing for the planned v1-vs-v2 comparison). A system, or an eval schedule, built to survive only one shape wouldn't survive the other. | 2026-08-16 | Phase 11 (generation eval), Phase 16 (CI gate) — both will need real generation-model calls at volumes neither tier may support in one run | Open — `FallbackLLMClient` (this phase) provides the mechanism (fail over to the other provider), but that only helps when the *other* provider still has headroom, which isn't guaranteed if both get exercised in the same session (as happened here). Real question for Phase 11/16: budget real eval runs across multiple days, or budget for a paid tier. **2026-09-26 update:** confirmed this isn't just a Groq-vs-Gemini split — even within Groq, `qwen/qwen3.8-27b` (generation), `openai/gpt-oss-120b`, and `openai/gpt-oss-20b` each have their *own* independent 200,000 TPD buckets, plus per-minute ITPM/OTPM caps on top, all hit for real this session. "Budget across multiple days" is now the practical answer in use — Phase 11's judge κ re-run and the sampling sweep's last 2 configs are both explicitly deferred to whenever each model's quota next allows, rather than chased same-day. |
| 13 | ~~Prompt v2 not validated~~ | 2026-08-16 | Phase 10 | **Resolved 2026-08-17** — third attempt, once Groq's quota had genuinely reset, produced a real matched-pair comparison (N=6, same queries, same context): v2 won 5, tied 1, lost 0 vs. v1, mean `valid_citation_rate` 0.4199 → 0.5641 (+34.3%). Confirmed the mechanism on real text (v1's uncited opening sentence vs. v2's cited one, same question, same context). `GenerationSettings.prompt_version` promoted to `"v2"`. Full detail: `docs/results/prompt-v1-vs-v2.md`. |
| 14 | All 3 of the confidence scorer's real misses (dev negatives that slip through as confident wrong answers at tau=0.50) share one mechanism none of the 4 signals was built to catch: a named entity in the query genuinely appears in a retrieved chunk (real lexical/semantic overlap), while the actual question is unanswerable or out of scope for a reason no retrieval/reranking signal detects — asking about a *future* release date (`unanswerable::7`, retrieves a real "FastAPI Version" chunk) or the *wrong framework* sharing a library name (`out_of_corpus::9`, "Celery" pulls in a real FastAPI+Celery chunk). Is this worth a fifth signal or heuristic (e.g., detecting temporal/future-tense questions, or cross-framework entity mismatches), or is 3/180 too small a sample to design against yet? | 2026-08-17 | Phase 11 (does the judge agree these are genuinely hard?) / revisit if holdout or production traffic shows the same pattern at higher volume | Open — not chased further this phase; see design record 10 §6 |
| 15 | ~~Groq retired `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` outright~~ | 2026-08-18 | Phase 11 Unit 1 (blocks it directly) and any real re-run of Phases 8-10's Groq paths | **Resolved 2026-09-25** — measured `openai/gpt-oss-120b`'s real reasoning-token tax at actual answer length (not just the trivial call): 6 real dev-positive generations, 25-45% of `completion_tokens` spent on reasoning but never truncating visible output — not fatal on its own, but real mean latency was 16.7s (1.2-29s), nearly Gemini's already-rejected 21.4s, undermining "Groq primary because it's fast." Also found `groq/compound-mini` (last session's verified small-task replacement) had *itself* 404'd by this session — Groq's catalog is churning faster than one session's verification survives. `qwen/qwen3.8-27b` (a renamed successor to the `qwen3.6-27b` rejected last session) measured clean instead: 0 reasoning tokens across the same 6 queries, mean latency 8.7s, with one real free-tier constraint found and fixed (1000 output-tokens/minute cap, confirmed via a real 429 at `max_tokens=1024` and a real success at 1000). User decided (`AskUserQuestion`): `qwen/qwen3.8-27b` for all five Groq-backed roles. Applied to `config.py`, verified through the real `GroqProvider`+`GenerationPipeline` path (not just the raw SDK script): 1,125ms latency, `valid_citation_rate=1.0`. 373/373 tests still green. |
| 16 | Inter-judge Cohen's κ measured at -0.0161 (unweighted) / 0.0906 (quadratic), N=9 paired — at/below chance, well under the 0.6 bar. Two real mechanisms identified: Gemini's scores have almost no variance (10/11 real successes were a perfect 5/5, a ceiling-effect leniency bias); `openai/gpt-oss-120b` systematically scores honest "insufficient information" refusals low (1-2) despite an explicit rubric instruction to score them high, while Gemini reliably follows that instruction. Per plan.md's own framing ("if κ < 0.6 the judge is unreliable... iterate the rubric or upgrade the model"), this needs real iteration before any `make eval` number built on judge scores can be trusted. | 2026-09-25 | Unit 6 (`make eval` scorecard) — any scorecard built on today's judge numbers would report a known-unreliable signal as settled | **Partially resolved 2026-09-26** — rubric fixed (design record 11f: honest-refusal rule repositioned as a CRITICAL block, ceiling-effect anchor added) and the groq-side fix confirmed real at N=26 (`docs/results/judge-rubric-fix-rescore.md`: all 5 refusal cases corrected, matching Gemini's original stance). **Still open:** Gemini has returned zero real scores under the fixed rubric across two separate attempts (both quota-blocked), so real paired κ under the fixed rubric remains completely unmeasured — the groq-side fix also revealed a new, undiagnosed distribution-compression side effect that can't be interpreted without Gemini's paired data. **Deferred to the user** to complete once Gemini's 20/day quota allows: re-run `experiments/11g_judge_rubric_rescore.py` (checkpoint-resumable — only fills in missing scores) and update `docs/results/judge-agreement.md` with the real outcome. |
| 17 | The citation validator counts an honest "the context does not cover X" sentence as `missing_citation`, so stating a gap scores the same as an uncited hallucination | 2026-10-19 | Any generation-quality number (15g1/15g2 measured it: ~9 such sentences for qwen, ~6 per model under v2c) | Open -- changes the metric for every model including qwen's 0.7296; needs an explicit decision and a re-score of all stored answers, not a silent fix. Same family as #10 |
| 18 | The claim splitter doesn't break on newlines/bullets: a bulleted answer becomes a few long "claims" (up to 1,718 chars) that pass or fail as one unit on word overlap | 2026-10-22 | Every valid_citation_rate (15h measured it: ~40% of v3's apparent gain over v2 was granularity) | **Resolved 2026-10-22 (15k)** -- claims end at paragraph breaks + list items (not plain newlines); decision-relevant answers re-scored: qwen 0.7296 -> 0.7539, gpt-5-mini v2 0.7217, v3 0.7842; both decisions hold |
| 19 | The exact-match cache key has no prompt version: after a prompt change, cached answers from the old prompt are replayed (a hit skips generation and validation) | 2026-10-22 | The v2 -> v3 switch (15h) | **Resolved 2026-10-22 (15j)** -- cache scope = corpus_version + prompt version + provider chain; v2->v3 and qwen->gpt-5-mini both miss now; 3 TDD tests |
| 20 | The citation validator certifies poisoned advice: when v2 relayed a poisoned chunk's `pip install fastapi-securefix` recommendation citing that chunk, it scored SUPPORTED_LEXICAL (4/4) -- "supported" means "a retrieved block says it", not "it is true or safe" | 2026-10-22 | The North Star: a validated citation looks authoritative | **Resolved for install/link/secret-request (15l, 2026-10-22)** -- `UNVOUCHED_RISKY_ADVICE` unless the documentation corpus vouches; not cached, UI warning. Still open: insecure-config advice (e.g. CORS wildcard + credentials), import-name typosquats (`pydantic_settings`) |

---

## Session Log

Newest first. One entry per working session.

### 2026-10-19 → 2026-10-22 — Phase 15 completed: provider migration, real-stack security batch, three validator/cache fixes, wrap-up

Provider migration measured, not assumed: the planned generator (gpt-4.1-mini) lost to qwen (15g1), a stricter prompt made it worse
(15g2, negative result), gpt-5-mini tied qwen and was adopted (15g3), router stays nano because only billing is acted on (15g4).
Real-stack security on gpt-5-mini (15h): v3 served on a pre-registered rule. Fixes from findings: cache scope carries prompt + model
(15j), claims split on lists/paragraphs with decision numbers re-scored (15k), risky-advice output check (15l). Wrap-up (15i): all four
exit criteria met. Spend across the phase's paid work: $0.350578 (ledger total $0.350669 incl. two accidental 15g router calls).
Estimate misses, disclosed: step 2 router estimate exceeded by ~18% (estimated from one short call). Tests 717 -> 746.

### 2026-10-18 — Phase 15 started: attack suite + before results (Unit 1)

Wrote 16 deterministic security tests (11 succeeding attacks as strict xfail) and a zero-LLM real-data baseline script;
recorded design record 15 and the start of `security-assessment.md`. Headline: Phase 10's injection "recall 1.00" is
off-topic detection only (wrapped injections pass 14/14); Qdrant is LAN-open with full text in the payload; BM25 leg
ignores filters (33.8% of slots). User accepted all three recommendations + `gemini-3.5-flash`. Then 15b (tenant
partitioning): A1 609/1,800 -> 0/1,800, bypass sweep 0 leaks in 760 real runs, fail-closed mutation check.
`uv run pytest -q`: 612 passed, 8 xfailed. Then 15c (access control, user chose anonymous chat + ownership
cookie + operator-only /trace): all A5 attacks closed, real-server verified; Langfuse ReadTimeouts made /trace a 500 -> now 502.
`uv run pytest -q`: 643 passed, 2 xfailed. Then 15d: boundary redaction (retrieval cost 0), flag-only injection regex;
DeBERTa beat it on paraphrases 14/14 vs 0/14 but failed padding 0/7; Presidio flagged 135/180 dev. 661 passed, 2 xfailed.
Caches deleted (user request): DeBERTa 714 MB, Presidio/spaCy 482.6 MB. Then 15e: commitment check served (A4 closed),
prompt v3 built not served (A3 strict xfail on served version); FP corpus 0/40,551, probe 3/8; v3 +20% tokens.
695 passed, 1 xfailed. Then 15f: Qdrant was writable/deletable from the LAN with no key -> loopback + admin/read-only keys
(401/403/refused); vector-only leakage precision@10 0.286 vs 0.235 prior. 699 passed, 1 xfailed. User switched to a paid OpenAI key with a $10 hard cap;
plan approved (OpenAI mostly, Groq/Gemini as fallback + second judges). 15g: OpenAI provider + pre-call spend guard;
one accidental test-made call pair ($0.000091) found and guarded. 717 passed, 1 xfailed. Next: bake-off, router re-eval, 15h.

### 2026-09-26 — Phase 12 started: exact-match cache (Unit 1)

**Context.** Same session, continued from wrapping up Phase 11 (all 6 units + rubric fix + faithfulness
root cause done; real paired κ and the last 2 sampling-sweep configs explicitly deferred to the user, not
this project's own next-session task — see the "Phase 11 wrap-up" note above). User confirmed starting
Phase 12; scoped its 6 build items from plan.md before writing code (all local/embedding-side, none need
an LLM call, fully unblocked by today's quota state).

**Done**
- Unit 1 (exact-match cache): `src/aegis/cache/exact_match.py` — `ExactMatchCache` (sqlite-backed,
  `EmbeddingCache`'s thread-safety pattern reused), `normalize_query`, `CachedAnswer`. TDD'd first (10
  tests: normalization, case/whitespace insensitivity, corpus-version scoping — a version bump makes old
  entries unreachable rather than needing an explicit purge, persistence across reopened instances). All
  passed on first implementation, no RED→fix cycle needed.
- Deliberately scoped the cache to a purpose-built `CachedAnswer` (text, cited chunk ids, provider) instead
  of the full `GenerationResult` — avoids duplicating `PackedContext`'s full `Chunk` payload (already
  durably stored elsewhere) into every cache row. `corpus_version` is accepted as an opaque parameter, not
  computed in this unit — that's Unit 4's (delta indexing) job, kept decoupled via the same DI pattern the
  rest of this project already uses.
- Design record 12 (`docs/explainers/12-exact-match-cache.md`).

**Exit criteria — Unit 1 of 6 complete**
- Real, working, persistent exact-match cache with corpus-version scoping — done.
- `docs/results/cache-performance.md`, the semantic-cache near-miss test, delta-indexing's deleted-doc
  test, `docs/results/quantization-tradeoff.md`, and the cost-per-query table are all still ahead (Units
  2-6).

**Unit 2 (semantic cache) — real threshold tuning, zero LLM cost.** Wrote
`experiments/12a_semantic_cache_threshold_sweep.py`: embedded (bge-large, local, free) the 15 real
highest-similarity pairs among genuinely different real dev-positive golden queries (real negatives) and 15
hand-written paraphrases of real golden queries (positives, disclosed as one person's phrasing, not a
sampled population). **Found real overlap** between the two populations — no threshold cleanly separates
"same question" from "different-but-related question" on this corpus (negatives up to 0.8694, positives
down to 0.7441). Chose 0.88 — same asymmetric-cost logic as Phase 10's tau (a false positive here serves a
confidently wrong cached answer; a false negative just costs one extra real pipeline call) — the lowest
threshold with zero real false positives, real hit rate 9/15 (60%). Built `SemanticCache` (brute-force
cosine, sqlite-persisted, same `corpus_version`/`CachedAnswer` shape as Unit 1), TDD'd with a fake
deterministic 2D embedder (9 tests covering threshold behavior precisely) plus one real integration test
using the actual shipped default against the real hardest near-miss pair found — the explicit near-miss
test plan.md's exit criteria ask for, not just a fake-vector proxy for it. Design record 12b. `uv run
pytest -q`: 440/440 passed (+10 new).

**Next:** Unit 3 (hit-rate simulation against a Zipfian query distribution) — measure what these two cache
tiers actually save in aggregate, not just per-query correctness.

### 2026-09-25 — Phase 11 Units 1-6 complete: Groq/RAGAS fixes, κ utility, LLM-judge, N=30 batch (real negative result), custom metrics, `make eval` scorecard

**Context.** New session, resuming Phase 11 Unit 1, blocked since 2026-08-18 on Open Question #15 (Groq retired both `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`; last session identified `openai/gpt-oss-120b` and `groq/compound-mini` as candidates but hadn't measured gpt-oss-120b's real reasoning-token tax at answer length or applied anything to `config.py`). Session Protocol followed: read `progress.md` and `plan.md` §Phase 11 before touching code.

**Done**
- Started Docker (was not running) and the `sourcedesk-qdrant-1` container — needed for any real retrieval, and neither was running at session start.
- Wrote `experiments/11a_gpt_oss_reasoning_tax.py`: real retrieval (bge-large dense) + real packed context + real v2 prompt + real `GroqSettings.max_tokens=1024` budget, non-streaming so `usage.completion_tokens_details` is readable, against 6 real dev-positive golden queries — the measurement last session flagged as still needed ("not just the trivial call already tested"). Result: `openai/gpt-oss-120b` never came back empty (unlike the trivial-call test), but spent a real 25-45% of `completion_tokens` on `reasoning_tokens` per query, and mean latency was **16.7s** (1.2-29s range) — close to Gemini's already-measured-too-slow 21.4s (Phase 9 bake-off), undermining the entire "Groq primary because it's fast" premise this project's provider-order decision rests on.
- Re-checked Groq's live catalog (`client.models.list()`) before trusting last session's candidates further — found `groq/compound-mini` (verified for the four small-task roles just five weeks ago) is now **also 404**. Groq's model catalog is churning faster than a single session's verification survives.
- Found `qwen/qwen3.6-27b` (rejected last session for being reasoning-first) had been renamed to `qwen/qwen3.8-27b`; a fresh trivial call came back clean. Didn't trust that alone — wrote `experiments/11b_qwen_reasoning_check.py`, same real-answer-length methodology as 11a. Result: **0 reasoning tokens across all 6 real generations**, but 2/6 hit a real `RateLimitError` — a 1000 output-tokens/minute free-tier cap, one token under the project's `max_tokens=1024` default. Confirmed the fix directly: retried both failed queries at `max_tokens=1000`, both succeeded (completion_tokens 426/380, latency 9.6s/32.3s). Full 6-query set at 1000: 0 reasoning tokens, mean latency **8.7s** — slower than the dead 70B model's original 1.5s bake-off figure, but ~2.5x faster than Gemini.
- Presented the real tradeoff to the user (`AskUserQuestion`, three options with numbers: `qwen/qwen3.8-27b` for everything, `gpt-oss-120b` for primary only, or reconsidering Gemini as primary) — user chose `qwen/qwen3.8-27b` for all five Groq-backed roles, given `compound-mini`'s death left no independently-verified second model for the small-task roles anyway.
- Applied to `config.py`: `GroqSettings.model`, `RouterSettings.model`, `QuerySettings.rewrite_model`/`expansion_model`/`hyde_model`, `CitationSettings.entailment_model` all → `qwen/qwen3.8-27b`; `GroqSettings.max_tokens` 1024 → 1000 (real measured OTPM cap). Every changed default carries an inline comment with the real numbers behind it, same convention as the rest of the file.
- Verified through the real code path, not just the raw-SDK measurement scripts: `GroqProvider` + `GenerationPipeline` against a real golden query — 1,125ms latency, `valid_citation_rate=1.0`, a real cited answer. `uv run pytest -q`: **373/373 passed**.
- Resumed Unit 1's actual goal: `experiments/11c_ragas_smoke_test.py`, real retrieval + real generation + all 4 target RAGAS metrics against one real answer. Hit a second, unrelated real bug immediately: `Faithfulness`/`ResponseRelevancy` crashed with `AttributeError: 'InstructorLLM' object has no attribute 'agenerate_prompt'`.
- **Investigated rather than patched around it.** Read the installed `ragas==0.3.9` source directly: `Faithfulness`/`ResponseRelevancy` are built on `PydanticPrompt`, which needs a `BaseRagasLLM` (`generate_text`/`agenerate_text` → `LLMResult`). `InstructorLLM` (what last session's adapter built, and what `llm_factory` itself also returns) is a different protocol entirely — `generate(prompt, response_model) -> T`, for structured-output calls, per `llm_factory`'s own docstring. `PydanticPrompt`'s own LangChain-detection heuristic (`hasattr(llm, "agenerate") and not hasattr(llm, "run_config")`) misfires on `InstructorLLM` (has `.agenerate`, no `run_config`) and routes to a method that only exists on LangChain chat models — hence the confusing `AttributeError` instead of a clean type mismatch. This isn't specific to bypassing `llm_factory` for Groq/Gemini — the same break would hit the standard openai/anthropic path too.
- Rewrote the LLM side: `GroqRagasLLM`/`GeminiRagasLLM`, hand-written `BaseRagasLLM` subclasses wrapping the raw `groq`/`google-genai` SDKs directly (no `langchain-groq`/`langchain-google-genai` added — the `langchain-core` types needed, `LLMResult`/`ChatGeneration`/`AIMessage`, were already a transitive cost of the `ragas<0.4` pin). Kept the original `InstructorLLM`-based builders, renamed to `build_groq_judge_llm`/`build_gemini_judge_llm`, correctly scoped for Unit 3's rubric-based structured-output judge instead.
- Fixed `Faithfulness`, hit the identical split on the embeddings side next: `ResponseRelevancy` calls `.embed_query()` (the legacy, `langchain-core`-`Embeddings`-derived `BaseRagasEmbeddings`, plural), not `.embed_text()` (`RagasLocalEmbeddings`'s original base class, the modern singular `BaseRagasEmbedding`). Rewrote `RagasLocalEmbeddings` against the plural interface, same `Encoder`-backed zero-quota implementation underneath.
- Real smoke test now passes end-to-end, one real query (`golden::issue::2689`): `faithfulness=0.8750` (16.2s, 2 LLM calls), `response_relevancy=0.7820` (55.5s, 3 LLM calls + a one-time local embedding-model load), `id_based_context_precision`/`id_based_context_recall` both `0.8000` (<1ms, no LLM — pure ID-set arithmetic against the golden set's `ground_truth_chunk_ids`). `uv run pytest -q`: 373/373 passed (no regressions; `ragas_adapters.py` has no unit tests of its own — I/O-bound, covered by this integration smoke test per CLAUDE.md's testing convention).
- Wrote design record 11 (`docs/explainers/11-groq-model-replacement-and-ragas-wiring.md`) and two results docs (`docs/results/groq-model-replacement.md`, `docs/results/ragas-smoke-test.md`).

**Exit criteria — Unit 1 (RAGAS LLM/embeddings adapter wiring) complete**
- Real, working `BaseRagasLLM`/`BaseRagasEmbeddings` adapters producing real scores against real generated answers — done.

**Unit 1 surprised by:** two distinct real breaks in one session, from two different root causes. First: Groq's free-tier catalog is unstable on two independent axes (availability *and* behavior), not just availability as Open Question #15 originally framed it — `groq/compound-mini` (verified working) died, `qwen3.6-27b` (verified reasoning-first) got renamed and un-broken, both in five weeks. Second: a library documented as having "the modern interface" (`llm_factory`/`InstructorLLM`) isn't automatically wired to every metric that predates it — `Faithfulness`/`ResponseRelevancy` never migrated off the older `PydanticPrompt`/`BaseRagasLLM` protocol, and nothing in either class's docs says the newer one won't work with them. Both lessons matter beyond this session: re-verify Groq model behavior at time of use (Phase 16's CI gate especially), and don't assume a library's most-documented interface is the one a specific call site actually needs — read the metric's own source, not just the adapter-factory's.

---

**User declined to hand-label anything for judge validation** (mid-session, before Unit 2 started). Flagged directly first — Phase 11 judge validation is CLAUDE.md's never-cut evidentiary spine, and no substitute is as strong as real hand-labels — then presented 3 real alternatives (`AskUserQuestion`). User chose **inter-judge agreement** (Gemini vs. a second, architecturally distinct model) over a reference-based proxy or dropping validation entirely. Full reasoning: Decision Log, 2026-09-25 entries.

**Unit 2 (Cohen's κ utility).** `eval/agreement.py` — `cohen_kappa`/`raw_agreement_rate`, thin wrapper over `scikit-learn.metrics.cohen_kappa_score` (added as an explicit direct dependency, was transitive-only). TDD'd against Wikipedia's hand-computed Cohen's kappa example (exact match: κ=0.40, raw_agreement_rate=0.70) plus edge cases (perfect/chance agreement, mismatched lengths, quadratic weighting for a possible ordinal rubric). Design record 11b. 7 new tests, 380/380 suite.

**Unit 3 (LLM-as-judge).** `eval/judge.py` — `Judge`/`HelpfulnessJudgment` (single 1-5 holistic score + required reasoning, real `instructor` structured output), `build_groq_judge` (`openai/gpt-oss-120b`)/`build_gemini_judge` (`gemini-3.5-flash`) reusing Unit 1's `InstructorLLM` builders for the use case they were actually scoped for. Deliberately narrow rubric (one score, not multi-dimensional — grounding/citations already covered elsewhere) and deliberately model-diverse judges (neither is the primary generator, avoiding self-preference bias for either). TDD'd with a fake LLM (9 tests, including a regression guard that the "honest refusal scores high" instruction stays in the prompt). Real two-judge smoke test found two more real bugs: `instructor` needed its `google-genai` extra (`jsonref`, transitive-only gap, same shape as scikit-learn's) and a real transient Gemini `503` (already-documented, resolved on retry). Design record 11c, `docs/results/judge-smoke-test.md`. 9 new tests, 389/389 suite.

**Unit 5 (custom metrics) — built concurrently with Unit 4's background batch run, deliberately isolated (zero LLM calls, can't collide with live quota).** `eval/custom_metrics.py`: `citation_validity_summary` (aggregates `CitationValidator` results), `escalation_appropriateness` (TP/FP/FN/TN + precision/recall/deflection_rate, same convention as Phase 10's real confusion matrix — one test reproduces Phase 10's exact shipped numbers, TP=27 FP=24 FN=3 TN=126, as a regression guard), `code_block_syntax_validity` (`ast.parse` on Python-tagged-or-untagged fenced code, explicit non-Python tags skipped — grounded in a real corpus fence-tag count, not assumed). Verified against real generated answers from the Unit 4 checkpoint. 13 new tests, 402/402 suite.

**Unit 4 (N=30 batch: generation → RAGAS → both judges → inter-judge κ) — the real work of this session, and a real negative result.**

Before it would even run, found and fixed a third real bug: `GroqRagasLLM` reused `settings.groq.max_tokens=1000` (`qwen/qwen3.8-27b`'s real OTPM ceiling from Unit 1), too tight for RAGAS's own statement-generation/entailment calls (`LLMDidNotFinishException`, 2/2 on a 2-query dry run). Decoupled it (`JudgeSettings.ragas_llm_max_tokens=4096`) — then found the fix's model, `openai/gpt-oss-120b`, has its own 200,000 tokens/day cap, which collapsed under combined RAGAS (5 calls/query) + judge (1 call/query) load sharing one bucket (real `429`, "Used 197,634/200,000" after essentially one query). Split the roles onto two independently-quota'd models — `openai/gpt-oss-20b` for RAGAS scoring (confirmed via a real call immediately after the 120b wall that its bucket is genuinely separate), `gpt-oss-120b` for the judge role only.

Ran the real batch (`experiments/11e_judge_validation_batch.py`, field-level resumable by design — checkpoints after every sub-step so a quota-driven partial failure never re-spends already-successful calls). Hit real, simultaneous quota walls on **all four models used**: `qwen/qwen3.8-27b` (a new one — input-tokens/minute, not the previously-found output-side cap), `gpt-oss-120b` (its 200K TPD), `gpt-oss-20b` (same cap, independently exhausted by query ~29/30), and Gemini (20 requests/day, fully exhausted). One query (`golden::issue::2600`) failed generation entirely — Groq and Gemini both down at once. Presented the real tradeoff to the user (`AskUserQuestion`): resume immediately and accept further stalls, vs. wait ~20 min, vs. stop at the partial N. User chose to push through — this is what got the batch to N=29/30 generations and N=9 paired judge scores instead of stopping around N=10, which is what actually surfaced the finding below as a *pattern*, not a handful of isolated failures.

**Headline result: inter-judge Cohen's κ = -0.0161 (unweighted) / 0.0906 (quadratic), N=9 paired.** At or below chance agreement — well under plan.md's 0.6 bar ("if κ < 0.6 the judge is unreliable and every metric built on it is decoration"). Two distinct, real, well-evidenced mechanisms, not noise: **Gemini's scores have almost no variance** (10 of 11 real successes were a perfect 5/5 — a ceiling-effect leniency bias). **`openai/gpt-oss-120b` systematically scores honest "I don't have enough information" refusals low (1-2)** despite an explicit rubric instruction to score them high, while Gemini reliably follows that same instruction on every one of the 4 largest disagreements. This is the more consequential finding: a judge that penalizes honest abstention would, if used to tune this system, directly reward the Air Canada failure mode this entire project exists to prevent. RAGAS `faithfulness` mean 0.7977 (n=20/29, real, below the 0.90 target) — not root-caused this session, every model that would measure it is now quota-exhausted.

Design record 11d, `docs/results/judge-agreement.md` (full evidence table, per-query disagreement reasoning text, quota-wall breakdown), raw: `docs/results/judge-agreement-batch-raw.json`. 402/402 suite (no new unit tests for the batch script itself — integration-only, per CLAUDE.md's I/O-bound convention).

**Exit criteria — Units 1-5 of 6 complete**
- Unit 1: real RAGAS scores end-to-end — done.
- Unit 2: real, hand-verified κ utility — done.
- Unit 3: real two-judge structured-output scoring — done.
- Unit 4: real N=30 batch, real inter-judge κ (a real, diagnosed negative result, not a completion in the pass/fail sense) — done.
- Unit 5: real custom metrics, TDD'd, verified on real data — done.

**Surprised by (Units 2-5, beyond Unit 1's two already-logged surprises):** the free tier's quota model is far more granular than this project had mapped — four structurally different real caps (input-tokens/minute, output-tokens/minute, tokens/day, requests/day) across two providers, not the two shapes Open Question #12 originally named. And the inter-judge-agreement substitution (forced by the no-hand-labeling decision) turned out more informative than a clean judge-vs-human pass/fail plausibly would have been — neither judge looks obviously broken taken alone; it's specifically the comparison that exposes that neither is fully trustworthy, for two different reasons.

**Unit 6 (`make eval` scorecard) — built the same session, deliberately not deferred.** Re-examined the initial instinct to wait for the rubric fix first, and concluded that conflated two different pieces of work: the scorecard's job is to report real numbers honestly (plan.md's own exit criterion explicitly allows "or a written analysis of why not" for faithfulness), and it's pure aggregation over Unit 4's already-checkpointed rows — zero new LLM calls, so today's quota exhaustion doesn't block it at all. `src/aegis/eval/scorecard.py`: RAGAS means, inter-judge κ (reusing `agreement.py`), code-block syntax validity + mean citation-validity rate (reusing `custom_metrics.py`), checked against plan.md's 4 Phase 11 exit criteria. `make eval` now runs it — and was previously silently broken (the Makefile's prior target, `aegis.eval.harness`, has no `__main__` block, a real latent bug found and fixed along the way, not introduced by this unit). Real output: 2/4 criteria `[FAIL]` (faithfulness 0.7977 < 0.90, kappa -0.0161 < 0.6), 2/4 `[PASS]` (custom metrics implemented, the command itself runs). Scoped to Phase 11's own build list only, not a combined retrieval+generation scorecard (Phase 19 owns that). Citation validity reports the batch's stored scalar mean only, not Unit 5's fuller per-`ClaimStatus` breakdown — the batch never persisted full `ValidationResult` objects, and getting that breakdown now would need a live tier-2 entailment call, a real disclosed scope cut. TDD'd (14 tests, including a regression guard against `docs/results/judge-agreement.md`'s exact published numbers, computed fresh from the real raw file). Design record 11e. `uv run pytest -q`: 416/416 passed (+14 new).

**Exit criteria — all 6 of 6 Phase 11 build-list units complete.** The two numeric plan.md targets themselves (faithfulness ≥0.90, κ ≥0.6) remain **not met** — real, disclosed, and now surfaced automatically every time `make eval` runs, per Definition of Done tracker rows #2 and #8.

**Judge rubric fix, applied and real-spot-checked the same session (not deferred).** Checked quota status first with a trivial real call to all 4 models — all had reset. TDD'd the fix (4 new RED tests asserting the refusal rule precedes the numeric scale, names the real override language, adds a ceiling-effect anchor, and asks for an up-front refusal classification), then rewrote `JUDGE_PROMPT_TEMPLATE`: the honest-refusal rule moved into a `CRITICAL RULE` block ahead of the scale, explicitly quoting back the override language `gpt-oss-120b`'s real reasoning had used ("does not provide concrete guidance", "leaves the problem unresolved") as exactly what not to do; added "reserve 5 — name what a 4 version would be missing" to counter Gemini's ceiling effect, without going multi-dimensional (11c's rejection of that still holds). `uv run pytest -q`: 420/420 (+4 new).

Verified live rather than trusting the tests alone (CLAUDE.md verification discipline) — re-scored the exact 4 real Unit 4 disagreement cases and 3 real non-refusal cases with the new rubric, spending only 7 Groq + 1 Gemini call before both quotas ran out again. **Real result: `gpt-oss-120b`'s scores on the 4 disagreement cases moved from `[1,2,1,1]` to `[5,5,5,4]` — an exact match to Gemini's original scores on those same rows** — direct confirmation the targeted fix works on the exact failure it targeted. The 3 non-refusal scores were unchanged (no overcorrection toward blanket high scores). Gemini's own side of the fix has zero live evidence yet (only 1 Gemini call completed before its 20/day cap, cumulative across the whole session, ran out) — design record 11f, `docs/results/judge-rubric-fix-spot-check.md`.

**Next session's first task:** re-run Unit 4's full N=30 batch under the fixed rubric once both `gpt-oss-120b` (TPD) and Gemini (20/day) quotas reset — write to a fresh raw-results file, not overwriting Unit 4's shipped `judge-agreement-batch-raw.json`, so today's real negative result stays comparable to whatever the fix produces at scale. `make eval` will pick up the new numbers automatically once that file exists; no scorecard changes needed. Carried over, untouched: Phase 9's sampling-sweep grid (separate blocker, daily quota).

**Next**
- Resume Phase 11 Unit 1: the RAGAS `Faithfulness`/`ResponseRelevancy`/`IDBasedContextPrecision`/`IDBasedContextRecall` smoke test against one real answer (now unblocked), to get a real per-query cost number before committing to the N=30 batch.
- Design record 11 not started — nothing shippable to document yet; write it once Unit 1 actually completes (should fold in this session's model-selection story, not just the RAGAS mechanics).
- Carried over, untouched: Phase 9's sampling-sweep grid (separate blocker — daily quota, not model death).

### 2026-08-18 — Phase 11: Generation Eval, RAGAS & Judging the Judge (started, blocked)

**Context.** New session, continued from Phase 10 (complete). Goal: RAGAS faithfulness/answer relevancy/context precision/context recall, an LLM-as-judge for helpfulness, Cohen's κ against hand labels, custom metrics (citation validity, escalation appropriateness, code-block correctness), and one `make eval` scorecard — plan.md's never-cut Phase 11.

**Done**
- Brainstormed and scoped Phase 11 into 6 units with the user before touching code (bounded-path classification — extends the existing DI/config/TDD pattern every phase already follows, no new spec doc needed beyond plan.md itself).
- Three real tradeoffs surfaced and decided with the user (`AskUserQuestion`): **N=30** shared generation batch for RAGAS + judge + hand-labels (not plan.md's literal 50 — real quota math: 50 real answers costs ~240k Groq tokens, 2.4x one day's 100k cap; 30 was judged likely-same-day given both providers' fresh daily quotas), **Gemini `gemini-3.5-flash`** as the starting judge (deliberately a different vendor from the Groq generator, to avoid the self-preference bias plan.md's own design-record ask names as a judge failure mode), **static syntax validity (`ast.parse`)** for code-block correctness (not sandboxed execution — real security exposure running untrusted LLM-generated code, and Phase 15 doesn't exist yet to harden against it).
- Started Unit 1 (RAGAS LLM/embeddings adapter wiring). `uv add ragas` pulled in `ragas==0.4.3` plus a large transitive stack (`langchain`, `langgraph`, `langchain-openai`, `openai`) despite this project using none of them directly — real Build-vs-Import consequence worth naming, not just a line in a lockfile diff.
- **`import ragas` was broken as installed** — `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`, traced (via a real web search, not guessed) to a documented ragas 0.4.x bug affecting any non-Google-Cloud user (GitHub issues #2741/#2745/#2753: `ChatVertexAI` moved out of `langchain-community` and ragas's eager top-level import never got updated). Fixed by pinning `ragas<0.4` + `langchain-community<0.4` (resolves to 0.3.9/0.3.31) — rejected installing the suggested `langchain-google-vertexai` workaround package as dead weight for an SDK this project will never call.
- Read the real installed `ragas==0.3.9` source directly (not trusted from docs written for a different version) to verify the actual `llm_factory`/`InstructorLLM`/`BaseRagasEmbedding`/`SingleTurnSample`/`IDBasedContextPrecision` APIs before writing any adapter code. Found `llm_factory`'s own provider dispatch only covers openai/anthropic/google(legacy `google-generativeai`)/litellm — none match this project's native `groq`/`google-genai` SDKs — but `instructor` (which `llm_factory` wraps) has first-class `from_groq`/`from_genai` adapters already, and `InstructorLLM`'s constructor takes a pre-built instructor client directly, so `llm_factory`'s narrow dispatch can be bypassed entirely rather than installing a second, legacy Google SDK alongside the one already in use. Also found `IDBasedContextPrecision`/`IDBasedContextRecall` need no LLM at all — pure ID-set arithmetic against `SingleTurnSample.retrieved_context_ids`/`reference_context_ids`, meaning this project's existing golden-set ground-truth chunk IDs cover context precision/recall for free, reserving the real LLM/quota budget for `Faithfulness`/`ResponseRelevancy` (the two metrics with no deterministic substitute).
- Wrote `src/aegis/eval/ragas_adapters.py` (`build_groq_ragas_llm`, `build_gemini_ragas_llm`, `RagasLocalEmbeddings` wrapping Phase 3's `Encoder` — zero extra quota for embeddings).
- Ran a real smoke test (one real dev-positive golden query, real dense retrieval, real Groq generation) to verify the wiring end-to-end before committing to the N=30 batch — **hit a second, much bigger problem before RAGAS was even reached**: real generation itself 404'd.
- **Investigated directly rather than patching around it**: confirmed via a live `client.models.list()` call that Groq has retired `llama-3.3-70b-versatile` *and* `llama-3.1-8b-instant` entirely — both absent from the catalog, both real 404s. This breaks every Groq default in `config.py` (`GroqSettings`, `RouterSettings`, `QuerySettings`, `CitationSettings`), not just Phase 11 — logged as Open Question #15. Real-tested replacement candidates rather than guessing: Groq's current flagship models (`openai/gpt-oss-120b`, `qwen/qwen3.6-27b`) are reasoning-first and burn hidden `reasoning_tokens` (measured: 53 tokens, consuming a `max_tokens=10` budget with zero visible output — the same failure shape as Phase 9's Gemini thinking-budget bug) that `reasoning_effort="low"` didn't eliminate; `groq/compound-mini` measured clean (real `'OK'`, no reasoning tokens) as a verified drop-in for the small-task roles.
- Stopped and surfaced this to the user rather than silently picking a replacement and burning real quota on an unvalidated model choice (CLAUDE.md: real tradeoffs get presented with a recommendation, not silently decided). Session ended before the user's decision landed.

**Exit criteria — none yet; Unit 1 of 6 incomplete**
- RAGAS faithfulness ≥ 0.90 — not measured, blocked before the first real RAGAS call could run
- `docs/results/judge-agreement.md` — not started (Units 2-4)
- Custom metrics implemented — not started (Unit 5)
- `make eval` scorecard — not started (Unit 6)

**Surprised by:** two independent, real infrastructure breaks in one session before any actual Phase 11 measurement could run — a documented ragas packaging bug affecting any non-Google-Cloud user, and Groq silently retiring the exact two models every prior phase's defaults depend on. Neither was hypothetical or found by reading changelogs; both were found by trying the real call and reading the real error, which is the same discipline CLAUDE.md's verification section asks for applied to third-party infrastructure rather than this project's own code (echoes Phase 9's "verify against a real call, not the docs" lesson with Gemini's model listing and thinking-budget default).

**Next**
- **First decision, before anything else:** finalize `GroqSettings.model`'s replacement (`openai/gpt-oss-120b`, measuring its real reasoning-token tax on actual answer-length generations before trusting it, vs. reconsidering Gemini as primary) — this was presented to the user but not yet decided as of session end. `groq/compound-mini` is already verified for the four small-task settings and just needs applying to `config.py`.
- Once Groq is generation-capable again: finish Unit 1's smoke test (RAGAS `Faithfulness`/`ResponseRelevancy`/`IDBasedContextPrecision`/`IDBasedContextRecall` against one real answer, to get a real per-query cost number before committing to the N=30 batch), then Units 2-6 as scoped above.
- Design record 11 not started — nothing shippable to document yet; write it once Unit 1 actually completes.
- Carried over, untouched: Phase 9's sampling-sweep grid (separate blocker — daily quota, not model death).

### 2026-08-17 — Phase 10: Calibrated Abstention & the Confusion Matrix

**Context.** New session, continued from Phase 9 (left nearly complete — only the sampling-sweep grid still blocked on Groq's quota, untouched this session). Goal: the confidence scorer combining 4 signals, a swept tau against the full precision/recall curve, the escalation confusion matrix with real numbers in all 4 quadrants, tau chosen with a written business justification, and an escalation payload — plan.md's never-cut Phase 10, "the component Air Canada didn't have."

**Done**
- TDD'd the confidence scorer (`guardrails/confidence.py`, 258 lines, 17 tests): `reranker_top_score` and `score_gap` (cross-encoder over only the top-3 dense candidates — rank 1/2 is all either signal reads), `semantic_agreement` and `distribution_shape` (cheap — reuse already-computed dense scores/embeddings). Same DI/fake-testing pattern as every other component (`CrossEncoderReranker`, `Encoder`). `decide_escalation`/`handle` mirror Phase 8's `router/policy.py` exactly: a pure decision function plus a single enforcement point structurally incapable of calling both `generate_fn` and `escalate_fn`, proved with a raising spy the same way router policy's test does.
- Benchmarked real cross-encoder cost on real corpus-length chunks before committing to a full run (CLAUDE.md's compute-budget rule) — confirmed Phase 7's ~780ms/pair figure was still representative, and got real `bge-reranker-v2-m3` score ranges (already sigmoid-scaled, [0,1]) to design normalization against.
- Ran `experiments/10a_tau_sweep.py` for real: dense retrieval (Phase 6's winning config) + cross-encoder reranking of a 3-candidate pool, for all 180 dev golden queries (150 positive + 30 negative), CPU. 736s (12.3 min) — a 6-query probe had projected ~5-9 min; reported the miss honestly rather than pretending the projection was right (design record 10 §6).
- **The naive equal-weighted combination plan.md suggests as a starting point measured worse than one signal alone** (0.8333 vs. 0.9333 recall at precision>=0.5) — investigated rather than shipped blind (CLAUDE.md: a surprising result demands investigation before proceeding). Root cause: `score_gap` and `distribution_shape` both measured close to uncorrelated with the real should-escalate label (rank-correlation -0.16 and -0.02) and diluted `reranker_top_score`'s real, strong signal (-0.58). Also found `distribution_shape`'s raw scale was corrupted by a single ~76,856 outlier (near-zero-variance dense scores among one query's top-k, tying to Open Question #6's near-duplicate-chunk corpus property) — fixed with winsorized (p1/p99) normalization bounds.
- Confirmed both corrections with the user (`AskUserQuestion`) before shipping, since reweighting by measured strength is a real deviation from plan.md's literal starting point: reweighted to 0.60/0.15/0.20/0.05 (recovers 0.9000 recall at precision>=0.5, vs. the 0.9333 a "drop the two weak signals" alternative would have measured — kept all 4 signals alive anyway, for the same "toggleable, not silently dropped" transparency reason Phase 7/8 kept reranking/expansion/HyDE built and measured even when off).
- Presented 3 real tau candidates with full confusion matrices (deflection-preserving tau=0.15, recall-prioritized tau=0.50, high-recall tau=0.65) and a recommendation, given plan.md's own framing of tau as "the decision that matters most" — **tau=0.50 chosen**: 27/30 (90%) of dev negatives correctly escalate (only 3 slip through — 2 `unanswerable`, 1 `out_of_corpus`; `billing_legal` and `adversarial_injection` both 100% caught), while still directly answering 126/150 (84%) of legitimate questions.
- Investigated the 3 real misses directly rather than reporting the number blind: both diagnosable misses share one mechanism — a named entity in the query genuinely appears in a retrieved chunk (temporal impossibility: "next major FastAPI version's release date" retrieves a real "FastAPI Version" chunk; cross-framework confusion: "Celery with Django" retrieves a real FastAPI+Celery chunk) — logged as new Open Question #14.
- Generated a real escalation payload against the live corpus (`negative::adversarial_injection::0`, a real prompt-injection attempt) to show what a human agent would actually see — confirms the mechanism is "nothing in the corpus supports an answer," not explicit injection detection (that's Phase 15's job).
- Wrote `docs/results/confusion-matrix.md` (full curve, per-category and per-stratum breakdowns, the two real FN examples with root cause) and `confusion-matrix.png`, and design record 10.

**Exit criteria — all met, with one honestly-reported gap**
- `docs/results/confusion-matrix.md` with real numbers in all four quadrants ✓
- tau selected from the curve with written business justification ✓ (confirmed with the user)
- Deflection rate reported at the chosen tau ✓ (0.8400)
- All negatives behave correctly — **27/30 (90%), not literally all** — the 3 real misses are diagnosed and documented (§6), not hidden. (Scope note: measured against the golden set's 30 *dev* negatives, not the literal "40" in plan.md's exit-criterion text, consistent with this project's standing dev/holdout separation — see Decision Log.)

**Surprised by:** how much worse the "obvious" starting point (equal weights) was than doing nothing extra at all (one signal alone) — not just suboptimal, actively worse, which is a sharper lesson than "the sweep found a better combination." Also surprised the two real misses weren't random noise but one clean, nameable mechanism (real entity overlap masking an unanswerable or out-of-scope question) — a more useful finding than "recall is 90%" on its own. And surprised, again, that a real-data compute benchmark (not even a synthetic one this time) still undersold the full run by ~40% — the "benchmark before committing" lesson from Phase 7 needed a buffer multiplier, not just a real sample.

**Next**
- Phase 11: generation eval, RAGAS, and judge validation (topics 26, 30, 32 continued) — the natural next check on whether this phase's tau and weights hold up against a validated LLM judge's opinion of the 126 queries now answered directly, not just against golden-set labels.
- Open Question #14 (the entity-overlap failure mode) is new and unresolved — worth revisiting if it recurs at higher volume (holdout, or production).
- Phase 9's sampling-sweep grid remains the one carried-over incomplete item, unrelated to this phase.

### 2026-08-16 — Phase 9: Grounded Generation & Citations (in progress)

**Context.** New session, continued from Phase 8. Goal: multi-provider LLM client, context budgeter, versioned prompt templates, a two-tier citation validator, a sampling sweep, and a provider bake-off, per plan.md's Phase 9 — the first phase that calls an LLM to *answer* rather than classify/rewrite/expand.

**Done**
- Confirmed with the user up front that neither a Gemini API key nor a local Ollama install existed yet, despite plan.md calling for "Groq primary, Gemini secondary, Ollama optional." User chose to provision a real `GEMINI_API_KEY` (still pending as of phase-end) rather than build untested or skip the exit criterion. Added `google-genai` as a new dependency.
- Measured real per-chunk token stats on the persisted 19,236-chunk corpus (p50=79, p90=363, p95=390) to ground `GenerationSettings.context_budget_tokens=4096` in a real number rather than a guess.
- TDD'd the context budgeter (`generation/budgeter.py`, 126 lines, 9 tests): strict-prefix greedy selection by rank (no skip-ahead), one exception (truncate rather than drop an oversized #1 chunk — real chunks reach 6,325 tokens), then a separate "lost in the middle" bookend placement pass. Citation numbers track rank, not display position.
- Built versioned prompt templates (`generation/prompts/`, v1, 4 tests): numbered `[n]` citations, explicit permission to say "I don't know," a hard no-policy-commitments rule.
- TDD'd the multi-provider fallback client (`generation/llm/`, 137 lines across `fallback.py`/`groq_provider.py`/`gemini_provider.py`, 10 tests with fakes): narrow fallback boundary (only pre-first-token failures trigger switching providers, a mid-stream failure propagates rather than splicing two providers' output together). `generate_with_provider()` reports which provider answered via a stateless internal generator, not shared mutable state — deliberately avoiding a repeat of Phase 6's `EmbeddingCache` thread-safety bug.
- TDD'd the two-tier citation validator (`guardrails/citations.py`, 196 lines, 9 tests): cheap lexical overlap first, LLM entailment (small model, JSON mode) only on claims tier 1 can't confidently pass. **TDD caught a real bug before shipping**: the first `_lexical_overlap` implementation reused Phase 4's `auto_label.py` substring-containment technique verbatim and produced false positives (`"on"` matching inside `"injection"`); fixed to exact word-boundary token-set intersection.
- Composed everything into `GenerationPipeline.answer()` (`generation/pipeline.py`, 87 lines, 5 tests) and ran a real end-to-end smoke test (real Qdrant retrieval, real Groq generation, real citation validation) — 0 missing/invalid citations, and the validator correctly flagged a real invented code example as `UNSUPPORTED`.
- Ran the sampling sweep (`experiments/09a_sampling_sweep.py`) and hit a real, hard constraint: Groq's free-tier `llama-3.3-70b-versatile` quota (100,000 tokens/day) was consumed almost entirely by one 20-query config (95,944 tokens), then fully exhausted (100,000/100,000) on a scoped-down retry. Confirmed with the user twice (`AskUserQuestion`): first to cut the grid down, then — after the cut-down retry *also* hit the wall — to document the one real, complete config (temperature=0.0, top_p=0.5, N=20: mean `valid_citation_rate` 0.6634) as a partial result rather than wait out an uncertain-duration reset.
- Proved the fallback mechanism against live infrastructure, not just fakes: a `GroqProvider` pointed at a nonexistent model raised a real `NotFoundError`, and `FallbackLLMClient` recovered via a second `GroqProvider` on a separate, unaffected quota bucket (`llama-3.1-8b-instant`).
- Wrote `experiments/09b_llm_bakeoff.py` and `docs/results/sampling-sweep.md`, and the first pass of the design record — at that point genuinely blocked on `GEMINI_API_KEY`.
- **User provisioned `GEMINI_API_KEY` mid-session.** Verified it before trusting it: `client.models.list()` said the configured `gemini-2.5-flash` was available, but a real call 404'd ("no longer available to new users") — found and fixed by testing real candidate models directly, pinning to `gemini-3.5-flash`. Found a second real bug the same way: `gemini-3.5-flash` defaults to extended thinking, and a real call at `max_tokens=20` came back with zero visible text (`finish_reason=MAX_TOKENS`, whole budget spent on hidden `thought` tokens) — fixed with `thinking_config=ThinkingConfig(thinking_budget=0)`.
- Ran the real provider bake-off (`experiments/09b_llm_bakeoff.py`). Groq's quota was still mostly exhausted (2/15 succeeded, 13 real `RateLimitError`s); Gemini succeeded on 12/15 (3 transient `503`s). Not an equal-N quality comparison, but real and decision-relevant regardless: Gemini's mean latency (21,437ms) was ~14x Groq's (1,549ms) — the concrete justification for "Groq primary, Gemini secondary." Wrote `docs/results/llm-bakeoff.md` with the full table and the two same-query head-to-head comparisons.
- Wrote and registered prompt v2 (`generation/prompts/v2.py`) targeting the diagnosed missing-citation root cause (opening sentences disproportionately uncited) with an explicit first-sentence rule and worked example. First validation attempt (`experiments/09c_prompt_v2_check.py`, against Gemini since Groq had no headroom) hit a *third* quota shape: Gemini's free tier caps requests at 20/day/model (not tokens). Got 2/10 real v1 completions, 0/10 v2.
- Fixed the citation validator's sentence-splitter blind spot on fenced code blocks (Open Question #11), no quota needed: TDD'd a reproduction of the real smoke-test bug first, then fixed by reusing Phase 2's `split_code_segments` rather than writing new fence-detection logic.
- **Retargeted v2 validation at Groq** (the actual production-primary model), reduced `max_tokens` to conserve quota, added incremental writes. Second attempt: 1/8 v1, 0/8 v2 — Groq's quota was still exhausted hours after the first wall (`Used: 98750/100000`), confirming a genuine once-a-day reset rather than fast trickle-refill. Third attempt, after actually waiting for the reset: **complete**. v1 finished all 8 sampled queries; v2 completed 6 before a later wall. On the 6 matched queries: v2 won 5, tied 1, lost 0 vs. v1 (mean `valid_citation_rate` 0.4199 → 0.5641, +34.3%). Confirmed the mechanism on real generated text — v1's answer to `issue::2689` opens with an uncited claim; v2's answer to the identical question, same context, opens with the same claim, cited `[4]`. **Promoted `GenerationSettings.prompt_version` to `"v2"`** on this evidence. 356/356 suite passing throughout.

**Exit criteria — all 5 addressed with real evidence**
- Streams tokens with working provider fallback ✓ — proven against live Groq *and* Gemini APIs (forced real failure → real recovery on Groq; a real chained Groq/Gemini call also succeeded)
- Validator catches deliberately corrupted citations ✓ — `test_invalid_citation_when_marker_points_to_a_chunk_not_in_context`, and the real smoke test caught a genuine hallucinated code example
- `docs/results/llm-bakeoff.md` ✓ — real data, both providers, honestly caveated unequal sample size
- 100% of factual claims carry a parseable citation — v1 measured 78.9% (21.1% `missing_citation`); root-caused, fixed (v2), and re-measured: real matched-pair result is 5 wins/1 tie/0 losses, now shipped. The underlying number still isn't literally 100%, but the gap is diagnosed, fixed, and measurably smaller, not left as an unaddressed miss.
- `docs/results/sampling-sweep.md` — partial: one real, complete config; the rest still blocked on Groq's daily quota (the one item without a full resolution this session)

**Surprised by:** how many distinct, real free-tier constraints one session could surface across two providers, and that both turned out to be genuine once-a-day resets rather than fast-refilling buckets — confirmed only by a second real attempt hours later hitting the identical wall, not assumed from a successful small probe call. Also surprised, twice, by real Gemini integration bugs invisible from documentation/list-endpoint inspection alone (the 404-despite-listed model, and the silent thinking-mode token sink) — "verify against a real call" (CLAUDE.md §5) applies to third-party integrations as much as to this project's own code. And genuinely pleased, not just surprised, that the full measure→diagnose→fix→re-measure loop on the citation-coverage gap actually closed cleanly: a real, mechanistically-understood problem, a targeted fix, and real evidence the fix worked, checked against actual generated text rather than trusted on the aggregate number alone.

**Next**
- Complete the remaining sampling-sweep temperature x top_p configs once Groq's quota allows another multi-query run — the script needs no further changes.
- A larger confirmatory run for v2 (N≥15-20, matching v1's original baseline scale) would still strengthen the evidence, though it isn't required to justify the default already shipped.
- Phase 9 is close to fully done; only the sampling-sweep grid is still incomplete.

### 2026-08-16 — Phase 8: Agentic Router & Query Understanding

**Context.** New session, continued from Phase 7. Goal: an LLM router with a hard-coded escalation policy for billing/account/legal queries, plus four query-understanding techniques (rewriting, multi-query expansion, HyDE, iterative retrieval), each A/B'd against Phase 6's winning retriever, per plan.md's Phase 8.

**Done**
- Added `RouterSettings` and `QuerySettings` to `config.py`. Router and query-rewriting/expansion use a small/fast model (`llama-3.1-8b-instant`) deliberately distinct from the eventual generation model; HyDE stays on the larger `llama-3.3-70b-versatile` since its hypothetical-answer content directly determines embedding quality, not just latency.
- TDD'd `RouterClassifier` (`router/classifier.py`, 131 lines, 8 tests) and `router/policy.py` (51 lines, 7 tests) — `billing_account_legal` hard-escalates via a pure `decide()` function and an enforcement point (`handle()`) that structurally cannot call both `generate_fn` and `escalate_fn`; proved with a `generate_fn` spy that raises if ever invoked. Every classifier failure mode (malformed JSON, out-of-vocabulary label, a real Groq `APIError` hit mid-eval) fails safe to the same escalate outcome, not three different paths.
- Router eval (`experiments/08a_router_eval.py`): 58 golden queries labeled by reusing `strata.py`'s existing how_to/error_debugging classification (decided with the user 2026-08-12) + all dev-split billing_legal negatives. Confusion matrix: 68.97% overall accuracy, but `billing_account_legal` recall is **1.0000** — the number that matters. `known_issue` recall (0.48) investigated directly: partly a ground-truth artifact (`strata.py` classifies on title+body+*resolution*, so a query can be labeled `error_debugging` from a word appearing only in the maintainer's reply, not the user's actual question).
- Built and TDD'd four query-understanding wrappers, same toggleable-composition shape as Phase 7's `RerankedRetriever`: `QueryRewriter` (75 lines, 4 tests), `ExpandedRetriever`/`QueryExpander` (113 lines, 6 tests — caught a real dedup bug where whitespace-only paraphrases weren't filtered, fixed before it shipped), `HydeRetriever`/`HydeGenerator` (100 lines, 4 tests), `IterativeRetriever`/`QueryReformulator` (106 lines, 6 tests).
- Built the 24-item synthetic multi-turn eval set for rewriting (`experiments/08b_generate_rewrite_eval_set.py`, LLM-generated per the user's 2026-08-12 decision, 4/stratum sampled, anchored to real golden-query ground truth).
- Ran the full ablation (`experiments/08c_query_understanding_ablation.py`, 150-query dev set). **First run hung 17+ minutes at 0% CPU with ~3GB swap** — the identical MPS-under-memory-pressure signature Phase 3 and Phase 7 already documented; I just hadn't applied that lesson to this new script. Fixed the same way (forced `device="cpu"`), re-ran clean.
- **Investigated both regressions directly rather than reporting them blind** (CLAUDE.md: a result this far outside expectation demands investigation). Multi-query expansion (−49.4% NDCG@5, the worst regression measured anywhere in this project): pulled a real example and found the small paraphrase model sometimes generates invented answers instead of rephrasing the question, and even reasonable paraphrases let unweighted 4-way RRF fusion let 3 wrong lists outvote 1 correct one. HyDE (−23.6%): found it's the *same* "sharp technique loses to broad ground truth" mechanism Phase 7 found for the cross-encoder reranker — an independent second confirmation of that pattern (Open Question #8 updated).
- Wrote `docs/results/routing-confusion-matrix.md`, `docs/results/query-understanding-ablation.md`, and the design record. Set `QuerySettings.rewrite_enabled = True` (the one technique that measurably pays for itself); expansion/HyDE/iterative retrieval stay off, each for a distinct, documented reason.

**Exit criteria — all met**
- Routing confusion matrix in `docs/results/` ✓
- Test asserts the billing route cannot reach generation ✓ (`test_router_policy.py::test_generate_fn_that_raises_on_call_proves_billing_never_invokes_it`)
- `docs/results/query-understanding-ablation.md` — each technique's Recall@10 delta and latency cost ✓
- Techniques that don't pay for themselves are disabled, decision written down ✓ (3 of 4)

**Surprised by:** how much more useful "investigate the surprising result" was here than in Phase 7 — not one mechanism but two entirely different ones for two different techniques, both real and independently verifiable against live data, not post-hoc rationalization. Also surprised at myself, mildly: this project has two prior decision-log entries about the exact MPS-hang failure mode, and I still wrote a new script that hit it a third time by not re-reading them before choosing a device. The fix was trivial once diagnosed; the miss was in not applying an already-paid-for lesson.

**Next**
- Phase 9: grounded generation & citations (topics 3, 21-25) — multi-provider LLM client, context budgeter, versioned prompts with citations, the two-tier citation validator, sampling sweep, provider bake-off. First phase that calls a real LLM to *answer* rather than classify/rewrite/expand.
- Open Question #8 (golden-set ground-truth permissiveness) now has a second independent confirmation (HyDE, this phase) on top of Phase 7's reranker finding — still waiting on Phase 11's Cohen's κ to quantify directly rather than remain inferred.
- New Open Question #9: multi-query expansion's unweighted RRF fusion mechanism (breadth-across-lists outvoting peak relevance) is understood but not fixed — only matters if expansion is revisited.
- Open Questions #6 and #7 remain open, unchanged this phase.

### 2026-08-11 — Phase 7: Reranking

**Context.** New session, continued from Phase 6. Goal: cross-encoder reranker on top of Phase 6's winning config, pool-size-vs-latency sweep (10/20/50/100), and a direct measurement of the bi-encoder-vs-cross-encoder gap, per plan.md's Phase 7.

**Done**
- Added `RerankSettings` to `config.py` (model, device, batch size, swept pool sizes, `final_top_k`, latency budget reference, and — added at the end of the phase — `enabled: bool = False`).
- TDD'd `CrossEncoderReranker` (`rerank/cross_encoder.py`, 74 lines, 6 tests) and `RerankedRetriever` (`rerank/pipeline.py`, 46 lines, 5 tests) — same dependency-injection/composition patterns as `Encoder` and `HybridRetriever`.
- **Local hardware could not run the sweep.** MPS hit macOS's "stuck" scheduler state twice (first from two large transformer models MPS-resident simultaneously — the bi-encoder never freed before the reranker loaded, echoing a Phase 3 decision-log warning almost exactly; then again after fixing that, under this session's broader system-wide memory pressure). Pivoted to CPU, which was stable but — benchmarked on short synthetic text at ~30.5ms/pair, not real corpus-length text — turned out to actually run at ~780ms/pair on real ~400-token chunks, projecting a ~5.7 hour sweep after 19 real minutes of a doomed run. Killed it.
- **Moved compute to a free Kaggle T4 GPU.** Built `experiments/07a_export_rerank_candidates.py` (local dense retrieval only, cheap, exports each query's top-100 candidate pool with real chunk text) and `experiments/07b_kaggle_rerank_sweep.py` (standalone, no `aegis` import, runs on Kaggle). First Kaggle attempt also failed: `BATCH_SIZE=64` exceeded every swept pool size, so each query's whole pool was scored as one un-chunked batch; at pool=50 that, stacked on ~300 prior forward passes' worth of fragmented GPU memory with no cache-clearing, produced a real CUDA OOM (3.54GiB requested, 2.95GiB free, 11.61GiB resident, T4's ~14.56GB total). Fixed with `BATCH_SIZE=16`, `torch.cuda.empty_cache()` between pool sizes, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and incremental per-pool-size result writes (so a second crash couldn't cost already-collected data again). Re-run completed all 4 pool sizes.
- **The result contradicted the phase's premise, so investigated rather than reported blind.** Every pool size made NDCG@5 *worse* than the no-rerank baseline (−9.0% at pool=10 to −55.8% at pool=100), monotonically. Pulled one real query's top-20 reranked output locally and confirmed the model itself is not broken — it correctly ranked the single most on-point passage #1. Root cause: `auto_label.py`'s ground-truth labeling (token-overlap ≥15%, explicitly "biased toward inclusion") produces a median of 17 ground-truth chunks spanning 12 documents per query — a broad-topic bar a bi-encoder's fuzzy similarity clustering happens to satisfy better than a cross-encoder's sharp, specific relevance judgment.
- Wrote `docs/results/rerank-tradeoff.md` (full table + plot) and the design record, covering bi-encoder/cross-encoder/ColBERT attention patterns, why reranking can't replace retrieval (O(n) scoring, no precomputable index), and the full measured curve.
- Set `RerankSettings.enabled = False` as the shipped default, with the measured reasoning in both `config.py` and the design record.

**Exit criteria — all met**
- `docs/results/rerank-tradeoff.md` with the pool-size curve ✓
- Pool size chosen against the latency budget, with reasoning ✓ (none clears the budget; recommended starting point if reranking is revisited is pool=10, documented as the least-bad, not a good, option)
- NDCG@5 improvement over Phase 6's best, as a number ✓ (negative: −0.0677 absolute / −9.0% relative at best)
- Reranking toggleable ✓ (`RerankedRetriever(..., enabled=...)`, tested)

**Surprised by:** the hardware saga repeating the exact same mistake twice in two different forms — benchmarking cross-encoder throughput on short synthetic text and extrapolating to real ~400-token corpus chunks, first on CPU (25x underestimate) then again on a GPU via too-large a fixed batch size (OOM instead of just "slow," but the same root cause: not validating a resource assumption against real data length before committing to a long run). The larger surprise was that reranking *hurt*, and that the reason traced cleanly back to Phase 4's golden-set labeling methodology rather than anything about the reranker or this phase's own code — a genuinely better relevance judge losing to a noisier ground truth is a real, reportable finding, not a failed experiment, and it's the kind of result CLAUDE.md explicitly wants surfaced rather than buried under a "reranking didn't help, moving on."

**Next**
- Phase 8: agentic router & query understanding (topic 17, query parsing — the one Module 3 topic Phase 7 didn't touch). Router, routing policy (billing/legal hard-escalates, never reaches generation), query rewriting, multi-query expansion, HyDE, iterative retrieval — each A/B'd honestly per plan.md step 7.
- Open Question #8 (this phase): does the golden set's permissive token-overlap labeling distort Phase 6's absolute numbers too, not just Phase 7's reranking comparison? Natural checkpoint is Phase 11's Cohen's κ measurement against an Opus judge.
- Open Questions #6 and #7 (duplicate `_No response_` chunks; harness cache not detecting index regeneration) remain open, unchanged this phase.

### 2026-08-04 — Phase 6: Retrieval Evaluation & the Ablation Grid

**Context.** Continued from Phase 5 (session picked up mid-work after a gap — Docker had stopped and needed restarting; the eval harness and `EmbeddingCache` thread-safety fix from before the gap were already in place and verified). Goal: `ranx` metrics, a config-hashed eval harness, and the full chunking × enrichment × retrieval-strategy × encoder ablation grid against the golden set, per plan.md's never-cut Phase 6.

**Done**
- Verified `ranx` against a hand-computed 5-document example (exact match on Precision@5/Recall@5/MRR/MAP@5/NDCG@5); found and normalized a real library quirk (`evaluate()` returns a bare scalar for one metric, a dict for multiple).
- Built the eval harness (`eval/harness.py`): `RetrievalConfig` (frozen dataclass, hash excludes the human-readable `name`), `run_eval` (parallelized, per-stratum, cached by `config_hash + queries_hash` — deliberately not config hash alone, since a same-config-different-queries collision was caught by a dedicated test before shipping).
- **Found and fixed a real threading bug** while running the harness for real the first time: `EmbeddingCache`'s single `sqlite3.Connection` isn't safe across the harness's `ThreadPoolExecutor` workers. TDD'd a reproducing test, fixed with `check_same_thread=False` + a lock.
- Ran the free axes first (retrieval strategy, encoder — both against already-indexed Phase 3 collections, zero new compute). **Found hybrid fusion (RRF/linear) underperforming pure dense** at default equal weights; investigated, tuned weights and `k` (dense_weight sweep, retrieved each query's candidate pool once and re-fused in-memory rather than re-querying per config), confirmed hybrid only ever *approaches* dense's performance as `dense_weight → ∞`, never exceeds it — true in aggregate and in every one of the six strata, not just on average.
- **Found and fixed a second real bug**: `enrichment_prefix` (Phase 2 code) duplicated the document title verbatim for every issue chunk, because `github_issues.py` prepends the title as the doc's own H1 and `heading_path` already starts with it. TDD'd a regression test, fixed. Effect was small (recall@10 0.1364 → 0.1454) — the larger cause turned out to be structural: GitHub issue-template boilerplate headings (`Description`, `Question`, `How to do it`) shared across thousands of unrelated issues compress dense retrieval's discriminative signal when prepended. Contextual enrichment measured as a clear net negative on this corpus (0.4288 → 0.1751 recall@10) — the opposite of plan.md's stated default expectation, with a real mechanistic explanation, not just a number.
- **Found and fixed the biggest bug**: every golden query's own source issue is indexed in the corpus and is a trivial near-verbatim top-1 match — Phase 4 already solved this exact problem for candidate labeling (`must_not` filter) but it was never carried into Phase 6's actual evaluation. Silently inflated every measurement in this phase's first pass (most visibly MRR: dense 0.616 → 0.864 once fixed). Added `exclude_doc_id` as a first-class parameter on `SparseIndex`/`DenseIndex`/`HybridRetriever.search` (native Qdrant `must_not` for dense, over-fetch-then-filter for sparse), then re-ran every prior measurement.
- Built document-level ground-truth transfer (`eval/doc_level.py`, TDD'd) to compare chunking strategies fairly — their chunk IDs live in disjoint spaces, only `doc_id` is stable across all four chunkers.
- Chunked and embedded the full corpus with the two strategies never persisted before (`experiments/06a_prepare_ablation_indexes.py`): fixed (8,984 chunks, 830s), recursive (9,561 chunks, 741s), semantic (9,168 chunks, 723s chunking + 640s embedding — the first full-corpus semantic-chunking run ever attempted in this project, resolving Open Question #5). Also re-embedded heading-aware with enrichment on (twice — once before, once after the title-duplication fix).
- Ran the full grid. Winner: heading-aware chunking, enrichment off, dense retrieval, `bge-large-en-v1.5`. Wrote `docs/results/retrieval-ablation.md` and the design record.

**Exit criteria — all met**
- `docs/results/retrieval-ablation.md` with the deliverable table ✓
- Winning config selected with written reasoning ✓
- Recall@10 and MRR lift from baseline stated as numbers ✓ (+0.332 / +0.496 vs. worst measured config — Phase 0's literal baseline isn't corpus-comparable, see results doc §7)
- Per-stratum breakdown showing where it's weak ✓ (dense wins/ties every stratum; sparse weakest on `version_specific`, counter-intuitively)
- Harness reruns any config from a hash ✓ (`load_cached`)

**Surprised by:** three real production bugs in one phase, each a different mechanism (thread-safety, a data-duplication bug in text construction, a methodology gap not carried forward from an earlier phase) — all invisible to unit tests because none of them involve running the full pipeline against real golden-set queries, which is the one thing this phase does for the first time. Hybrid search *not* beating dense, in aggregate and in every stratum, on a golden set that's mostly natural-language questions rather than literal identifiers (contrast with Phase 5's deliberate `ECONNREFUSED` spot-check, which is exactly the query shape where BM25 does win). Contextual enrichment being a clear net negative — plan.md's "usually the cheapest large win" assumption held for the docs corpus it was probably written for, not for this one, which is 83% GitHub issues with template boilerplate. Semantic chunking finishing in ~23 minutes total, not the "multi-hour job" Phase 2 worried about — the concern scaled from one large document's cost, but most of the corpus is short issues.

**Next**
- Phase 7: reranking. Cross-encoder (`bge-reranker-v2-m3`) on top of this phase's winning config; measure the pool-size vs. latency tradeoff and the bi-encoder-vs-cross-encoder gap directly. The sparse-side `doc_type`-filtering gap (Phase 5 design record) and Open Question #6 (duplicate `_No response_` chunks) remain open — worth a look if per-stratum weakness shows up again once reranking is in place. Open Question #7 (harness cache doesn't detect index regeneration) is a standing risk for any phase that rebuilds a collection without changing a config field.

### 2026-08-03 — Phase 5: Retrieval Core — Analyzer, Sparse, Dense & Fusion

**Context.** New session, continued from Phase 4. Goal: the actual retrieval pipeline Phase 6 will measure — custom analyzer, BM25 (`bm25s`), dense retrieval with metadata pre-filtering, RRF and linear fusion, and one composed interface across all four strategies. `analysis/` and `retrieval/` were empty stubs at session start.

**Done**
- TDD'd the custom analyzer (`analysis/analyzer.py`, 75 lines): one rule applied uniformly — a compound identifier (`snake_case`, `dotted.path`, `SCREAMING_CASE`, `--flag`) is emitted whole *and* decomposed into parts, except version/IP-style numeric sequences (`0.100.0`, `127.0.0.1`), kept whole only. Test suite built from 40 real tokens pulled from `documents.jsonl` (not invented), per the exit criterion.
- **Found and fixed a real regex bug during RED→GREEN**: `_TOKEN_RE`'s `--?` was meant to mean "0-2 leading dashes" but actually requires exactly one mandatory dash — every plain word in the corpus silently vanished from tokenization, only `--flag`-style tokens survived. 46 of the first 56 tests failed at once; fixed to `-{0,2}`.
- Built `SparseIndex` (`retrieval/sparse.py`): `bm25s` (imported) wired to the custom analyzer instead of its own default tokenizer. TDD'd with 5 tests, including the literal `ECONNREFUSED` exit-criterion case.
- Built `DenseIndex` (`retrieval/dense.py`): wraps Phase 3's Qdrant collections with native metadata pre-filtering (`product_area`, `doc_type`), reusing Phase 3's own finding that pre-filtering (not post-filtering) is required for reliable recall. Chunks are hydrated from the in-memory corpus by ID, not reconstructed from Qdrant's payload (which lacks `content_hash`/`embedding_text` by Phase 3 design) — verified against the live `aegis_chunks_bge_small` collection (19,236 points).
- TDD'd RRF and linear fusion together (`retrieval/fusion.py`, 83 lines, 12 tests): `score(d) = Σ weight_r / (k + rank_r(d))` for RRF; per-list min-max normalization + weighted sum for linear, with a divide-by-zero guard for degenerate equal-score lists.
- TDD'd `HybridRetriever` (`retrieval/hybrid.py`, 70 lines, 8 tests, fake indexes as spies): one `.search(query, strategy)` interface over `dense`/`sparse`/`hybrid_rrf`/`hybrid_linear`, each side pulling `max(top_k, candidate_pool_size)` before fusing so fusion has breadth to find cross-method overlap in.
- Ran the full pipeline against the real 19,236-chunk corpus: BM25 puts the real `ECONNREFUSED` issue at rank 1 (score 3.1467); dense search doesn't find it in the top 200 at all. Confirmed the analyzer, `SparseIndex`, `DenseIndex`, and `HybridRetriever` all work end-to-end together, not just in isolation.
- Design record: `docs/explainers/05-lexical-dense-and-fusion.md` — the analyzer's decomposition rule, RRF/linear fusion derivations, four rejected alternatives (Qdrant native sparse, payload-only chunk reconstruction, stemming/stopwords, camelCase splitting), and the regex bug as a "what surprised me."

**Exit criteria — all met**
- Analyzer tests green on 40 real corpus tokens ✓ (56/56 total)
- BM25 returns sane results for `ECONNREFUSED`-style queries where dense fails ✓ (measured: rank 1 vs. absent from top 200)
- RRF and linear fusion both implemented and unit-tested ✓ (12 tests)
- All retrieval modes behind one interface ✓ (`HybridRetriever` + `RetrievalStrategy`)

**Surprised by:** the regex bug (`--?` vs. `-{0,2}`) failing loud and immediately rather than subtly — a reminder that testing against real corpus tokens catches classes of bugs a hand-picked example wouldn't (a synthetic test with only `--reload`-style examples would have passed while silently dropping every plain word). Unfiltered hybrid retrieval is still issue-dominated even after fusing two rankings, because fusing two individually issue-heavy lists doesn't fix the underlying imbalance — `HybridRetriever` doesn't yet forward `doc_type` filtering to the sparse side, logged as a Phase 6 watch item. RRF's `k=60` compressing the gap between nearby top ranks more than expected — a test asserting "cross-method agreement beats a single #1" needed the single-list rank gap widened to ~10 before it held, which says something real about what `k` actually controls.

**Not measured yet (Phase 6's job):** `k1`/`b` numeric sweep against the golden set; Recall@10/MRR comparison across `dense`/`sparse`/`hybrid_rrf`/`hybrid_linear`; whether RRF or linear fusion actually wins on this corpus. All the relevant knobs (`bm25_k1`, `bm25_b`, `rrf_k`, `rrf_dense_weight`, `rrf_sparse_weight`, `hybrid_candidate_pool_size`) are in `config.py`, not hardcoded, specifically so Phase 6 can sweep them without touching this code.

**Next**
- Phase 6: the ablation grid. `ranx` metrics, the config-hashed eval harness, and the actual sweep (chunking × enrichment × retrieval strategy × encoder) against the golden set's 180 dev queries, reported per-stratum. This is the phase that turns Phase 5's "it works" into "here's the number, and here's why this config." Also worth checking there: whether the sparse-side `doc_type` filtering gap (see Surprised by) shows up as a measurable per-stratum weakness.

### 2026-08-03 — Phase 4: The Golden Evaluation Set

**Context.** Continued from Phase 3. Goal: harvest 180+ labeled queries (≥20 per stratum) from FastAPI's 2,204 resolved issues, combine with 40 negatives, freeze and hash. Phase 4 is the most critical phase in the schedule — Phases 5–11 need a fixed benchmark, and every claim downstream depends on numbers measured against this set.

**Done**
- Implemented stratification heuristic (`strata.py`): 6 categories (how-to, error/debugging, conceptual, version-specific, multi-hop, ambiguous), priority-ordered with version/error/multi-hop detecting strongest signals first, fallback to conceptual/how-to cues, ambiguous as catch-all. TDD'd with 10 real corpus examples.
- Measured real stratum distribution (2,109 post-low-signal-filter): error_debugging is 52% of corpus (1,096), conceptual is the smallest (41, 1.9%), multi-hop is larger than expected (13.3%).
- Built harvest (`harvest.py`): load resolved issues, filter out ~4.3% with non-substantive resolutions (pure redirects, canned replies, ≤6 words), stratify, sample. Includes low-signal detection: bare "thanks" or ≤30 words + explicit redirect pattern.
- Hand-authored 40 negatives (`negatives.py`, 10 per category): out_of_corpus (Django, Flask, etc.), billing_legal (refunds, accounts, legal), unanswerable (opinion, roadmap, guarantees), adversarial/injection (prompt injection attempts for Phase 15).
- Ran retrieval (`retrieve_candidates.py`) on 333 candidates (2x buffer per stratum + 40 negatives). **Found boilerplate-pollution failure mode in dense search:** unfiltered top-20 was 83.6% issue chunks and 30.9% of queries got zero doc chunks — same GitHub issue-template boilerplate problem Phases 1–3 each hit independently. Fixed: split into 12 doc + 8 issue filtered searches instead. Result: 100% of queries now guaranteed documentation representation.
- **Found and blocked a self-match leak** (smoke test): a query's own source issue scored 0.958 (vs. 0.7–0.8 for real answers), trivially dominating every retrieval result. This would have made circular ground-truth selection possible, inflating Phase 6's numbers. Added `must_not` filter to exclude source_issue from retrieval.
- Labeled all 180 positives programmatically (Haiku + token-overlap heuristic, ≥15% query-token overlap in chunk text = relevant). Conservative, deterministic, auditable. Phase 11 will independently re-label via Opus and measure Cohen's κ agreement.
- Froze the golden set (`freeze_golden_set.py`): combined labeled positives + negatives, applied holdout split (5 pos/stratum + 10 neg, ~13.5%), hashed both splits (SHA256 on sorted lines). Frozen files: `golden_set_v1.jsonl` (dev, 180 queries) and `golden_set_holdout_v1.jsonl` (holdout, 40 queries).
- Design record: `docs/explainers/04-building-a-golden-set.md` — methodology, failure modes (boilerplate-pollution, self-match leak), why stratification matters, why Phase 4 must be benchmarked before optimization.

**Exit criteria — all met**
- 180 labeled queries, 30 per stratum ✓
- 40 negatives with expected behaviors ✓
- Every query has ≥1 verified ground-truth chunk (measured: per-stratum averages 2–4 chunks selected per query depending on stratum)
- Frozen, committed, hashes recorded in progress.md ✓
- Holdout split sealed ✓

**Surprised by:** boilerplate-pollution hit retrieval harder than expected (30.9% of queries left doc-less despite docs being 11% of corpus), revealing it's a systemic failure mode across multiple phases, not a one-off. The self-match leak was caught via smoke-test before going live — a reminder to always run the tool end-to-end once before delegating. Conceptual queries being genuinely rare (1.9% of corpus) — they'll be the hardest stratum to measure and improve on.

**Next**
- Phase 5: Retrieval core (custom analyzer, BM25, dense search, RRF fusion). Build the retrieval pipeline; Phases 5 and 6 ship together (implementation + measurement).

---

### 2026-08-02 — Phase 3: Embeddings, distance & the vector store

**Context.** New session, continued from Phase 2. Goal: embed and index the full
corpus with real candidate encoders, prove cosine≡dot on real vectors, measure
HNSW's recall/latency tradeoff and pre- vs. post-filter metadata filtering — all
per plan.md's exit criteria. Started by persisting the first real chunked corpus
(`data/processed/chunks.jsonl`, heading-aware, provisional — Phase 2 deliberately
never wrote one), since a user question ("where are the chunks stored?") surfaced
that the project had built four chunking strategies but never actually committed
one to disk.

**Done**
- Environment: started Docker/Qdrant (had stopped), added `torch` +
  `sentence-transformers` (MPS confirmed available), added `matplotlib` for the
  required recall-vs-latency plot
- Built, TDD'd: `EmbeddingCache` (SQLite, keyed on `(model, content_hash)`),
  `Encoder` (sentence-transformers wrapper, dependency-injectable for fast unit
  tests, cache-aware de-duplication verified with a call-counting fake), brute-
  force exact k-NN plus two recall metrics
- Built `vector_store.py` (Qdrant collections, named `dense` vector, payload
  indexes on `product_area`/`doc_type`/`doc_version`) and the embed-and-index
  orchestration script
- Embedded and indexed the full 19,236-chunk corpus with all three candidate
  encoders (`bge-small` dev workhorse, `bge-large-en-v1.5`, `BGE-M3`) into three
  separate Qdrant collections
- **Hit real M1 16GB memory pressure** on the first (bge-small) full run —
  1,517s actual vs. ~163s predicted from a clean 200-chunk sample, confirmed via
  `vm.swapusage` (~3.5GB swapped) and a specific findable trigger (a batch
  spanning a legitimate 2,830-token atomic-code-block chunk forcing the whole
  batch to pad to 512 tokens). Fixed with sort-by-length batching, a smaller
  batch size for the two large models, and periodic `torch.mps.empty_cache()`
  — `bge-large` (665s) and `BGE-M3` (799s) both then finished *faster in
  wall-clock time* than bge-small's unmitigated run, despite being much bigger
  models
- Proved cosine≡dot on 2,000 real chunk-pair vectors (max abs diff 1.79e-07,
  not a toy example)
- Ran the HNSW sweep (36 configs: m×ef_construct×ef_search) against
  brute-force KNN ground truth. **Found ID-set recall was structurally broken**
  on this corpus — flat ~0.67 regardless of parameters, which is impossible.
  Root cause: 39.7% of the corpus's vectors are exact duplicates (the largest
  group: 1,354 issues whose entire chunk is the GitHub issue-template
  placeholder `_No response_` — the same boilerplate-pollution category Phase 1
  and Phase 2 each independently hit). Built a score-threshold recall metric
  that credits any tied neighbor; corrected recall was 1.0000 across all 36
  configs — HNSW's approximation costs nothing at this corpus scale
- Measured pre-filter vs. post-filter metadata filtering across 5 `product_area`
  categories. **Found a second real bug**, in the score-threshold metric just
  built: it divided by the number of results *returned* instead of *requested*,
  so a method returning too few results could still score a perfect 1.0 — caught
  because the experiment's own output was self-contradictory ("short of results
  on 100% of queries" next to "recall=1.0000"). Fixed and regression-tested.
  Corrected result: naive post-filtering scores 0.1667–0.4033 recall depending
  on category (even `general`, 26% of the corpus, comes back short 90% of the
  time); native pre-filtering scores 1.0000 on every category tested
- Design record: `docs/explainers/03-embeddings-distance-and-ann.md`, including
  a §3a on contrastive training (topic 11) — cosine is meaningful here because
  these specific encoders are trained to make it so, not by default

**Exit criteria — all met**
- All chunks embedded and indexed; cache confirmed working (full re-embed after
  a cold run: 0.9s, 100% cache hits)
- `docs/results/ann-recall-vs-latency.md` with a real plot
  (`ann-recall-vs-latency.png`)
- Pre-filter vs. post-filter difference measured
  (`docs/results/prefilter-vs-postfilter.md`)
- Both candidate encoders (`bge-large-en-v1.5`, `BGE-M3`) indexed, alongside the
  dev workhorse

**Surprised by:** two real bugs, both in measurement code I had just written, not
the system under test — and both caught the same way: noticing a result that
couldn't possibly be true (flat recall that doesn't respond to a search
parameter; "short of results" next to "perfect recall") rather than writing it
down. The M1 memory-pressure risk CLAUDE.md already named turned out to be
exactly right, and took an actual 25-minute bad run to be believed. HNSW recall
saturating at 1.0 regardless of parameters at ~19K points — the tradeoff curve
plan.md anticipated doesn't exist yet at this scale. The pre/post-filter gap
being large even for the corpus's single biggest category, not just narrow ones.

**Next**
- Phase 4: the golden evaluation set. 180 labeled queries + 40 negatives,
  harvested from closed issues with maintainer resolutions (2,204 available,
  Phase 1). This replaces Phase 3's self-sampled-query methodology with real
  query relevance for everything downstream. Open question #6 (the "_No
  response_" duplicate-boilerplate chunks) should get a first real signal once
  golden-set labeling is underway — worth checking whether they ever surface as
  spurious top-k results before building a special-case filter for them.

### 2026-07-30 — Phase 2: Chunking laboratory

**Context.** New session, continued from Phase 1. Goal: four chunking strategies
behind one config-swappable interface, benchmarked against each other on the real
Phase 1 corpus — no winner declared yet (that needs Phase 4's golden set and
Phase 6's ablation grid).

**Done**
- Built shared infrastructure first, TDD'd: `tokens.py` (token counting via
  `tiktoken`/`cl100k_base`, plus a byte-level-BPE safe-boundary check —
  slicing an arbitrary token window can silently split a multi-byte
  character, and `tiktoken.decode()` replaces it with `�` instead of
  raising), `segments.py` (text/code segment splitting, fenced blocks kept
  atomic), `merge.py` (greedy pack-with-overlap), `base.py` (the `Chunker`
  protocol + contextual-enrichment helper)
- Built and TDD'd all four strategies: fixed-size (token sliding window),
  recursive (paragraph → sentence → word descent), heading-aware (splits on
  Markdown headings, code blocks atomic — the flagship, ~132 lines matching
  plan.md's estimate), semantic (sentence embeddings, cosine-distance
  breakpoints, dependency-injected embedder for fast unit tests)
- Wired contextual enrichment (`{title} > {heading_path} > {version}`,
  toggleable, stored in a separate `Chunk.embedding_text` field so citations
  keep showing real source text) and a config-driven `build_chunker()`
  factory — the "one interface, config-swappable" exit criterion
- Ran the measurement script (`experiments/02_chunking_stats.py`) against
  129 docs + a seeded issue sample + issue #2266, then found and fixed
  **three real bugs**, all surfaced only by measuring the full corpus (unit
  tests use small clean inputs by construction): (1) `recursive_split_pieces`
  and `greedy_merge` each inserted their own separator at a merge boundary,
  doubling it and inflating real chunk token counts past `max_tokens`
  (issue #15000, 443/400); (2) even after that fix, `greedy_merge`'s
  sum-of-parts token accounting still didn't match the true joined text's
  token count (BPE isn't additive across concatenation) — redesigned to
  verify by actually joining and re-counting before accepting each piece;
  (3) the fenced-code-block regex required a closing line to be *only*
  backticks, which real issue text violates twice over (CRLF line endings,
  a closing fence sharing its line with trailing HTML) — either one silently
  swallowed every heading and paragraph after it into one giant code segment
  (issue #15000 went from 2 detected sections to the correct 9 once fixed)
- After fixing all three: 0/8,984 fixed-strategy chunks and 0/9,561
  recursive-strategy chunks exceed `max_tokens` over the *full* 3,659-document
  corpus (no sampling). Heading-aware: 427/19,236 exceed it — 420 are the
  documented atomic-oversized-code-block exception, 7 are an accepted
  residual limitation (documents with a genuinely odd count of literal
  `` ``` `` in the raw text — an actually-unclosed fence in the source)
- Corrected a real overclaim before it shipped: `release-notes.md` and issue
  #2266 (Phase 1's flagged stress cases) were assumed to trigger the
  atomic-oversized-code exception. Measured result: neither does, at
  `max_tokens=400`. Found the real example instead —
  `doc::advanced/security/oauth2-scopes.md`, a 184-line/1,352-token example
  script included via a Phase 1 `{* *}` directive and repeated across 8 of
  27 sections
- Design record: `docs/explainers/02-chunking-strategies.md`

**Exit criteria — all met**
- Four strategies, one interface, config-swappable (`build_chunker(strategy)`)
- Tests: no chunk exceeds `max_tokens` (fixed/recursive: 0 violations,
  full corpus; heading-aware: only the documented atomic exception + the
  quantified residual limitation, both explained, neither silent); heading-aware
  never splits a fenced block (code-block integrity 98.4%–99.1% for
  heading-aware/semantic, the shortfall from the same residual limitation)
- Contextual enrichment toggleable (`ChunkingSettings.contextual_enrichment`)
- `docs/results/chunking-stats.md` — chunk count, token distribution,
  code-block integrity rate, wall time per strategy, plus a stress-case
  spotlight table

**Surprised by:** three real, corpus-driven bugs that no unit test caught because
unit tests use small clean inputs — found only by actually measuring the full
corpus, which is the whole reason Phase 1's "actually open the parsed output"
discipline exists. Semantic chunking's per-sentence embedding cost is large and
real (~211–248s for one big document, confirmed compute-bound not overhead)
independent of whether it wins on retrieval quality. The two documents Phase 1
flagged as the corpus's stress cases turned out not to be this phase's stress
case at all — a genuine measurement-contradicts-expectation moment, logged
rather than quietly adjusted away.

**Next**
- Phase 3: embeddings, distance & the vector store. Embed `Chunk.embedding_text`
  (not `.text`) into Qdrant; bake off `bge-large-en-v1.5` vs `BGE-M3`; brute-force
  KNN as ground truth for the HNSW recall/latency curve. Open question #5 (is
  semantic chunking worth its cost) carries forward to Phase 6, not Phase 3.

### 2026-07-29 — Phase 1: Corpus acquisition & ingestion

**Context.** Same session as Phase 0, continued directly. Goal: turn Phase 0's
concrete, reproduced hallucination (missing `pattern=` content because raw
`.read_text()` never followed FastAPI's `{* *}` includes) into a fixed, tested
pipeline, plus acquire the closed-issue half of the corpus for the first time.

**Done**
- TDD'd a structure-preserving markdown parser (`src/aegis/ingest/markdown.py`):
  resolves `{* path *}` and `{!path!}`/`{!> path!}` includes (two syntaxes, only found
  the second by manually opening parsed output — automated leftover-syntax checks
  missed it entirely), flattens `///`/`////` admonitions and tabs including ones
  nested inside list items, strips decorative HTML wrappers
- Discovered (not assumed) that FastAPI's own plugin resolves includes against a
  fixed base directory (`docs/en/`) regardless of a file's nesting depth — confirmed
  by comparing identical include prefixes across files at different depths
- Built the docs loader: 129 pages (revised down from plan.md's ≥300 estimate after
  measuring the real corpus size — see Decision Log), stress-tested against the full
  corpus, 10 samples manually opened and verified
- Authenticated `gh` (needed a live debugging round — `gh auth status` kept failing
  from the tool's shell even after the user logged in; resolved on the user's end,
  cause not fully diagnosed)
- Built the GitHub issues loader: fetched all 3,530 closed issues (PRs excluded) +
  comments via the authenticated REST API, concurrent `httpx` with rate-limit backoff.
  Resolution-comment heuristic: most recent comment from an OWNER/MEMBER/COLLABORATOR
  author. 2,204/3,530 (62.4%) have one
- Found and fixed two systematic `product_area` classifier bugs via manual sampling,
  not automated checks: raw-text keyword matching fired on `async` in ~20% of issues
  from pasted code snippets, and on `openapi`/`pydantic` in 40% of issues from a
  literal old issue-template checklist line naming Pydantic/Swagger UI/ReDoc. Both
  fixed (code-block stripping + boilerplate-line stripping) with regression tests
- Hit GitHub's 5,000/hour rate limit re-running the full fetch a second time to apply
  the fix; recovered by reprocessing already-fetched raw data locally instead of
  re-fetching — see Decision Log, this also changed how hash-reproducibility got
  verified (pure-function check, not live-refetch diff)
- 10 issue documents manually sampled and opened; all structurally correct
- Wrote `data/processed/documents.jsonl`: 3,659 documents (129 docs + 3,530 issues)
- Design record: `docs/explainers/01-ingestion-and-parsing.md`

**Exit criteria — all met** (criterion revised from ≥300 to ≥125 docs, see Decision
Log; 129 docs / 3,530 issues actual)
- `documents.jsonl` with ≥125 docs and ≥2,000 issues
- All metadata fields populated (verified programmatically, zero missing)
- Re-running ingestion produces identical hashes (verified as a pure-function
  property of `content_hash`, not a live re-fetch — see Decision Log)
- Ten randomly sampled parsed docs manually verified, actually opened (done for both
  doc types — 20 total, not just the required 10)

**Surprised by:** the `{!path!}` include syntax that no automated check caught: found
only by reading actual output. Two rounds of boilerplate text (not real content)
dominating a keyword classifier — the issue-template checklist alone affected 40% of
the corpus. `release-notes.md` (686K chars) and issue #2266 (a 193K-char pasted build
log) are both legitimate long-tail content, not bugs, and will pressure-test Phase 2's
chunk-size choice directly.

**Next**
- Phase 2: chunking laboratory. `release-notes.md` and issue #2266 are concrete stress
  cases for whatever chunking strategies get compared. Four strategies (fixed,
  recursive, heading-aware, semantic), one interface, benchmarked against each other.

### 2026-07-29 — Phase 0: Foundations & the baseline that fails

**Context.** First implementation session. Kicked off Phase 0 per plan.md's "First
Three Actions." User opted to keep the project folder as `SourceDesk` rather than
renaming to `aegis` — the rename's rationale (space in path) doesn't apply to the
actual path, and renaming would've disrupted the already-open editor. The Python
package is still `aegis` (`src/aegis/`).

**Done**
- Installed `uv`; git-initialized the repo (was not previously version-controlled);
  built the full `src/aegis/` package skeleton per plan.md §3's repo layout
- `docker-compose.yml` with Qdrant, pinned to `v1.18.3` after hitting a client/server
  version-compatibility warning on the initially-used `v1.12.4`
- `config.py` (pydantic-settings, per-group `env_prefix`, reads `.env`) and `types.py`
  (`Chunk` with the full metadata schema plan.md specifies, not just what Phase 0 uses)
- `pytest` wired up, `Makefile` with the five entry points (`ingest`/`index`/`eval`/
  `serve`/`test` — only `test` is functional yet, the rest are stubs for later phases)
- Naive baseline (`src/aegis/baseline.py`, ~150 lines): shallow-cloned `fastapi/fastapi`,
  raw-text-loaded 85 tutorial/advanced doc pages (no markdown parsing), 500-char
  non-overlapping fixed chunks, `bge-small-en-v1.5` via `fastembed`, Qdrant dense-only
  top-3, Groq `llama-3.3-70b-versatile`, no citations, no abstention
- Ran it against 5 known-answer FastAPI questions, twice. Found and doc-verified 3
  confident hallucinations across two independent, unrelated mechanisms (fabricated
  code / wrong status code on a question with zero relevant retrieved content; a
  confidently-recommended *deprecated* API on a question where retrieval found the
  right file but fixed-size chunking severed the "(deprecated)" heading from its body)
- Design record: `docs/explainers/00-the-baseline-that-fails.md` — includes real
  token/cost math (full docs corpus is 2.84× over the 128K context window before Phase
  1 even adds the issue corpus; retrieval is ~300× cheaper than stuffing the reduced
  subset) sourced from an actual Groq pricing fetch and real file measurements, plus
  full transcripts in `docs/results/00-baseline-transcript.json`

**Exit criteria — all met**
- `make test` green (2 passed)
- Qdrant reachable from Python (`QdrantClient(...).get_collections()` succeeded)
- Baseline answers end-to-end (5/5 questions, both runs)
- ≥ 1 recorded confident hallucination (3 found, doc-verified against the actual
  FastAPI docs, not asserted from memory)

**Surprised by:** two unrelated failure mechanisms in the same 5-question run (missing
content from unresolved `{* ... *}` includes vs. a heading severed from its body at a
chunk boundary) — expected chunking to be a problem in the abstract, didn't expect two
distinct causal paths to show up this fast. Also: temp=0.0 on Groq isn't fully
deterministic run-to-run (logged as open question #4).

**Next**
- Phase 1: corpus acquisition & ingestion. Needs a real structure-preserving markdown
  parser that resolves FastAPI's `{* ... *}` includes and `///` admonition blocks
  (Phase 0 proved concretely what breaks if it doesn't), plus closed-issue collection
  via the GitHub API (target 2,000–3,000, verify the ≥500-with-resolution assumption —
  open question #2).

### 2026-07-28 — Plan revision (v2): rebalanced for hiring, not learning

**Context.** User clarified they already understand all topics in `RAG_Topics_Learned.md` — the goal is landing an industry role, not learning RAG from scratch. v1 defaulted to building nearly everything by hand, which was the wrong optimization target once that was known.

**Done**
- Ranked what actually differentiates a RAG portfolio project in interviews: evaluation rigor > calibrated abstention > production ops > public demo > framework fluency > from-scratch implementations
- Rewrote the build/import split (`plan.md` §1): import BM25 (`bm25s`), embeddings, HNSW, reranker, retrieval metrics (`ranx`), RAGAS, Langfuse, FastAPI. Keep building: custom analyzer, heading-aware chunker, RRF, confidence scorer, citation validator, routing policy, eval harness, semantic cache — the ~900 lines that are genuinely the project's own contribution
- Cut ColBERT to a design-record-only topic; cut the three-way encoder bake-off to two real candidates
- Added Phase 17: build the orchestration directly first, then port to LangGraph and measure the delta — gets the framework keyword plus a defensible, evidence-based opinion instead of a preference
- Freed ~1 day from the import swaps, spent it on the LangGraph phase; net schedule still 21 days, more buffer in Phases 5–7
- Updated `CLAUDE.md` — explainers reframed as engineering design records (decision-and-evidence focused, not concept tutorials), added the build-vs-import test as an explicit rule
- Updated `progress.md` (this entry) — renumbered phase board, added the Build vs Import Ledger to keep the stated ratio honest

**Next**
- Phase 0: rename folder to `aegis`, `uv init` on Python 3.12, package skeleton, Qdrant in Docker, then build the naive baseline and catch it hallucinating.
