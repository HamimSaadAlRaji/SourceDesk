# 12 — Exact-Match Response Cache (Unit 1)

**Phase:** 12 · **Date:** 2026-09-26 · **Files:** `src/aegis/cache/exact_match.py`,
`src/aegis/config.py` (`CacheSettings`), `tests/test_exact_match_cache.py`
**Topics from the course:** Cost vs response quality (topic 34) — this unit is the cheapest possible way to
avoid paying for a full retrieval+generation call at all

## 1. What I built

`ExactMatchCache` — a sqlite-backed cache mapping a normalized query hash, scoped to a `corpus_version`
string, to a `CachedAnswer` (text, cited chunk ids, provider). `.get(query_text, corpus_version)` returns a
hit or `None`; `.set(...)` stores one. `normalize_query` lowercases and collapses whitespace before hashing
so trivial phrasing differences (case, extra spaces) still hit.

## 2. Why this approach

Support traffic is repetitive — the same handful of questions ("how do I upload a file", "how do I add a
custom header") recur constantly, and plan.md's own framing (step 3, "support traffic is Zipfian, not
uniform") is exactly why this is worth building before the fancier semantic cache: an exact hash lookup is
close to free (no embedding call, no similarity search) and catches the highest-frequency case cleanly, with
zero risk of the semantic cache's near-miss failure mode (serving a wrong answer to a similar-but-different
question).

Caching a purpose-built `CachedAnswer` rather than the full `GenerationResult` (`generation/pipeline.py`) is
a deliberate scope cut: `GenerationResult.context` is a `PackedContext` carrying full `Chunk` objects (heading
paths, full embedding text, doc versions). That data is already durably stored in the corpus/vector store —
duplicating it into every cache row would bloat the cache for a document set this project already indexes
elsewhere, and would couple the cache's on-disk schema to `PackedContext`'s internal shape, which isn't this
cache's concern.

`corpus_version` is accepted as an opaque string parameter, not computed here. That's a deliberate
decoupling (same DI pattern as the rest of this project, e.g. `EmbeddingCache` doesn't know how a content
hash is computed either) — deciding *what* corpus_version actually is (a chunk-set hash, a monotonic build
counter, something else) is naturally Unit 4's job (delta indexing), since that's the component that knows
when the corpus has actually changed. This cache only needs the invariant that two different real corpus
states produce two different version strings.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ----------------- |
| Cache the full `GenerationResult` (including `PackedContext`'s chunks) | Simpler — reuse the existing type directly, no new schema | Duplicates already-stored chunk text/metadata into every cache row for no serving benefit, and couples the cache to a pipeline-internal type that can change shape independently of caching concerns. |
| An in-process dict (no persistence) | Simpler, no sqlite dependency | Wouldn't survive a process restart — a real production cache needs to persist across deploys, same reasoning `EmbeddingCache` (Phase 3) already established for this exact tradeoff on a different cache. |
| Compute `corpus_version` inside this module (e.g., hash `chunks.jsonl` on every cache access) | One fewer parameter to thread through callers | Re-hashing the full processed corpus file on every single cache lookup would cost more than the retrieval this cache exists to avoid — the version needs to be computed once, at index-build time, and passed in, which is exactly what Unit 4 (delta indexing) will naturally produce as a byproduct of tracking what changed. |

## 4. How it works

```python
def _cache_key(query_text: str, corpus_version: str) -> str:
    normalized = normalize_query(query_text)
    return hashlib.sha256(f"{corpus_version}::{normalized}".encode()).hexdigest()
```

The corpus version is folded directly into the hash input rather than used as a second lookup column —
simpler than a composite sqlite key, and means a version bump automatically orphans every old row's key
(they just stop being reachable) rather than needing an explicit purge step for this cache specifically.
Same sqlite pattern as `embedding/cache.py` (Phase 3): `check_same_thread=False` plus a `threading.Lock`
around every access, since this cache will sit on the same request-serving path a production deployment
would parallelize across requests.

## 5. What surprised me / what to watch for

Orphaned rows from old corpus versions are never deleted, only made unreachable (a different hash means the
row is dead weight, not garbage-collected) — fine for now given this project's corpus reindexes
infrequently and at modest scale, but a real, disclosed limitation if this cache ran in a genuinely
long-lived production deployment across many reindexes. Worth a cleanup pass (delete rows whose corpus
version doesn't match the current one) if that ever matters — not built speculatively here since nothing in
this project's scale exercises it yet.

## 6. Where this fits and what's next

Sits at the very front of the pipeline per plan.md's own architecture diagram (§3) — checked before the
semantic cache, before the router. Unit 2 (semantic cache) is the harder, real-tradeoff-laden sibling: unlike
this exact-match lookup, it has a genuine new failure mode (serving a similar-but-wrong answer) that plan.md
explicitly asks to be tested against directly, not just built.
