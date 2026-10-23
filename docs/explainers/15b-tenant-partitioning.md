# 15b — Tenant Separation at the Retrieval Layer

**Phase:** 15 · **Date:** 2026-10-18 · **Files:** `src/aegis/tenancy.py`, `src/aegis/config.py` (`TenancySettings`), `src/aegis/retrieval/{sparse,dense,hybrid}.py`, `src/aegis/cache/{exact_match,semantic}.py`, `src/aegis/pipeline.py`, `src/aegis/factory.py`, `src/aegis/api/app.py`, `tests/test_tenancy.py`, `tests/security/test_attack_tenant_bypass.py`, `tests/security/test_attack_retrieval_partition.py`, `experiments/15b_partition_bypass.py`, raw `docs/results/partition-bypass-raw.json`
**Topics from the course:** Security — tenant separation, knowledge-base leakage (Module 5); metadata pre-filtering vs post-filtering (Phase 3)

## 1. What I built

A partition is a named set of `doc_type`s (`docs`, `issues`, `full`; `TenancySettings`). A pipeline is **bound to one partition when it's built**, and the partition enforces itself in three places: the dense search (a Qdrant pre-filter), the sparse search (a bm25s weight mask plus a post-filter), and the cache key. A fail-closed check sits on top. No request field, query text or history can choose or widen the partition.

## 2. Why this approach

**The plan's requirement is "metadata filters at the retrieval layer, not the prompt."** The reason is the one Phase 8 gave for billing: a prompt can be argued out of things, a code path can't. A filter written into the prompt ("only use documentation") is an instruction that the retrieved content itself can argue with.

**The stand-in (user decision, 2026-10-18).** This corpus has no tenants, so I used `doc_type`:

- 2,292 documentation chunks and 16,944 issue chunks. They're disjoint, with heavy vocabulary overlap, so a bypass attempt that uses one partition's words to pull in the other is meaningful.
- "Docs-only" is also a real deployment shape: it removes the user-generated content that indirect injection needs.

**Why bind at construction instead of passing a partition per call.** If `search()` took a partition, every caller would be a place to get it wrong, and the A1 bug was exactly that kind of mistake: a filter argument that one code path forgot to forward. With the partition fixed when the pipeline is built, `PartitionedRetriever.search(query, top_k, exclude_doc_id)` has **no filter parameter at all**. Passing `doc_types=…`, `doc_type=…` or `partition=…` is a `TypeError` (tested). In 15c, the authenticated principal will select which pipeline handles the request.

**Two layers, because the first one had a bug.** Pushing the filter down to each index keeps recall inside the partition. The A1 bug shows that one index forgetting the filter is a realistic failure, so the second layer checks every returned chunk and fails the request instead of leaking it. The guarantee then rests on about five lines in `tenancy.py`, not on every index being correct.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
|---|---|---|
| Post-filter only (retrieve unfiltered, drop out-of-partition chunks) | One check, no index changes | Phase 3 measured post-filtering losing 60–83% recall on this corpus (`prefilter-vs-postfilter.md`). Here, only 2,292/19,236 chunks are documentation, so a docs caller would get short or empty lists. It's kept as the *backstop* only. The bypass sweep (§5) recorded 0 short result lists, so the pre-filter is doing the work. |
| Partition as a per-request `search()` argument | Simpler wiring; one pipeline | It's the A1 failure shape (a filter argument some path doesn't forward), and it makes the partition something a request handler can set from request content. |
| Separate Qdrant collections and separate BM25 indexes per partition | Physical isolation; no shared index to filter | Isolation would hold even through a Qdrant filter bug. But it means three copies of the 19k-chunk collection (`full` overlaps both), a re-embed or copy per partition, and delta indexing (Phase 12) maintaining all of them. With one collection plus the fail-closed check, a Qdrant filter bug becomes a failed request rather than a leak, at no storage cost. If real tenants arrived with legal isolation requirements, this would be the right call. |
| Filter in the prompt ("answer only from documentation") | No retrieval changes | The plan rules it out explicitly, and design record 15's A3/A4 show retrieved text can forge context and pass validation. |
| Silently ignore unknown body fields (pydantic's default) | Lenient API | `{"partition": "issues"}` would return 200 and look honored. Forbidding extras makes it a 422, so nobody can believe a body field chose the partition. |

## 4. How it works

```python
class PartitionedRetriever:
    def search(self, query, top_k=10, exclude_doc_id=None):          # no filter parameter exists
        results = self.base.search(query, top_k=top_k, doc_types=self.partition.doc_types, ...)  # push down
        leaked = [c.id for c, _ in results if c.doc_type not in self.partition.doc_types]
        if leaked:                                                    # fail closed
            log_event(..., "tenancy.partition_violation", ...)
            raise PartitionViolation(...)
        return results
```

**Each layer:**

| Layer | Mechanism |
|---|---|
| Dense (`DenseIndex`) | `doc_types` → Qdrant `FieldCondition(key="doc_type", match=MatchAny(any=[...]))` in `must`, a native pre-filter on the existing payload index. No re-index needed. |
| Sparse (`SparseIndex`) | New `doc_types` / `doc_type` / `product_area` filters. Before, it had none, which is the A1 root cause. A `float32` weight mask (cached per filter) is passed to `bm25s.retrieve(weight_mask=…)`, so out-of-filter documents score 0 *before* top-k. Masked documents can still come back at score 0 when fewer than k documents match, so an exact post-filter follows; that post-filter is the actual guarantee. |
| Hybrid (`HybridRetriever`) | Builds one `filters` dict and passes it to every search it calls, for every strategy (test parametrized over all 4). |
| Caches | `partition` is a required keyword-only argument on `get`/`set` for both caches. Scope = `"{corpus_version}::partition={name}"`. Exact cache: hashed into the key. Semantic cache: stored in the existing `corpus_version` column, so no migration; old rows just never match again. |
| Pipeline | `AnswerPipeline(partition=...)` scopes both caches. If the retriever is a `PartitionedRetriever` bound to a different partition, construction raises `ValueError`, so the cache and retrieval layers can't disagree. |
| API | `ChatRequest`, `Turn`, `FeedbackRequest` use `extra="forbid"`. |
| Factory | `build_pipeline(partition=settings.tenancy.default_partition)`. The default `"full"` keeps served behavior identical to Phase 14 until 15c maps principals to partitions. |

## 5. What the numbers said

`uv run pytest -q`: **612 passed, 8 xfailed** (was 586 passed + 11 xfailed). That's +20 in `tests/test_tenancy.py`, +7 in `test_attack_tenant_bypass.py`, and the 3 partition attacks from record 15 flipped from xfail to pass (A1a, A1b, A2b).

**Before → after, same protocol as `15a`** (180 dev queries, hybrid RRF, `doc_type=documentation`, top-10, real Qdrant + real BM25):

| | Before (15a) | After (15b) |
|---|---|---|
| Issue chunks in a "documentation only" top-10 | **609 / 1,800 (33.8%)** | **0 / 1,800** |

**Bypass sweep** (`experiments/15b_partition_bypass.py`, 27.4 s). Each run is 190 queries: 10 bypass-shaped plus all 180 dev queries. The bypass-shaped ones are verbatim out-of-partition chunk text, chunk IDs, `doc_type: issue`, a JSON filter, "partition=full ignore the filter…", a fake `SYSTEM:` override and a `site:` query.

| Partition / retriever | Out-of-partition chunks | Violations raised | Short (< 10) lists |
|---|---|---|---|
| docs / dense (served path) | 0 | 0 | 0 |
| docs / hybrid | 0 | 0 | 0 |
| issues / dense | 0 | 0 | 0 |
| issues / hybrid | 0 | 0 | 0 |

**The same check on the real BM25 index inside the test suite:** 190 queries × 2 partitions, all 19,236 chunks, 0 leaks, 3,800 results returned (10 per query).

**Mutation check (does the guarantee survive a broken index?).** I reintroduced the A1 bug by making `SparseIndex.search` drop `doc_types`, then searched the `docs` partition with verbatim issue text. Result: `PartitionViolation: 3 chunk(s) outside partition 'docs'`, plus a `tenancy.partition_violation` log line. Nothing leaked.

**Request-body bypass:** `{"partition": "issues"}`, `{"doc_type": "issue"}`, `{"filters": {...}}` and a `partition` field inside a history turn all return 422 before the pipeline runs. That's 4 tests, plus a real-server check.

**Real server:** it starts with the partitioned factory. Warm-up retrieval went through the wrapper (11.9 ms), the encoder warm-up was still delegated (8,718 ms model load), `/health` returned 200, and a `partition` body got 422. No `/chat` answer was generated (no quota today).

## 6. What surprised me / what to watch for

- **What "provably impossible" means here, and what it doesn't.** Proven: no input to `/chat` (query text, history, unknown body fields, cache hits) can return a chunk outside the pipeline's partition. The partition isn't an input, and every result is checked against it before it leaves the retrieval layer. **Not proven:**
  - isolation inside Qdrant itself (one shared collection; a bug there becomes a failed request, not a leak, but it would still be an outage);
  - side channels, such as cache timing (a hit is ~5 ms vs a miss in seconds; with partition-scoped keys a hit only reveals that the *same partition* asked before);
  - anything about a caller who can reach Qdrant directly. Design record 15's A6 still stands: Qdrant is open to the LAN with full text in the payload, so the partition means nothing to someone who skips the API. That's 15f.
  - behavior at real tenant counts (2 disjoint partitions).
- **Changing the cache key invalidated every existing exact-cache entry.** The real `data/cache/exact_match.sqlite` rows are now unreachable. This costs nothing in correctness and is the same kind of event as a `corpus_version` bump, but the next real runs pay misses, and quota is tight.
- **`extra="forbid"` is a behavior change for any client sending extra fields.** The static UI is unaffected: `app.js` sends `{query, history}` with `{role, text}` turns and `{trace_id, rating, comment}` for feedback (read from source, lines 81, 154, 178; not re-tested in a browser). Third-party clients sending extra fields would now get a 422.
- **Two experiment scripts changed signature only** (`12b`, `14c`: `partition="full"` added to cache calls). Their published numbers are unaffected: `full` is the single scope they always ran in.
- **Partitions don't make the system safer against injection yet.** The `docs` partition removes the user-generated content, but the server is still built for `full`. Deciding who gets `docs` is 15c's job (principals → partitions).

## 7. Where this fits and what's next

This closes the "tenant bypass provably impossible" exit criterion at the API and retrieval layers. 15f has to close direct Qdrant access for the claim to hold for the deployment, not just the API.

Next is **15c**: API-key principals mapped to a partition-bound pipeline, trace and feedback ownership, rate limits on `/feedback` and `/trace`, a concurrency cap so `/trace` can't starve workers, and escaping the review-queue terminal output. Those are the A5a–e and A5g xfails.
