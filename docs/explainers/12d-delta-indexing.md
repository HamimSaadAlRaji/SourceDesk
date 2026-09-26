# 12d — Delta Indexing

**Phase:** 12 · **Date:** 2026-09-26 · **Files:** `src/aegis/ingest/delta.py`, `src/aegis/embedding/vector_store.py`, `tests/test_delta_indexing.py`
**Topics from the course:** Cost vs response quality (topic 34) — re-embedding every chunk on every content change is the cost this unit avoids; What makes production challenging (topic 29) — a deleted or superseded document remaining retrievable is exactly the silent failure mode (Air Canada) this project is built around catching.

## 1. What I built

`aegis/ingest/delta.py`: `IndexManifest` (which `chunk_id`s are indexed and their `content_hash`, persisted and diffable), `diff()` (added/changed/deleted/unchanged against a previous manifest), `corpus_version()` (a real hash over the manifest, replacing the placeholder string Units 1-3's caches were handed), and `apply_delta()` (embeds only what changed, deletes what's gone, touches nothing else). `vector_store.py` gained the stable point-ID scheme (`chunk_point_id`) and non-destructive `ensure_collection`/`delete_chunks` primitives this required. `tests/test_delta_indexing.py` proves it end-to-end against a real (throwaway) Qdrant collection: a deleted document's chunks are gone from the index and unretrievable.

## 2. Why this approach

Plan.md's own framing already named the mechanism (content-hash diffing) and the required proof (a real "deleted doc is unretrievable" test) — the actual work was discovering that the existing indexing code couldn't support either safely yet, and fixing that first.

`embedding/run.py`'s `embed_and_index` has always called `create_collection`, which calls Qdrant's `recreate_collection` — a full wipe — every single time, and `upsert_chunks` assigned each point's ID from its **position in the chunks list** (`id=i`), not anything derived from the chunk itself. That combination was internally consistent as long as every index operation was a full rebuild: positions were freshly reassigned every time, so they never went stale. The moment any operation stops being a full rebuild — which is the entire premise of delta indexing — position-based IDs break silently: deleting or reordering one chunk shifts every later chunk's position, and therefore its Qdrant point ID, so a later "upsert this one changed chunk" or "delete this one removed chunk" would silently target the wrong point. Verified this is a real, current constraint, not a hypothetical one: a direct probe against the live local Qdrant server confirms it rejects an arbitrary string ID like a real chunk_id (`doc::release-notes.md::heading_aware::961`) outright — point IDs must be an unsigned int or a UUID. `chunk_point_id()` (`uuid.uuid5` over the real chunk_id) gives a deterministic, stable, valid ID, so upserting or deleting the same logical chunk always targets the same point, however many times, in whatever order, regardless of what else changed around it.

`corpus_version()` being a hash over the manifest rather than a version label someone has to remember to bump is deliberate: Unit 1's own design record already named the failure mode this defends against ("a reindex automatically orphans stale entries rather than needing an explicit purge"). A hash makes that automatic in the strict sense — change, add, or remove one chunk anywhere and the version changes, with no step anyone can forget.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ----------------- |
| Keep positional point IDs, just skip `recreate_collection` for deltas | Smaller change, no new ID scheme | Directly unsafe: confirmed live against Qdrant that this scheme can silently target the wrong point the moment a rebuild isn't full — the exact bug this unit exists to prevent, just moved one layer down |
| A manually incremented `corpus_version` (e.g., a build number in config) | Simpler, human-readable | Requires someone to remember to bump it on every reindex; a hash over the actual manifest can't be forgotten because it isn't a separate step — it's a pure function of what's actually indexed |
| Re-embed and re-upload the real production collections (`aegis_chunks_bge_large`, etc.) with the new stable-ID scheme now | Every real collection would already be delta-ready | Re-embedding ~19k chunks with `bge-large` is real, measured minutes-to-hours-scale work (`docs/results/rerank-tradeoff.md`'s CPU-inference numbers give a sense of the per-item cost), disproportionate to what this unit needs to prove; nothing currently depends on incremental delta support against the live collections, and Phase 6/7's results are unaffected either way since they were produced under the old full-rebuild-every-time semantics. Left as real, explicit future work (§7), not silently skipped |

## 4. How it works

```python
class IndexManifest(BaseModel):
    entries: dict[str, str]  # chunk_id -> content_hash

    def corpus_version(self) -> str:
        canonical = json.dumps(sorted(self.entries.items()), separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def diff(old: IndexManifest, new: IndexManifest) -> DeltaPlan:
    old_ids, new_ids = set(old.entries), set(new.entries)
    common = old_ids & new_ids
    return DeltaPlan(
        added=sorted(new_ids - old_ids),
        deleted=sorted(old_ids - new_ids),
        changed=sorted(cid for cid in common if old.entries[cid] != new.entries[cid]),
        unchanged=sorted(cid for cid in common if old.entries[cid] == new.entries[cid]),
    )
```

`apply_delta` only ever touches `plan.to_embed` (added + changed) and `plan.deleted` — `unchanged` chunks are never re-embedded or re-uploaded:

```python
def apply_delta(client, collection_name, plan, new_chunks_by_id, encoder, dim):
    ensure_collection(client, collection_name, dim=dim)  # create-if-missing, never wipes
    if plan.to_embed:
        chunks = [new_chunks_by_id[cid] for cid in plan.to_embed]
        vectors = encoder.encode([c.embedding_text for c in chunks])
        upsert_chunks(client, collection_name, chunks, vectors)  # stable IDs now
    delete_chunks(client, collection_name, plan.deleted)
```

Concrete example from the real test: doc-a (2 chunks) and doc-b (2 chunks) are indexed (4 points). Doc-a is then removed from the real corpus. `diff(full_manifest, doc_b_only_manifest)` returns `deleted=["doc-a::1", "doc-a::2"]`, `to_embed=[]` — `apply_delta` issues exactly one Qdrant delete call for those 2 point IDs and no embedding calls at all.

## 5. What the numbers said

Real, `uv run pytest -q tests/test_delta_indexing.py` (against the live local Qdrant instance, throwaway collection): 15/15 passed, including the literal regression test:

- Before deletion: `qdrant_client.count(...) == 4`; a real query near doc-a's embedding region returns both of doc-a's chunks.
- After `apply_delta` on the deletion plan: `qdrant_client.count(...) == 2` (checked directly against Qdrant, not just through the wrapper) — doc-a's points are actually gone, not just hidden.
- The same query, run again through a real `DenseIndex` built from the post-deletion chunk list, returns zero doc-a chunks; doc-b's 2 chunks are still both retrievable, confirming the delta didn't collaterally damage anything else.

Full suite: `uv run pytest -q`: 455/455 passed (440 prior + 15 new).

## 6. What surprised me / what to watch for

**The real bug wasn't in the diffing logic — it was one layer down, in an assumption `embedding/run.py` had been quietly relying on since Phase 3.** Position-based point IDs never once caused a visible problem, because every real use of `create_collection`/`upsert_chunks` so far has been a full wipe-and-rebuild. It would have caused silent, hard-to-trace corruption (upserting or deleting the wrong point) the first time anyone tried a real incremental operation against a live collection — which is precisely what this unit needed to do. Confirmed the failure mode directly (a live probe against Qdrant rejecting an arbitrary chunk_id string as a point ID) rather than reasoning about it abstractly, per CLAUDE.md's verification discipline.

**`DenseIndex`'s in-memory `_chunks_by_id` safety net (`retrieval/dense.py`) can mask a real Qdrant deletion bug if a test only checks through it.** If `delete_chunks` silently failed to delete anything, but a test only rebuilt `DenseIndex` from the smaller, correct chunk list, the search results would look correct anyway — the safety net drops any Qdrant hit whose ID it doesn't recognize, which is exactly what an *undeleted* stale point would look like too, from that vantage point. The test asserts the real Qdrant point count directly (`client.count`) in addition to the search-level behavior, specifically so a broken `delete_chunks` couldn't pass by accident.

## 7. Where this fits and what's next

Closes plan.md's Phase 12 step 4 exit criteria (content-hash diffing, purge on delete, the deleted-doc regression test) and gives Units 1-3's caches a real `corpus_version` instead of a placeholder — wiring that in (computing and passing a real `IndexManifest.corpus_version()` at the call sites that currently pass a literal string) is real, but small, follow-up work, not yet done. Explicitly not done in this unit, and worth flagging before Unit 6's cost model: the live production collections (`aegis_chunks_bge_large`/`_small`/`_m3`) still carry the old positional point IDs from their last full rebuild — they work fine as-is (nothing reads point IDs directly), but a first real delta operation against any of them would need one full re-upsert under the new stable-ID scheme before incremental deltas could target them safely. Units 5 (quantization) and 6 (cost model) are next; neither depends on this gap being closed first.
