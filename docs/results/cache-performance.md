# Cache Performance — Phase 12 Summary

Ties together Phase 12's caching and delta-indexing units (1-4). Quantization (Unit 5) and the
cost-per-query model (Unit 6) have their own dedicated docs, linked below rather than duplicated here.

## What was built

| Unit | Component | Real result |
|---|---|---|
| 1 | Exact-match cache (`src/aegis/cache/exact_match.py`) | Normalized-query hash + `corpus_version` scoping. 10 tests, TDD'd. |
| 2 | Semantic cache (`src/aegis/cache/semantic.py`) | Threshold tuned on real data: 0.88, the lowest threshold with **zero false positives** on the 15 real hardest-negative pairs, catching 9/15 (60%) real paraphrases. See `docs/results/semantic-cache-threshold.md`. |
| 3 | Hit-rate simulation (`experiments/12b_hit_rate_simulation.py`) | Real Zipfian-traffic simulation over the 150 real golden intents: hit rate saturates near 98% once traffic volume exceeds ~20-30 requests/intent, but starts as low as 56.7% cold (1 request/intent) depending on skew. See `docs/results/cache-hit-rate-simulation.md`. |
| 4 | Delta indexing (`src/aegis/ingest/delta.py`) | Real content-hash diffing + a real Qdrant integration test proving a deleted document's chunks become unretrievable. Found and fixed a real latent point-ID bug along the way. See design record 12d. |

## The two required regression tests

- **Deleted document becomes unretrievable** (the Air Canada failure mode as an automated test):
  `tests/test_delta_indexing.py::test_deleted_document_becomes_unretrievable`. Real Qdrant collection,
  asserts both the point count drops and a rebuilt `DenseIndex` no longer surfaces the deleted document,
  while the surviving document stays retrievable.
- **Semantic cache doesn't serve near-misses**:
  `tests/test_semantic_cache.py` — the real hardest near-miss pair found in Unit 2's threshold sweep
  (`golden::issue::1320` vs. `332`, real similarity 0.8694) correctly does not hit at the shipped 0.88
  threshold.

## Real, load-bearing findings from this phase

1. **Traffic volume matters more than traffic shape, once the intent pool is small relative to request
   volume** (Unit 3). This golden set's 150 unique intents saturate to ~98% hit rate regardless of Zipf
   skew once N/intent exceeds ~20-30 — a property of this specific evaluation set's intent diversity, not
   a general claim about this system's real-world hit rate against a much larger real question space.
2. **A real, previously-latent indexing bug was found and fixed before delta indexing could work safely**
   (Unit 4): Qdrant point IDs were assigned by list position, silently correct only because every prior
   real use case did a full wipe-and-rebuild. Fixed with a deterministic `chunk_point_id()`
   (`uuid.uuid5`) before any incremental upsert/delete was attempted.
3. **int8 quantization is nearly free; binary is not** (Unit 5, `docs/results/quantization-tradeoff.md`) —
   and the same investigation surfaced a bigger, disclosed finding: this golden set's ground truth was
   labeled from candidates the live Qdrant index itself surfaced, so recall against it partly measures
   agreement with that specific labeling mechanism, not embedding quality alone.
4. **Escalation is the cheapest non-cache-hit path, in both latency and tokens** (Unit 6,
   `docs/results/cost-per-query.md`) — roughly 3x faster than generating an answer, and the only
   non-cache-hit configuration with zero LLM token cost, since it never reaches the generation call.

## What's still open (disclosed, not silently dropped)

- The live production Qdrant collections (`aegis_chunks_bge_large`/`_small`/`_m3`) still carry the old
  positional point IDs from their last full rebuild — harmless as-is, but a first real delta operation
  against any of them needs one full re-upsert under the new stable-ID scheme first (design record 12d §7).
- `corpus_version` is still a caller-supplied placeholder at the cache call sites — `IndexManifest.
  corpus_version()` (Unit 4) is the real mechanism, not yet wired in.
- Escalation's cost (Unit 6) is a real partial lower bound: `escalate_fn` itself (whatever a real
  escalation handler does — queueing, notifying an agent, rendering the UI) doesn't exist yet anywhere in
  this codebase; that's Phase 14's job.

## Design records

`docs/explainers/12-exact-match-cache.md`, `12b-semantic-cache.md`, `12c-hit-rate-simulation.md`,
`12d-delta-indexing.md`, `12e-quantization-tradeoff.md`, `12f-cost-per-query-model.md`.
