# 06 — Retrieval Metrics & the Ablation Grid

**Phase:** 6 · **Date:** 2026-08-04 · **Files:** `src/aegis/eval/metrics.py`, `src/aegis/eval/harness.py`, `src/aegis/eval/doc_level.py`, `src/aegis/retrieval/sparse.py`, `src/aegis/retrieval/dense.py`, `src/aegis/retrieval/hybrid.py`, `src/aegis/chunking/base.py`, `src/aegis/embedding/cache.py`, `experiments/06a_prepare_ablation_indexes.py`
**Topics from the course:** Retrieval evaluation metrics (Precision, Recall, MAP@k, MRR); Evaluation strategies (component vs. system, code-based vs. LLM-judge vs. human feedback)

## 1. What I built

A metrics wrapper around `ranx` (verified once against a hand-computed example), a config-hashed, cached eval harness (`run_eval`), a document-level ground-truth-transfer module for comparing chunking strategies fairly, and the actual four-axis ablation grid — chunking, contextual enrichment, retrieval strategy, encoder — swept one axis at a time from a confirmed center, against the 150 positive queries in the golden set's dev split. Along the way, this phase found and fixed three real bugs in Phase 3/5 code that no earlier phase's tests caught, because none of them ran real golden-set queries through the full pipeline until now.

## 2. Why this approach

Phase 5 built the retrieval pipeline; this phase is what turns "it works" into "here's the number, and here's why this config." The eval harness is deliberately decoupled from any specific retriever (`search_fn` is a plain callback) so its own logic — per-stratum breakdown, negative exclusion, config-hash caching — is unit-tested with fakes, independent of live Qdrant/`bm25s`. That decoupling is also what let three real production bugs surface as clean, isolated, investigable signals instead of noise buried in one large end-to-end script.

The sweep follows plan.md's explicit instruction not to run the cartesian product: confirm each axis's winner, then use it as the center for the next axis, rather than sweeping all combinations independently.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ------------------ |
| Score the chunking axis at chunk level, same as every other axis | Consistency — one scoring method everywhere | Impossible by construction: ground truth is heading-aware chunk IDs, and fixed/recursive/semantic chunkers each embed their own strategy name in the chunk ID (`Chunk.doc_id` is the only field confirmed stable across all four). Chunk-level scoring would silently score every non-heading-aware strategy as ~0 regardless of actual quality. Resolved with document-level ground-truth transfer instead (§4), confirmed with the user before running. |
| Compare against Phase 0's literal `aegis_baseline` collection for the "lift from baseline" number | It's the real, previously-built naive baseline plan.md's Definition of Done seems to ask for | Checked the actual collection: it indexed 85 of 129 doc pages, raw 500-char slices, and **zero of the 3,530 issues** — the golden set is mostly issue-sourced. Scoring it would measure "wrong corpus," not "worse config," a confound rather than a fair comparison. Used the worst fully-specified config *within* the controlled ablation grid instead (same corpus, only retrieval config differs) — see `docs/results/retrieval-ablation.md` §7. |
| Keep the eval harness's cache keyed on `RetrievalConfig` alone | Simpler; config fully describes what was measured | Two different query sets evaluated under an identical config would silently collide on the same cache file and return each other's results — caught by a dedicated test (`test_run_eval_cache_is_scoped_to_the_evaluated_query_set`) before it ever shipped. Cache key is `config_hash + queries_hash` instead. |
| Parallelize the harness's per-query retrieval with threads, trusting existing components are thread-safe | `ThreadPoolExecutor` is the obvious tool for I/O-bound per-query retrieval | They weren't: `EmbeddingCache` used a single `sqlite3.Connection` with no thread-safety guarantee, and crashed immediately (`SQLite objects created in a thread can only be used in that same thread`) the first time the harness actually ran multi-threaded. Fixed with `check_same_thread=False` + a lock, TDD'd with a real concurrent-access test, not disabled. |

## 4. How it works

**`ranx` verification** (`eval/metrics.py`): a 5-document, one-query example with known relevant docs (`d1`, `d3`, `d5` at ranks 1, 3, 5) and hand-computed Precision@5, Recall@5, MRR, MAP@5, NDCG@5. `ranx.evaluate` matched every hand-computed value exactly. Found one real library quirk while verifying: `evaluate()` returns a bare `numpy.float64` for a single metric but a `dict` for multiple — normalized in the wrapper so no caller has to special-case it.

**The eval harness** (`eval/harness.py`): `RetrievalConfig` is a frozen dataclass; `config_hash()` hashes every field except the human-readable `name`. `run_eval` takes a `search_fn: Callable[[GoldenQuery], list[tuple[str, float]]]`, excludes negatives (they have no ground-truth chunk to retrieve — Phase 10's job), retrieves in parallel via `ThreadPoolExecutor`, scores overall and per-stratum via `ranx`, and caches to disk keyed by `config_hash + queries_hash`.

**Document-level ground-truth transfer** (`eval/doc_level.py`), for the chunking axis only:

```python
def doc_level_search_fn(search_fn, chunk_id_to_doc_id):
    def wrapped(query):
        doc_hits, seen = [], set()
        for chunk_id, score in search_fn(query):
            doc_id = chunk_id_to_doc_id.get(chunk_id)
            if doc_id is not None and doc_id not in seen:
                seen.add(doc_id)
                doc_hits.append((doc_id, score))
        return doc_hits
    return wrapped
```

Ground truth is transferred the same way: each query's heading-aware `ground_truth_chunk_ids` maps to the *set* of `doc_id`s those chunks belong to, via `chunk_ids_to_doc_ids`. A retrieved chunk from any strategy then counts as a hit if its document is in that set — coarser than exact match, but the only fair way to compare chunking strategies whose chunk IDs live in disjoint spaces.

**Self-match exclusion** (`retrieval/sparse.py`, `retrieval/dense.py`, `retrieval/hybrid.py`): every golden query has a `source_issue` — the real closed GitHub issue it was harvested from (Phase 4). That issue is still indexed in the corpus, and its text is near-verbatim the query text, so it's a trivial top-1 match in every retriever — a match ground truth deliberately excludes, wasting a top-10 slot in every measurement until this was caught (§6). Fixed with an `exclude_doc_id` parameter on every search method: native Qdrant `must_not` for `DenseIndex` (exact, no shortfall risk — same pre-filter-over-post-filter reasoning as Phase 3's `product_area`/`doc_type` filters), over-fetch-then-filter for `SparseIndex` (`bm25s` has no native per-query filter).

## 5. What the numbers said

Full tables in `docs/results/retrieval-ablation.md`. Headline:

- **Winning config:** heading-aware chunking, contextual enrichment off, dense retrieval, `bge-large-en-v1.5`. Recall@10 0.4288, MRR 0.8640, NDCG@10 0.7064, MAP@10 0.3695.
- **Retrieval strategy:** dense wins. Hybrid (RRF or linear), even weight-tuned, only ever *approaches* dense's performance as `dense_weight → ∞` — it never exceeds it, in aggregate or in any of the six strata.
- **Encoder:** `bge-large-en-v1.5` beats BGE-M3 by more than 3x (0.4288 vs. 0.1348 recall@10).
- **Contextual enrichment:** off wins by a wide margin (0.4288 vs. 0.1751) — the opposite of plan.md's "usually the cheapest large win" expectation on this corpus.
- **Chunking:** heading-aware wins by more than 2x over fixed, recursive, and semantic (all three cluster around 0.24-0.26 doc-level recall@10). Semantic chunking's real, now-measured cost (~723s to chunk + 640s to embed the full corpus) buys essentially nothing over recursive — resolves Open Question #5.
- **Lift vs. worst measured config (sparse alone):** Recall@10 +0.332 absolute (4.4x). MRR +0.496 absolute (2.35x).

## 6. What surprised me / what to watch for

**Three real bugs, three different mechanisms, all found by running real golden-set queries through the pipeline for the first time** — the same pattern as every prior phase's "measuring the full corpus catches what small unit tests can't," now extended to *retrieval quality* instead of just chunk/embedding correctness:

1. **`EmbeddingCache` thread-unsafety.** The harness's `ThreadPoolExecutor` immediately crashed the first real run. A single `sqlite3.Connection`, fine under every prior phase's single-threaded usage, was never exercised concurrently until now.
2. **`enrichment_prefix` title duplication.** `aegis.ingest.github_issues` (Phase 1) prepends the issue title as the document's own H1; `enrichment_prefix` (Phase 2) naively concatenated `document.title` with `heading_path` without accounting for the overlap, duplicating the title verbatim in every issue chunk's embedded prefix — 83% of the corpus. Only measurable because this phase, for the first time, ran contextual enrichment against real retrieval quality rather than just chunk-count/token-distribution stats (Phase 2's own measurement scope).
3. **The self-match leak (the biggest one).** Every golden query's own source issue is indexed in the corpus and is a trivial near-verbatim match — Phase 4 already discovered and fixed this exact problem for golden-set *labeling* (`must_not` in `retrieve_candidates.py`), but I didn't carry the same exclusion into Phase 6's actual *evaluation* methodology. It silently inflated wasted top-10 slots across every single measurement in this phase's first pass — retrieval-strategy axis, encoder axis, weight tuning, per-stratum, enrichment axis — before being caught by manually inspecting one query's actual top-5 results and recognizing the #1 hit as the query's own source text. MRR was the most visibly distorted metric (dense: 0.616 → 0.864 once fixed) since the self-match was usually rank 1, pushing the first *real* hit down by at least one position everywhere.

**Even after fixing the enrichment bug, enrichment still hurt badly** (0.1454 → still only 0.1751 with self-match also excluded, vs. dense's 0.4288). The dominant cause turned out to be a corpus-specific property, not a code bug: GitHub issue templates share generic section headings (`Description`, `Question`, `How to do it`) across thousands of unrelated issues. Prepending them is exactly the boilerplate-pollution failure mode Phases 1-4 each independently hit, now showing up inside contextual enrichment specifically — a technique plan.md expected to be close to free money. Confirmed by inspecting the actual embedding text of both a correct-but-unretrieved chunk and the incorrectly-top-ranked ones for one real query, not inferred from the aggregate number alone.

**A cache-staleness gap, found and worked around, not yet fixed.** After fixing the enrichment bug and re-embedding, re-running the same `RetrievalConfig` silently returned the *stale* (buggy) cached result — the harness's cache key captures declared config fields, not "the underlying Qdrant collection was regenerated." Worked around by manually deleting the affected cache file for this phase; not fixed generally, since the harness's caching model assumes a stable config→data mapping that this phase's iterative index-rebuilding broke as a one-off, not a case the harness needs to handle going forward (each phase builds its indexes once).

## 7. Where this fits and what's next

The winning config (heading-aware, off, dense, bge-large) is now the corpus's evaluated default heading into Phase 7 (reranking measures its lift *over* this config, not from scratch). The `exclude_doc_id` self-match exclusion and the eval harness itself are now load-bearing for every remaining eval-dependent phase (7, 10, 11, 16, 17) — any of them that builds a `search_fn` against the golden set needs to apply it too, or will silently reproduce this phase's first-pass bug. The sparse-side `doc_type` filtering gap flagged in Phase 5's design record remains open and untested against real numbers; worth a follow-up ablation cell if lexical retrieval's per-stratum weakness (§5 of the results doc) turns out to matter once reranking is in place.
