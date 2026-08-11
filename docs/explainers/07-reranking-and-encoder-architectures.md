# 07 — Reranking & Encoder Architectures

**Phase:** 7 · **Date:** 2026-08-11 · **Files:** `src/aegis/config.py`, `src/aegis/rerank/cross_encoder.py`, `src/aegis/rerank/pipeline.py`, `experiments/07_rerank_sweep.py`, `experiments/07a_export_rerank_candidates.py`, `experiments/07b_kaggle_rerank_sweep.py`, `docs/results/rerank-tradeoff.md`
**Topics from the course:** Bi-encoders, cross-encoders, and ColBERT; Reranking; Transformer architecture (attention, feed-forward, encoder/decoder)

## 1. What I built

A cross-encoder reranker (`bge-reranker-v2-m3`) that rescores a pool of dense-retrieval candidates by attending over `(query, chunk_text)` jointly, wired behind a toggleable pipeline (`RerankedRetriever`) so it can be A/B'd against Phase 6's winning dense-only config by flipping one flag. Swept candidate pool sizes 10/20/50/100 against the 150-query golden dev set, measuring NDCG@5 and reranking latency at each. The measured result: reranking makes retrieval quality **worse**, monotonically, at every pool size — and every pool size already exceeds the full-pipeline latency budget on its own. Reranking is **off** by default in this system's config.

## 2. Why this approach

`CrossEncoderReranker` takes the same shape as `aegis.embedding.encoder.Encoder`: a thin class around an injectable `score_fn`, so the pure reorder/truncate logic is unit-tested with a fake and the real 568M-parameter model is only ever exercised by a real measurement script, never by the test suite. `RerankedRetriever` mirrors `HybridRetriever`'s strategy-switch shape (Phase 5) rather than adding a `rerank` parameter directly onto `HybridRetriever.search()` — composition over modification, so Phase 5's already-tested retrieval logic stays untouched and reranking is a separate, optional stage any caller can attach or skip.

The bigger "why" for this phase is what happened while measuring it. This machine (M1 Pro, 16GB) could not run the sweep: MPS hit macOS's "stuck" scheduler state twice under real memory pressure, and CPU inference, while stable, projected ~5.7 hours for the full sweep once benchmarked against real corpus-length text rather than short placeholder text. Given CLAUDE.md's rule that a claim needs a real number, not a rationalized skip, the sweep moved to a free Kaggle T4 GPU rather than being abandoned or approximated. That's a real engineering call under a real constraint, documented here rather than smoothed over.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ------------------ |
| Implement ColBERT for real, not just as a design-record topic | A third, real architecture point on the bi-encoder/cross-encoder spectrum | plan.md's explicit cut list (§7): ~0.5 day for a result reasonable to predict without building it. The bi-encoder-vs-cross-encoder comparison already measured here demonstrates the core lesson (independent encoding vs. joint attention); ColBERT's late-interaction mechanism is explained in §4 without needing a second index built and maintained for it. |
| Keep sweeping on local MPS after the first hang, assuming it was a one-off | Simpler, no new infrastructure, matches the project's "local-first" pattern elsewhere | Measured twice, not once: freeing the bi-encoder before loading the reranker (the fix for attempt 1) didn't prevent attempt 2 from also hanging under this session's broader system memory pressure. Two real, reproduced failures is evidence, not bad luck. |
| Ship reranking enabled at pool=10 as a "least bad" compromise | Smallest quality loss of the sizes tested; keeps the feature "on" for the portfolio narrative | Every axis says no: pool=10 still regresses NDCG@5 by 9.0% and its own p95 latency (5.2s) alone exceeds the entire 3s pipeline budget. Shipping a component that both makes answers worse and blows the latency budget to preserve optics would be exactly the "tidy positive" CLAUDE.md warns against over an honest negative. |
| Score `chunk.embedding_text` (the contextual-enrichment-prefixed text) for reranking, matching what the bi-encoder indexed | Consistency with what's actually stored per chunk | `embedding_text` only differs from `text` when contextual enrichment is on, and Phase 6 already measured enrichment as a net negative on this corpus (GitHub issue-template boilerplate). A cross-encoder also doesn't need enrichment's compensation for lost context the way a bi-encoder does — it already attends over the real passage jointly with the query. Scored `chunk.text`, the real source text, matching what a citation would show a reader. |

## 4. How it works

**Bi-encoder vs. cross-encoder vs. ColBERT — the attention pattern behind the difference.** All three answer "how relevant is this passage to this query," but differ in *when* the query and passage tokens are allowed to attend to each other:

- **Bi-encoder** (this project's `bge-large-en-v1.5`, Phase 3): query and passage are each encoded in **separate** forward passes, with no cross-attention between them — a passage's embedding is computed once, offline, without knowing what query will ever be compared against it. Relevance is a cheap post-hoc dot product between two fixed vectors. This is what makes ANN indexing (HNSW, Phase 3) possible at all: the corpus's vectors are static and precomputable. The cost of that speed is that the model never sees the query and passage together — it can't reason about their specific interaction, only about how close two independently-summarized points land in embedding space.

- **Cross-encoder** (this phase's `bge-reranker-v2-m3`): query and passage are concatenated into **one** input sequence and run through a single transformer together. Every token — query or passage — attends to every other token, at every layer, via standard self-attention:

  ```python
  # aegis/rerank/cross_encoder.py — the whole idea in two lines
  pairs = [(query, chunk.text) for chunk, _ in candidates]
  scores = self._score_fn(pairs)  # one joint forward pass per (query, passage) pair
  ```

  That joint attention is what makes cross-encoders more accurate at judging fine-grained relevance — the model can directly check "does this specific passage token sequence address this specific query," not just "are these two independently-summarized points nearby." It's also exactly why a cross-encoder **can't replace retrieval**: a score only exists for a specific `(query, passage)` pair after running them through the model together, so there's nothing to precompute or index. Scoring the whole 19,236-chunk corpus per query would mean 19,236 full transformer forward passes per query — this project's own measured throughput (§5) makes that arithmetic alone: even at Kaggle's fastest measured rate here, that's tens of minutes per query. A cross-encoder only works as a **second-stage** reranker over a small pool a cheap first-stage retriever (the bi-encoder, or BM25) has already narrowed down — O(n) scoring over n candidates, not O(corpus size).

- **ColBERT (late interaction, not implemented — design-record topic only per plan.md's cut list):** a middle point. Query and passage are still encoded **independently** (like a bi-encoder — no joint transformer pass, so passage vectors are still precomputable), but instead of pooling each into one vector, ColBERT keeps a vector **per token**. Relevance is computed at query time via "MaxSim": for each query token, take its maximum cosine similarity against any passage token, then sum across query tokens. This captures more fine-grained interaction than a single pooled vector (a bi-encoder can't tell you *which* words drove the match; ColBERT effectively can) without paying for full joint attention — at the cost of a much larger index (one vector per token, not per chunk) and slower scoring than a single-vector dot product. Cut from implementation here because the bi-encoder-vs-cross-encoder gap already measured (§5) demonstrates the load-bearing lesson (independent encoding vs. joint attention trades index-ability for interaction fidelity); ColBERT's specific engineering cost (per-token storage, MaxSim scoring) is well-documented in the literature and didn't need a second retrieval system built just to confirm it exists on the expected middle ground.

**The pipeline** (`rerank/pipeline.py`): `RerankedRetriever.search()` pulls `max(top_k, pool_size)` candidates from a base retriever, then either reranks them down to `top_k` (`enabled=True`) or returns the base retriever's own order directly (`enabled=False`) — the exact same config, one flag flipped, tested in `tests/test_rerank_pipeline.py` with fake retriever/reranker spies.

**The sweep methodology** (`experiments/07a_export_rerank_candidates.py` + `07b_kaggle_rerank_sweep.py`): dense candidates are fetched **once** per query at the largest pool size (100), then sliced locally for smaller pool sizes — a cross-encoder's score for a given `(query, chunk)` pair doesn't depend on what else is in its batch, so scoring the top-100 once and re-slicing for pool=10/20/50 is mathematically identical to reranking each pool size independently, at a fraction of the compute (100 scored pairs/query instead of 180).

## 5. What the numbers said

Full table and plot in `docs/results/rerank-tradeoff.md`. Headline:

| Pool size | NDCG@5 | Δ vs. no-rerank | Recall@10 | Rerank latency p95 |
|---|---|---|---|---|
| no-rerank (Phase 6 winning config) | 0.7550 | — | 0.4324 | — |
| 10 | 0.6873 | −9.0% | 0.4324 | 5,227 ms |
| 20 | 0.5344 | −29.2% | 0.2951 | 12,362 ms |
| 50 | 0.4072 | −46.1% | 0.2003 | 33,936 ms |
| 100 | 0.3340 | −55.8% | 0.1471 | 57,563 ms |

**Every pool size makes retrieval worse, and the effect grows monotonically with pool size.** Latency grows the same way — even the smallest pool tested (10) costs more than half the entire 3-second full-pipeline latency budget on reranking alone. **Reranking is disabled by default** in this system's config (`RerankedRetriever(..., enabled=False)`); neither axis supports turning it on.

## 6. What surprised me / what to watch for

**The hardware story is its own lesson, twice over, in the same shape.** MPS hung under real memory pressure (macOS's "stuck" scheduler state) — first when the bi-encoder and cross-encoder were both MPS-resident at once (matching a Phase 3 decision-log warning almost exactly), then *again* after fixing that, under this session's broader system-wide memory load. CPU inference was stable but I benchmarked its throughput on short synthetic placeholder text (30.5ms/pair) and only discovered real corpus-length text (up to ~400 tokens/chunk) runs at ~780ms/pair — 25x slower — after 19 real minutes of a doomed ~5.7-hour run. Moving to a Kaggle T4 GPU should have sidestepped all of this, but I made the *identical* mistake a second time: a fixed `BATCH_SIZE=64` picked for GPU headroom, not validated against real chunk length, meant pool≥50 was scored as one huge un-chunked batch, which — stacked on fragmented memory from ~300 prior forward passes with no cache-clearing between pool sizes — produced a genuine CUDA OOM. The lesson, twice-earned: **benchmark memory/throughput assumptions against real data length, not synthetic placeholder text, especially for anything attention-based, where cost scales with sequence length** — a short benchmark string is not a stand-in for a 400-token chunk when the operation is quadratic-ish in sequence length.

**The quality regression itself was the bigger surprise, and it demanded investigation rather than a shrug.** A stronger, more expensive model making retrieval measurably worse contradicts the entire premise of the phase, so per CLAUDE.md's rule to stop and investigate rather than move on, I pulled one real query's top-20 candidates and inspected the cross-encoder's actual output. It wasn't broken — it correctly ranked the single most on-point passage #1 with high confidence. The real mechanism: this golden set's ground truth (`auto_label.py`, Phase 4) is a token-overlap heuristic explicitly "biased toward inclusion" — any chunk sharing ≥15% of the query's words counts as relevant. Measured directly: **median 17 ground-truth chunks spanning 12 distinct documents per query**. That's a "broadly on-topic" bar, and a bi-encoder's fuzzy topical clustering happens to satisfy it better than a cross-encoder trained for sharp, one-passage relevance — so a *better* relevance judge disagreeing with a *loose* label set reads as "worse" in the aggregate metric. This is a real limitation of the Phase 4 labeling methodology surfacing here for the first time, not a defect in reranking, and it's worth carrying into Phase 11's judge-agreement measurement (Cohen's κ) — if Opus disagrees with the auto-labeled ground truth in the same direction the cross-encoder did, that would corroborate this explanation directly rather than leaving it as a plausible story.

## 7. Where this fits and what's next

This system's retrieval pipeline stays two-stage (dense retrieval alone) rather than three-stage (retrieval + rerank) heading into Phase 8, based on measured evidence that a third stage costs both quality and latency here — not because reranking wasn't tried or wasn't built. `RerankedRetriever` and `CrossEncoderReranker` remain fully wired and tested, so this decision is reversible with a config flag if the golden set's labeling methodology is revised later. The permissive-ground-truth finding (§6) is the most load-bearing open thread out of this phase: it's a property of *how the golden set was built*, not of any specific retrieval technique, and it should inform how much weight Phase 11's absolute numbers (not just its relative comparisons, which stay valid regardless) can bear.
