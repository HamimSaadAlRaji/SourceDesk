# 12b — Semantic Cache & Threshold Tuning (Unit 2)

**Phase:** 12 · **Date:** 2026-09-26 · **Files:** `src/aegis/cache/semantic.py`,
`experiments/12a_semantic_cache_threshold_sweep.py`, `src/aegis/config.py` (`CacheSettings`),
`tests/test_semantic_cache.py`
**Topics from the course:** Cost vs response quality (topic 34); Handling hallucinations (topic 25) — the
semantic cache is a hallucination vector this project builds itself if the threshold is wrong, per plan.md's
own framing

## 1. What I built

`SemanticCache` — embeds a query (bge-large, the same encoder used for retrieval), brute-force cosine-
matches it against every real cached entry scoped to the current `corpus_version`, and serves the best
match's `CachedAnswer` only above `CacheSettings.semantic_similarity_threshold` (0.88). The threshold itself
comes from a real, local, zero-LLM-cost sweep (`experiments/12a_semantic_cache_threshold_sweep.py`,
`docs/results/semantic-cache-threshold.md`), not an assumed value.

## 2. Why this approach

plan.md is explicit that a too-loose threshold "serves the wrong answer to a similar-but-different
question, which is a hallucination vector you built yourself" and asks to tune it on the golden set and test
for it explicitly — not just build the cache and pick a plausible-looking number. Real measurement against
this exact corpus surfaced something a guessed threshold would have missed: genuine paraphrases and
genuinely-different-but-related real golden queries have **overlapping** similarity distributions in this
embedding space (real hardest distinct-question pairs range up to 0.8694; real paraphrases range from 0.744
to 0.948) — there is no threshold that perfectly separates them. Given that overlap, the choice has to be
asymmetric-cost, same logic as Phase 10's tau: a false positive here serves a confidently wrong cached
answer (this project's own North Star failure mode); a false negative just costs one extra real
retrieval+generation call. So the threshold (0.88) is the lowest value in the sweep with **zero** false
positives on the real hardest-negative set, not the value that maximizes hit rate.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ----------------- |
| Pick a threshold from general embedding-similarity conventions (e.g., "0.9 is usually safe") without measuring this corpus | Faster, no sweep needed | The real sweep found this corpus's hardest real near-miss pairs reach 0.8694 — a generic "0.9 is safe" guess would have been right by luck here, but the whole point of plan.md's instruction is not to rely on luck; a corpus with tighter topical clustering could easily produce near-misses above 0.9. |
| Generate the paraphrase positives with an LLM instead of hand-writing them | More naturalistic phrasing variance, less single-author bias | Every model's real quota was already committed to other Phase 11 work session-wide; hand-writing 15 real paraphrases costs zero LLM calls and keeps Phase 12 fully unblocked, at the real, disclosed cost of being one person's phrasing rather than a sampled population — named directly in `docs/results/semantic-cache-threshold.md`, not hidden. |
| An ANN index (e.g., a small Qdrant collection) for the semantic cache's own similarity search | Consistent with `DenseIndex`'s approach to the main corpus | A query cache's real working set is orders of magnitude smaller than the 19k-chunk corpus HNSW exists to search efficiently — brute-force numpy cosine similarity is simpler, exact (no approximate-search tradeoff to reason about), and fast enough at this scale. Revisit only if real production cache size ever approaches corpus scale. |

## 4. How it works

The sweep (`12a_semantic_cache_threshold_sweep.py`) computes two real populations: the 15 highest-similarity
pairs among genuinely different real dev-positive golden queries (via a full pairwise similarity matrix,
`vecs @ vecs.T`, reusing Phase 3's confirmed cosine-equals-dot-product-on-normalized-vectors result), and 15
hand-written paraphrases of real golden queries. Sweeping `[0.75, ..., 0.96]` and taking the lowest threshold
with 0 real false positives found **0.88**, catching 9 of 15 real paraphrases (60%) — a real, moderate hit
rate, not inflated.

`SemanticCache.get` computes true cosine similarity (`dot(a,b) / (|a| * |b|)`) rather than assuming
pre-normalized input vectors, unlike `DenseIndex`'s reliance on `sentence-transformers`'
`normalize_embeddings=True` — a small, deliberate robustness choice so this cache doesn't silently misbehave
if ever wired to a different embedder that doesn't guarantee unit-norm output.

## 5. What the numbers said

Full table in `docs/results/semantic-cache-threshold.md`. Headline: at threshold 0.88, 0/15 real
false positives, 9/15 (60%) real true positives. The real, honest hit-rate ceiling this specific threshold
choice implies: looser real paraphrases (e.g., "How can I get upload files?" vs. "how do i receive an
uploaded file in fastapi?", real similarity 0.7441) will **not** hit this cache — a deliberate, disclosed
tradeoff, not a gap to hide.

## 6. What surprised me / what to watch for

The real overlap between "same question, different words" and "different question, same topic" was more
substantial than expected going in — five of fifteen real hand-written paraphrases scored *below* the
highest real distinct-question similarity (0.8694). This means the semantic cache's real hit rate is capped
well below "catches every paraphrase," and that's the correct, safe outcome given the asymmetric cost, not a
tuning failure to fix later. Also worth flagging: the 15 real paraphrases are one person's phrasing, not a
sampled population — the 60% hit-rate figure should be read as directional, not a population estimate (real,
disclosed limitation, `docs/results/semantic-cache-threshold.md`).

## 7. Where this fits and what's next

Sits behind the exact-match cache (Unit 1) in the architecture (plan.md §3) — a miss there falls through to
this tier before the router. Unit 3 (hit-rate simulation against a real Zipfian query distribution) is the
natural next step to measure what these two cache tiers actually save in aggregate, not just per-query
correctness.
