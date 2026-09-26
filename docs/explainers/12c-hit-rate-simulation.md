# 12c — Hit-Rate Simulation Against Zipfian Traffic

**Phase:** 12 · **Date:** 2026-09-26 · **Files:** `experiments/12b_hit_rate_simulation.py`, `docs/results/cache-hit-rate-simulation.md`
**Topics from the course:** Cost vs response quality (topic 34) — Units 1-2 measured each cache tier's per-request correctness; this unit measures what that correctness is worth in aggregate, under a traffic shape modeled on how real support questions actually repeat.

## 1. What I built

`experiments/12b_hit_rate_simulation.py`: draws a stream of up to 9,000 simulated queries from the 150 real dev-positive golden intents under a Zipfian popularity distribution, runs each one through the real `ExactMatchCache` → `SemanticCache` → miss chain (Units 1-2, unmodified), and reports cumulative hit rate, latency, and completion-token totals with vs. without caching. Output: `docs/results/cache-hit-rate-simulation.md`.

## 2. Why this approach

Units 1-2 already proved each tier works correctly in isolation (exact match round-trips; semantic match catches real paraphrases above 0.88 without false-positiving on real hard negatives). Neither measures what plan.md's exit criteria actually ask for here: hit rate and cost/latency reduction *under realistic traffic*. A uniform query sample would answer a different, easier question — every real support/FAQ workload is dominated by a handful of repeated questions, which is precisely why caching is worth building at all. Simulating that shape, rather than reporting per-request correctness again, is the only way to get a real aggregate savings number.

Cache-miss cost (the "what would this have cost without a cache" baseline) is bootstrapped from real, already-measured generation latency/tokens (`experiments/11b_qwen_reasoning_check.py`'s raw output) instead of issuing fresh LLM calls per simulated miss. Re-running live generations to measure how much the cache *avoids* running live generations would spend the exact quota this component exists to protect — the same discipline already applied when Phase 9/11 deferred their remaining live reruns to the user rather than burning more quota to re-derive a number a cheaper method could get honestly.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ----------------- |
| Uniform random sampling over the 150 intents | Simpler, no skew parameter to justify | Doesn't test the thing plan.md asks for — real support traffic isn't uniform, and a cache's whole value proposition rests on request concentration, which uniform sampling erases by construction |
| Fresh live generation per simulated cache miss, to get a real per-run cost instead of bootstrapping | Every number would come from this exact run, not a reused file | 9,000 requests project thousands of misses even at high hit rates; that's thousands of fresh Groq calls to measure a number this project already has 4 real samples of — spends the scarce quota Phase 9/11 are already rationing, for a number bootstrapping gets honestly from real data |
| Pick one Zipf skew value (e.g., "s=1.0, the classic Zipf's-law exponent") and report a single hit rate | Fewer numbers, cleaner headline | No real traffic log exists to justify any single value as "the" real one; asserting one would violate the same rule Unit 2 already applied to its threshold sweep — measure the sensitivity, don't guess a point |

## 4. How it works

For each of 3 skew values, a Zipf probability mass over the 150 intents' ranks is built, ranks are assigned to intents via a fixed random permutation (independent of file order), and `N` intent draws are sampled from that distribution:

```python
def zipf_pmf(n: int, skew: float) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = ranks ** (-skew)
    return weights / weights.sum()
```

Each drawn request becomes a query string — the golden intent's title line (what a real user would actually type, not the full issue body used for retrieval eval), or, for the 15 intents with a real Unit-2 paraphrase, that paraphrase 50% of the time. It's checked against the real caches in the real order a live system would use them:

```python
exact_hit = exact_cache.get(query_text, CORPUS_VERSION)
if exact_hit is None:
    semantic_hit = semantic_cache.get(query_text, CORPUS_VERSION)
    if semantic_hit is None:
        # real bootstrap-sampled miss cost + real cache .set() on both tiers
```

Every request's "if this had been a cold miss" cost is sampled once, up front, from the 4 real `(latency_ms, completion_tokens)` pairs — so the with-cache and without-cache totals are computed on the same per-request cost basis rather than two independently noisy ones.

`bge-large` calls are memoized by exact string (`_MemoizedEncoder`) rather than skipped or faked — a deterministic encoder in eval mode returns identical vectors for identical text, so this only avoids recomputing what a real repeat would compute anyway. Across all 3 skews and 9,000 requests each, exactly 165 real encode calls happen (150 titles + 15 paraphrases) — every semantic-tier decision in the results is still a real cosine-similarity comparison against a real embedding.

## 5. What the numbers said

Real run, `uv run python experiments/12b_hit_rate_simulation.py` (full output: `docs/results/cache-hit-rate-simulation.md`):

| skew | N | requests/intent | hit rate (total) | exact | semantic | latency reduction | token reduction |
|---|---|---|---|---|---|---|---|
| 0.8 | 150 | 1.0 | 56.7% | 56.0% | 0.7% | 52.3% | 56.3% |
| 0.8 | 9000 | 60.0 | 98.3% | 96.7% | 1.5% | 98.2% | 98.3% |
| 1.2 | 150 | 1.0 | 69.3% | 69.3% | 0.0% | 71.3% | 69.6% |
| 1.2 | 9000 | 60.0 | 98.3% | 97.5% | 0.8% | 98.2% | 98.2% |
| 1.6 | 150 | 1.0 | 86.7% | 86.7% | 0.0% | 86.2% | 87.2% |
| 1.6 | 9000 | 60.0 | 98.3% | 98.1% | 0.3% | 98.5% | 98.4% |

At the highest traffic volume tested (9,000 requests, 60 requests per unique intent), all three skews converge to essentially the same ~98% hit rate and ~98% latency/token reduction. At the lowest volume (150 requests, 1 per intent on average — a cold cache), skew matters a great deal: 56.7% (skew=0.8) vs. 86.7% (skew=1.6), because a higher skew concentrates repeats onto fewer intents faster, so the cache "warms up" sooner.

Real mean exact-hit latency: 0.014 ms. Real mean semantic-hit latency: 0.6-0.8 ms (dominated by the real `bge-large` cosine comparison against however many entries are cached so far, not a fixed cost). Both are 5-6 orders of magnitude below the real cache-miss cost (mean ~2.6s generation + 6.5ms retrieval, from the 4 real bootstrap samples).

## 6. What surprised me / what to watch for

**Traffic volume dominates skew, once volume is high relative to the intent pool — and this golden set's pool is small.** I expected skew to be the main driver of hit rate throughout. Instead, once N/intent-count exceeds roughly 10-20, hit rate saturates near its ceiling regardless of skew (all three land within 0.5pp of each other by N=4,500). The reason is arithmetic, not caching-specific: with only 150 real unique intents, even *uniform* traffic at N=9,000 would see each intent 60 times on average — almost every request past the first ~150 is structurally a repeat of something. **This means the ~98% hit-rate ceiling measured here is a property of this golden set's intent diversity (150 real distinct questions), not a general claim about this system's real-world cache hit rate against a much larger, more diverse real question space** — a production deployment facing thousands of genuinely distinct questions would sit further down this same curve, closer to the cold-start numbers (56-87%) than the saturated ones, until its own traffic volume caught up. Disclosed, not smoothed over: the honest headline is "this cache saturates fast against a 150-question space; the real-world ceiling depends on how large the real question space turns out to be," not a single 98% number.

**Semantic-tier hit share stays small (0.2-1.6% of all traffic) even though Unit 2 measured a 60% real paraphrase catch rate.** These aren't in tension — they're conditioned on different populations. Unit 2's 60% is "of the paraphrase pairs that reach the semantic tier, how many get caught." Here, only 15/150 (10%) of intents can ever produce a cross-phrasing repeat at all, and among those, the semantic tier is only *needed* when the incoming phrasing differs from whatever got cached first (the exact tier already catches same-phrasing repeats on its own). The two numbers are consistent; reported together because a reader who only saw the small percentage here might wrongly conclude the semantic cache barely works, when it's the small candidate population, not a weak catch rate, driving the small share.

## 7. Where this fits and what's next

This closes plan.md's Phase 12 step 3 exit criterion (hit rate, cost, and latency reduction under realistic traffic). Unit 4 (delta indexing) is next — it defines what a real `corpus_version` actually is and how it changes on a reindex, which this unit and Units 1-2 have so far only stubbed as a placeholder string. The saturation finding here also directly informs Unit 6's cost model: a per-query cost table needs at least two rows per config (cold-cache and warm-cache), not one, given how much this unit's own numbers move between N=150 and N=9,000.
