# 14h — The semantic-cache collision and the cold first request

**Phase:** 14 · **Date:** 2026-10-17 · **Files:** `src/aegis/config.py` (`CacheSettings.semantic_cache_enabled`), `src/aegis/factory.py`, `src/aegis/pipeline.py` (`warm_up`, optional semantic tier), `src/aegis/embedding/encoder.py` (`Encoder.warm_up`), `src/aegis/retrieval/dense.py` (`DenseIndex.warm_up`), `src/aegis/api/app.py` (warm-up log), `experiments/14d_semantic_cache_short_queries.py`, `docs/results/semantic-cache-short-queries-raw.json`, `docs/results/cache-performance.md` (addendum)
**Topics from the course:** Semantic caching (Module 5, #33); Latency vs. response quality (Module 5, #35); What makes production challenging (Module 5, #29)

## 1. What I built

Two investigations from 14e's real browser session, each ending in a fix:

1. **Collision.** The follow-up "what about making it required?" was served the cached answer to a different question. Cause found; the semantic cache tier is now **off by default** (user decision), with code and tests kept behind `CACHE_SEMANTIC_CACHE_ENABLED`.
2. **Cold start.** The first request after startup spent 8.4–10.5 s in `cache_lookup`. Two stacked causes found. Warm-up now pays both at startup: the first request's lookup went from 8,447–8,669 ms to 166 ms on the real server.

## 2. Why this approach

**Collision: the threshold was tuned on the wrong population.** Phase 12 (12b) chose 0.88 as the lowest threshold with zero false positives. But its negatives were pairs of *whole GitHub issue bodies*, while its positives were short titles vs. short paraphrases. Chat traffic is short, and rewritten follow-ups are short *and templated*: the Phase 8 rewriter writes nearly every one as "How do I … in FastAPI?". For short texts, the shared template dominates the embedding, and the one word that changes the meaning hardly moves it.

What a semantic hit skips decides what to do about it. It returns before routing, confidence scoring and citation validation, so a false positive serves a fully cited answer to *a different question*, with none of the guardrails having looked at the question actually asked. That's a direct instance of the Air Canada failure mode. Against that, Phase 12's own simulation (12c) put the semantic tier's share of all traffic at **0.2–1.6%**. The user chose to turn it off (`AskUserQuestion`, 2026-10-17). The exact tier stays: it can't confuse two different questions.

**Cold start: fix the warm-up, not the request path.** Startup can absorb any one-time cost; a user's first question shouldn't. Both causes were invisible, because warm-up "succeeded" while warming nothing (see §5).

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Skip the semantic tier for rewritten follow-ups only | Keeps semantic hits for direct questions; fixes the observed case | A user who types "How do I make a query parameter required in FastAPI?" directly collides at the same 0.9354, with no rewrite involved. It fixes the symptom's entry point, not the mechanism. |
| Raise the threshold to 0.94 | Keeps the tier, excludes the real pair | 0.9354 clears it by 0.005, on n=1 real collision. Phase 12's paraphrase catch rate falls from 9/15 to 4/15, so a tier worth 0.2–1.6% of traffic keeps less than half of that, behind a margin no measurement supports. |
| Strip the shared template ("in FastAPI") before embedding | Targets the mechanism directly | Brittle: it depends on the rewriter's phrasing, which is an LLM's output and changes with the model or prompt. And it only removes the one template I know about. |
| Re-run Phase 12's sweep with templated negatives and re-tune | The principled re-measurement | No real templated-negative population exists yet: only n=1 real collision, plus a few of my own probes. Tuning on probes I wrote would be tuning on illustrative data. Recorded as the condition for turning the tier back on (§7). |
| Warm the encoder by encoding a fresh random string at startup | One line, no new method | Writes a junk row into the persistent embedding cache on every startup. `Encoder.warm_up` calls the model directly and touches the cache in neither direction (tested). |

## 4. How it works

**Semantic tier toggle.** `factory.build_pipeline` passes `semantic_cache=None` unless `settings.cache.semantic_cache_enabled`. `AnswerPipeline` skips the semantic `get` and `set` when it is `None` (tested: exact-only lookup and write).

**Warm-up**, in order:

```python
with _timed(timings, "retrieval"):   candidates = self.retriever.search(_WARM_UP_QUERY, ...)
with _timed(timings, "confidence"):  self.confidence_scorer.score(_WARM_UP_QUERY, candidates)   # cross-encoder, MPS
with _timed(timings, "encoder"):     self.retriever.warm_up(_WARM_UP_QUERY)   # bge-large, CPU, bypasses the cache
```

- `Encoder.warm_up(text)` calls `self._encode_fn([text])` directly. `encode()` could never warm the model, because the fixed warm-up query's embedding is in the on-disk embedding cache after its first-ever run, so `encode()` returns the cached vector in 0.1 ms without loading anything.
- The encoder goes **last**: the first CPU encoder pass after the cross-encoder's first MPS run costs a further ~1.3–1.7 s once, and this is where that gets paid.
- It goes through the retriever, not the semantic cache: with the semantic tier off, dense retrieval is what encodes every miss.
- Startup timings are now logged (`api.warm_up`); nothing recorded them before.

## 5. What the numbers said

**Recovering the real collision (Langfuse, 2026-10-01 06:08–06:10 UTC).** The 14e write-up guessed the cached question was "How do I declare a query parameter?". The trace history shows the session's only cached answer was "How do I declare a query parameter **in FastAPI**?". The follow-up's rewrite was "How do I make a query parameter required in FastAPI?". bge-large cosine similarity:

| Pair | Similarity | Source |
| --- | --- | --- |
| "…make a query parameter required in FastAPI?" vs "…declare a query parameter in FastAPI?" | **0.9354** | real (the 14e collision) |
| same, cached side without "in FastAPI" | 0.8248 | real texts, minus the suffix |
| both without "in FastAPI" | 0.8770 | probe |
| "…required in FastAPI?" vs "…optional in FastAPI?" | 0.9122 | probe (mine, not traffic) |
| "…path parameter required…" vs "…query parameter required…" | 0.9048 | probe (mine, not traffic) |

**Short-title negatives** (`experiments/14d`, all 11,175 pairs of the 150 real dev golden titles): p50 0.527, p99 0.727, max **0.8507**. Zero pairs ≥ 0.88. My first hypothesis ("short queries in general collide") was **refuted**: real titles don't share a template. The collision comes from the rewriter's template, not from length alone.

**Cold start, decomposed** (n=1 per row unless noted):

| Measurement | Result |
| --- | --- |
| Real server, first fresh request, `cache_lookup` (before) | 8,447 ms and 8,669 ms (two server starts); 9,304 ms in 14e |
| `encode(warm-up query)` in a fresh process | **0.1 ms**, model not loaded (cache hit) |
| first uncached text, fresh process | **10,533 ms** (model load); second 133 ms |
| fresh thread vs main thread, after load | 137–145 ms vs 142–187 ms: **refuted** as a cause |
| embedding-cache lookup / write-back | 0 ms / 1 ms: refuted |
| Langfuse first span open + end | 0 ms: refuted |
| after fix 1 (encoder warmed first), real server, first request | 2,724 ms; then n=5: 2,213 / 154 / 146 / 151 / 214 ms |
| in-process, full factory pipeline, first encoder pass after warm-up | 1,322 ms, then ~150 ms |
| same, plus one extra encoder pass after confidence | extra pass 1,683 ms; first query **149 ms** |
| after fix 2 (encoder warmed last), in-process | first query 136 ms |
| after fix 2, real server, first two requests' `cache_lookup` | **166 ms**, 144 ms |

Startup now logs `encoder` ≈ 8.1–8.8 s and `confidence` ≈ 15.5–16.4 s on this machine (n=2 starts).

**Replaying the 14e conversation** (real server, semantic tier off, throwaway caches): "How do I declare a query parameter in FastAPI?" → answered. Then "what about making it required?" + history → rewritten "How do I declare a required query parameter in FastAPI?", cache **miss**, answered specifically ("you simply omit the default value… `q: str` makes it required"). The first, cold request's stages added up to ~2.0 s (n=1), against 22.7 s to first text in 14e (which also included a 17.9 s Gemini generation).

`uv run pytest -q` → **581 passed** (+5).

## 6. What surprised me / what to watch for

- **The 14e write-up's attribution was a guess, and slightly wrong.** It named the cached question without "in FastAPI". Measured that way the pair is 0.82, and my first sweep "showed" the cache shouldn't have hit. Only the trace history settled it. The general lesson: a cache hit must record *what it matched*. `SemanticCache` stores no query text, so a hit can't be audited. If the tier comes back, it needs `matched_query` and `similarity` on the span.
- **Warm-up was silently a no-op for the encoder since Phase 13.** It reported success and real timings (retrieval ~25 ms), because the retrieval it ran was served from the embedding cache. A warm-up step that measures nothing looks the same as one that works.
- **The post-MPS first CPU pass (~1.3–1.7 s) is measured, not explained.** It's order-dependent and once per process. My guess is CPU/MPS backend interplay in PyTorch, but that's unverified. If the encoder ever moves to MPS, or the reranker to CPU, re-measure.
- **This may bear on Phase 13's unexplained confidence variance.** 13k found query `753` costs ~0.5–0.8 s in one harness but ~2.5 s in the 13c batch script. A one-time, order-dependent cost when switching between CPU and MPS models is the same shape of effect. Not tested; noted for whoever reopens p95.
- **Startup is now ~25 s and fully paid before `/health` answers.** That's fine for a demo. A deployment health check with a shorter timeout would mark it dead.
- Phase 12's published cache numbers still stand as measurements. `cache-performance.md` has an addendum saying the semantic tier no longer ships.

## 7. Where this fits and what's next

Turning the semantic tier back on requires: (a) a real templated-negative set (pairs of real rewritten follow-ups and the questions before them; the review queue in 14g is one source), (b) a re-tuned threshold with zero false positives on it, and (c) `matched_query`/`similarity` recorded on every hit. Until then, exact-only. With both 14e findings closed, the only Phase 14 item left is the regime-C multi-turn eval behind the open exit criterion.
