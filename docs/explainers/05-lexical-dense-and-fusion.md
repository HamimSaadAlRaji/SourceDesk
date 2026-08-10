# 05 — Lexical, Dense & Fusion Retrieval

**Phase:** 5 · **Date:** 2026-08-03 · **Files:** `src/aegis/analysis/analyzer.py`, `src/aegis/retrieval/sparse.py`, `src/aegis/retrieval/dense.py`, `src/aegis/retrieval/fusion.py`, `src/aegis/retrieval/hybrid.py`
**Topics from the course:** Information retrieval basics; Three retrieval techniques (keyword, semantic, metadata filtering); Metadata filtering; Keyword search: bag of words → TF-IDF → BM25; Hybrid search & Reciprocal Rank Fusion (RRF)

## 1. What I built

Five small modules that together form one retrieval interface. `analysis/analyzer.py` tokenizes text so identifiers like `status_code`, `app.get`, `HTTP_401_UNAUTHORIZED`, `--reload`, and `0.100.0` survive intact instead of being shredded by a generic word tokenizer. `retrieval/sparse.py` wires that analyzer into `bm25s` for lexical search. `retrieval/dense.py` wraps Phase 3's Qdrant collections with native metadata pre-filtering. `retrieval/fusion.py` implements RRF and score-normalized linear fusion as two independent, comparable ways to combine ranked lists. `retrieval/hybrid.py` composes all of it behind one `HybridRetriever.search(query, strategy=...)` call, where `strategy` is exactly the axis Phase 6's ablation grid sweeps.

## 2. Why this approach

The corpus is FastAPI support content: half of it is code, error codes, CLI flags, and version numbers. Dense embeddings are trained to find semantic *similarity*, and similarity is close to worthless for a query like "ECONNREFUSED" — the model has no reason to place a rare, structurally meaningful token near the handful of chunks that happen to contain it (measured below: it doesn't, at all, in the top 200 of 19,236). Support queries frequently *are* an exact identifier, so lexical search needs to actually see that identifier as a token, not as three fragments after a generic tokenizer lowercases and splits on non-alphanumerics. The analyzer is the one component standing between "BM25 exists" and "BM25 does anything useful here" — which is exactly why it's the piece plan.md calls out to build, while BM25 itself is imported.

Fusion needs two independently-argued strategies, not one, because they encode different assumptions about what a raw retrieval score means, and this corpus is exactly the case where that assumption matters: BM25 scores and cosine-similarity scores live on incomparable scales that shift with query length and vocabulary rarity. RRF sidesteps the problem by only looking at rank; linear fusion confronts it directly via per-query min-max normalization. Which one actually wins is an empirical question for Phase 6, not something to decide from folklore — plan.md is explicit about this, and the fastest way to answer it later is to make both real, tested, and swappable now.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ------------------ |
| Qdrant native sparse vectors instead of a standalone `bm25s` index | One database for both dense and sparse; a single `query_points` call could do fused retrieval server-side | BM25 relevance doesn't depend on which encoder embedded a chunk, but Qdrant's sparse vectors would need indexing into all three Phase 3 collections (`bge_small`/`bge_large`/`bge_m3`) redundantly — three copies of the same sparse index for zero benefit. `bm25s` builds one in-memory index from `chunks.jsonl` directly, and exposes `k1`/`b` as constructor args instead of behind Qdrant's own (less transparent) sparse-vector configuration — closer to the metal for the Phase 5/6 sweep. |
| Reconstruct `Chunk` objects entirely from Qdrant's payload | One less thing to keep in sync; no separate `chunks.jsonl` lookup needed at query time | Phase 3's payload schema (`vector_store.py`) deliberately excludes `content_hash` and `embedding_text` — it was built as a filter/display cache, not a chunk store. Reconstructing from it would mean synthesizing a fake `content_hash`, silently lying in a field other components may later trust. `DenseIndex` instead takes the same `list[Chunk]` `SparseIndex` does and looks up full chunks by ID from the corpus already in memory — no Qdrant re-index, no synthetic data. |
| Stem/lemmatize and drop stopwords in the analyzer, matching a conventional IR pipeline | Standard practice; would shrink the vocabulary and might improve generic recall | Directly works against the analyzer's actual job. Stemming `response_model` and `responses`, or `hashed_password` and `hash`, to the same root would blur exactly the identifiers this component exists to keep exact — the whole reason a support query for `ECONNREFUSED` should hit the right chunk is that nothing normalizes it into something less specific. Stopwords are already down-weighted by BM25's own IDF; a hand list would just duplicate that for free. |
| Also split `camelCase` (e.g. `HTTPException`, `OAuth2PasswordBearer`) | The corpus has plenty of it; symmetry with the other four rules | Not in plan.md's named list (`snake_case`, dotted paths, `SCREAMING_CASE`, `--flags`, version strings), and CLAUDE.md is explicit about not adding scope beyond what's asked. Logged here rather than silently built — a real candidate if Phase 6 shows lexical recall on class-name-heavy queries lagging. |

## 4. How it works

**The analyzer's one rule, applied uniformly.** A compound identifier is emitted whole (so an exact-identifier query still matches) *and* decomposed into parts (so a natural-language query matches too) — except version/IP-style numeric sequences, whose "parts" are bare integers with no standalone meaning:

```python
_TOKEN_RE = re.compile(r"-{0,2}[A-Za-z0-9][A-Za-z0-9_.\-]*")
_VERSION_RE = re.compile(r"^v?\d+(\.\d+){1,3}$", re.IGNORECASE)

def _expand(raw: str) -> list[str]:
    lowered = raw.lower()
    if _VERSION_RE.match(raw):
        return [lowered]                       # "0.100.0" stays whole

    core = lowered.lstrip("-")
    parts = [p for p in _SPLIT_RE.split(core) if p]

    tokens = [lowered]
    if len(parts) > 1:
        tokens.extend(parts)                   # "status_code" -> + status, code
    elif core != lowered:
        tokens.append(core)                    # "--reload" -> + reload
    return _dedupe(tokens)
```

`status_code` → `["status_code", "status", "code"]`. `HTTP_401_UNAUTHORIZED` → `["http_401_unauthorized", "http", "401", "unauthorized"]`. `ECONNREFUSED` → `["econnrefused"]` (no separator, nothing to decompose, and critically *not* shredded into letters). `0.100.0` → `["0.100.0"]` only.

**RRF** (`retrieval/fusion.py`):

```
score(d) = Σ_r  weight_r / (k + rank_r(d))
```

rank 1-indexed, a chunk absent from a list contributes 0 for it. `k=60` (the value from Cormack et al.'s original RRF paper) damps how much the very top ranks matter relative to the rest — which is also the surprise in §6.

**Linear fusion**: each ranked list's raw scores are min-max normalized to `[0, 1]` independently before a weighted sum, so BM25's and cosine's incomparable scales never mix un-normalized. A list with zero score spread (all scores tied) normalizes every member to `1.0` rather than dividing by zero.

**The composed interface** (`retrieval/hybrid.py`): `HybridRetriever.search(query, top_k, strategy)` where `strategy ∈ {dense, sparse, hybrid_rrf, hybrid_linear}` is a `StrEnum` — literally the retrieval axis of Phase 6's ablation grid. For the two hybrid strategies, each side is asked for `max(top_k, candidate_pool_size)` results before fusing, not just `top_k` — fusion needs breadth to find cross-method agreement in, and asking for exactly `top_k` from each side would silently starve it.

## 5. What the numbers said

All numbers below are from the real 19,236-chunk corpus (`data/processed/chunks.jsonl`) and the live `aegis_chunks_bge_small` Qdrant collection (Phase 3), not a toy example.

| Check | Result |
| --- | --- |
| Analyzer test suite | 56/56 green, including all 40 real corpus tokens (`tests/test_analyzer.py`) — the Phase 5 exit criterion |
| Full test suite after Phase 5 | 245/245 green (`uv run pytest -q`) |
| BM25 on query `"ECONNREFUSED"`, real issue `issue::5578` (contains the literal token) | rank 1, score 3.1467 |
| Dense (bge-small) on the same query, same target chunk | **not found in the top 200** of 19,236 chunks (dense top-1 was an unrelated chunk, score 0.6935) |
| Unfiltered dense search, query about `response_model` | top-5 is 100% issue chunks (0.82–0.86 cosine) — the correct doc (`tutorial/response-model.md`) is absent |
| Same query, `doc_type=documentation` pre-filtered | top-1 is `tutorial/response-model.md::20`, score 0.7864 — the actually-correct chunk |
| Hybrid RRF vs. hybrid linear, same real query (`"how do I add a custom response_model..."`), unfiltered | Different top-5 orderings (RRF: `issue::800` first; linear: `issue::4796` first) — both surface `tutorial/response-model.md` but rank it differently among issue chunks |

The exit criterion "BM25 returns sane results for ECONNREFUSED-style queries where dense fails" isn't a hypothetical — it's exactly what happened, on a real issue, with a rank-1-vs-not-in-top-200 gap.

Numeric `k1`/`b` tuning and Recall@10/MRR comparisons between `dense`/`sparse`/`hybrid_rrf`/`hybrid_linear` are **not measured yet** — that requires `ranx` and the eval harness, which are Phase 6's job against the golden set. `RetrievalSettings.bm25_k1`/`bm25_b`/`rrf_k`/`rrf_dense_weight`/`rrf_sparse_weight` are wired as config, not hardcoded, specifically so Phase 6 can sweep them without touching this code.

## 6. What surprised me / what to watch for

**A regex bug that looked like a design problem at first.** The first version of `_TOKEN_RE` was `r"--?[A-Za-z0-9]..."`, intending "zero, one, or two leading dashes." It actually means "exactly one mandatory dash, then an optional second" — `--?` is `-` followed by `-?`, not `(-{0,2})`. Every plain word in the corpus (`run`, `with`, `status_code`, all 40 real tokens) silently vanished; only `--flag`-style tokens survived. All 46 of the first 56 analyzer tests failed at once, and the failure output made the shape of the bug obvious immediately (`analyze("run with --reload")` returned `['--reload', 'reload']` — two tokens instead of four, with exactly the non-dashed words missing). Caught before it ever touched BM25, because the 40-real-token test existed and ran before anything downstream was built on top of it — exactly the ordering CLAUDE.md's TDD requirement is for.

**Unfiltered hybrid retrieval on this corpus is still issue-dominated**, same as every phase before it (Phase 1's classifier, Phase 3's HNSW duplicates, Phase 4's golden-set harvesting). Neither RRF nor linear fusion fixes this on their own — fusing two rankings that are both individually issue-heavy just produces a fused ranking that's also issue-heavy. The `doc_type` pre-filter Phase 3 already validated is still the actual fix, and `DenseIndex.search` supports it, but `HybridRetriever` doesn't yet apply it to the sparse side (`SparseIndex.search` has no filter parameter — see §3). Worth watching in Phase 6: if per-stratum recall is weak on `how_to` queries specifically, this is the first place to look.

**RRF's `k=60` matters more than "just a damping constant."** Writing `test_rrf_rewards_cross_method_agreement_over_a_single_top_rank` required going back and recomputing by hand: at the default `k=60`, the difference between rank 1 and rank 4 in a single list is tiny (`1/61` vs. `1/64` ≈ 0.0002), so a naive test claiming "agreement beats a single #1" failed until the single-list rank gap was widened to ~10. At smaller `k`, top-rank position would dominate much more. This is worth remembering when Phase 6 sweeps `k` — it's not just a magic constant, it's choosing how much the fused ranking trusts "consensus across methods" vs. "either method's most confident pick."

## 7. Where this fits and what's next

`HybridRetriever` is what Phase 6's eval harness calls once per config in the ablation grid — this phase built the thing being measured, not the measurement itself. Phase 6 adds `ranx` metrics, the config-hashed harness, and the actual `k1`/`b`/`rrf_k`/weight sweep against the golden set's 180 dev queries, reported per-stratum. The `doc_type`-filtering gap on the sparse side (§6) and whether `hybrid_rrf` or `hybrid_linear` actually wins on this corpus are both open questions Phase 6 answers with numbers, not this phase's guesses.
