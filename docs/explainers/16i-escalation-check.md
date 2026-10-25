# 16i — The Escalation Check (the Air Canada metric)

**Phase:** 16 · **Date:** 2026-10-25 · **Files:** `src/aegis/eval/escalation_gate.py`, `src/aegis/eval/rerank_cache.py`, `tests/test_escalation_gate.py`, `tests/test_rerank_cache.py`, `ci/caches/rerank_scores.json`, `ci/baselines/escalation.json`, `experiments/16e_export_rerank_pairs.py`, `experiments/16g_score_rerank_pairs_local.py` (and `16f_kaggle_rerank_pairs.py`, written but not used), `escalation-gate` job in `.github/workflows/eval.yml`
**Topics from the course:** Evaluation strategies; Handling hallucinations. This is the check on the North Star: a confident wrong answer to a question that should have been escalated.

## 1. What I built

A gate that replays Phase 10's escalate-or-answer decision over the 180 dev queries (150 positives, 30 negatives) with the real confidence scorer at the shipped tau, and fails on `air_canada_fn` (negatives answered anyway; zero tolerance) or a drop in `deflection_rate` (positives answered; band). The expensive part, cross-encoder scores, comes from a committed content-addressed cache (`rerank_cache.py`, keyed by SHA-256 of model + query + chunk text), so a run takes **4 seconds** locally instead of a CPU pass over 538 pairs.

## 2. Why this approach

`_raw_signals` spends its time in two text-only functions: the cross-encoder score of (query, chunk) for the top-3 candidates, and the embedding of the top-10 candidates' text. On the M1, measured on real data: query vectors in the tracked embedding cache **180 / 180**, candidate chunk embeddings **1,352 / 1,352**, so the encode side is free; reranker pairs for the 180 queries **540 (538 distinct)**. A score depends only on (model, query text, chunk text), so score each pair once and look it up. Everything downstream (weights, bounds, tau, the escalate decision) is recomputed every run, which is the logic under test.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Score the 538 pairs on the runner every PR | No cache to maintain | The runner encodes ~10× slower than the M1 (2.4-2.9 s per query vs 0.27 s); the M1 pass alone was ~10 minutes. Order of an hour or more of CPU per PR against 2,000 free minutes a month. |
| Score on Kaggle (CLAUDE.md's rule) | GPU, minutes | Needs a manual handoff. The user waived the rule once for this job (below), so it was not needed; `16f_kaggle_rerank_pairs.py` stays as the path if the cache ever needs a large refresh. |
| Replay Phase 10's stored raw signals | Free | Tests nothing: a change to the scorer, weights, tau or retrieval would not be seen. |
| A 30-negatives-only subset | 6× cheaper | Loses the over-caution half of the matrix and still ~40 min of CPU on the runner. |
| **Content-addressed cache with a miss budget (chosen)** | Weights/bounds/tau changes re-evaluate for free; retrieval changes that move the top-3 compute only new pairs; a model or input change misses broadly and fails loudly | A one-off scoring run, and a cache to keep honest (see §6). |

## 4. How it works

`make_score_fn(model, inner, cache, max_uncached)` wraps the real cross-encoder: hits come from the cache, only misses reach the model, results return in input order, and the miss budget (60 pairs, `settings.gate.escalation_max_uncached_pairs`) is cumulative and checked *before* any model call; past it the gate raises `TooManyUncachedPairs` and says to refresh the cache. The model is loaded only on a miss (`_default_score_fn` is lazy), so an all-hit run never touches it. 9 tests. `score_escalation` wraps the existing, tested `escalation_appropriateness` (5 tests): `air_canada_fn` is a count, undefined rates are `NaN` and omitted from the report so a gated metric going missing fails.

## 5. What the numbers said

**Scoring, run once locally by user waiver.** CLAUDE.md sends any job projected over 10 minutes to Kaggle; on 2026-10-25 the user waived that for this one job. Measured: **538 pairs scored in 584 s (9 min 57 s wall including model load)**, `experiments/16g_score_rerank_pairs_local.py`, CPU, M1 Pro, sorted by length, batch 32, checkpointed per batch. **My projection was 11.8 min, 18% too high**: I benchmarked 12 pairs sampled across the set in a single batch, which pads every pair to the longest in the batch (1.32 s/pair). Length-sorted batches ran from 0.12 s/pair on the shortest pairs to 1.09 s/pair averaged over the run, and the last 26 (longest) pairs alone took ~290 s. The honest lesson: a benchmark must use the batching the real run will use, or it measures padding. The result is 538 scores in `ci/caches/rerank_scores.json` (48,935 bytes). The 347 margin pairs (top-5 beyond top-3) in the export were not scored.

**The gate reproduces Phase 10 exactly** (fingerprint `e99f16de52c7d00f`, tau 0.5, live index and again on a freshly restored 2-CPU optimizer-off index):

| Metric | Phase 10 published | Gate run |
| --- | --- | --- |
| Negatives answered anyway (`air_canada_fn`) | 3 | **3** |
| Negative recall | 0.90 (27/30) | **0.90** |
| Deflection (positives answered, 126/150) | 0.84 | **0.84** |

540 cache lookups (180 queries × 3), **540 hits, 0 misses**, no model load; two runs identical; **4 s wall**. Baseline = this run (`ci/baselines/escalation.json`).

**Does it catch a regression?** Mutation tests via `CONFIDENCE_TAU` (free, from cached scores):

| tau | Air Canada slips | Deflection | Gate |
| --- | --- | --- | --- |
| 0.50 (shipped) | 3 | 0.8400 | pass |
| 0.30 (looser: "answer more") | **6** | 0.8933 (improved) | ❌ FAIL on `air_canada_fn` |
| 0.65 (stricter) | 1 | **0.5000** | ❌ FAIL on `deflection_rate` |

The 0.30 row is the one to remember: deflection *improves* while the safety count doubles, so a gate on a single "better is better" metric would have called it a win. The 0.65 row matches Phase 10's recorded curve (97% recall, 50% deflection).

## 6. What surprised me / what to watch for

- **The check that guards the north star costs nothing to run** once scored; the effort was one-off. A job projected at hours collapsed to 4 seconds because most of its cost was a pure function of unchanged text.
- **The cache is the thing to keep honest.** A stale hit would hide a reranker regression. The key includes the model name and both texts, so a model swap or a changed chunk misses; the miss budget makes a broad change fail instead of silently scoring on a slow CPU. What it does not protect against: someone editing `ci/caches/rerank_scores.json` by hand. It is committed plain JSON, so any edit shows in a PR diff.
- **Hardware may reorder near-ties and shift a few top-3 pairs.** Across the live index and a restored CI-style index the distinct top-3 pairs were identical (538 = 538), and the CI-style run had 0 misses; a runner with a different CPU (16h) may differ, which the 60-pair budget and the unscored margin pairs are for. If CI shows misses, score the 347 margin pairs the same way.
- **The deflection band (0.03) is a placeholder.** The decisions are deterministic given cached scores, so noise comes only from candidate ordering on different hardware, which has not been observed yet.
- **This gate protects against changes, not against tau being wrong.** tau 0.5 still lets 3 of 30 negatives through; the gate pins that number, it does not improve it.

## 7. Where this fits and what's next

Every Phase 16 build item now has a gate: retrieval, router, generation, escalation, plus latency per stage. The PR comment has four sections. This check is the one Phase 17 needs most: the LangGraph port moves the confidence and escalate step, and "identical metrics through both paths" now includes `air_canada_fn`. Refreshing the cache after a deliberate reranker change: `uv run python -m aegis.eval.escalation_gate run --fixtures ci-fixtures --out results/escalation.json --save-cache --max-uncached 1000`, then re-baseline.
