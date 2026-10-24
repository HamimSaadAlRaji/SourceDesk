# 16c — The Gate Core: Fixtures, Comparison, Tolerance

**Phase:** 16 · **Date:** 2026-10-23 · **Files:** `src/aegis/eval/gate.py`, `src/aegis/eval/ci_fixtures.py`, `src/aegis/eval/retrieval_gate.py`, `GateSettings` in `src/aegis/config.py`, `ci/baselines/retrieval.json`, `tests/test_gate.py`, `tests/test_ci_fixtures.py`, `experiments/16d_encoder_noise.py`, `docs/results/encoder-noise-raw.json`
**Topics from the course:** Evaluation strategies (system vs. component, code-based vs. LLM-judge); Retrieval evaluation metrics (Recall, MRR, NDCG).

## 1. What I built

Three parts, none yet wired into a workflow (that is 16d):

- `gate.py` — `evaluate(current, baseline, specs)` returns pass/fail per metric with a tolerance band, plus hard problems; `render_markdown` produces the PR-comment table.
- `ci_fixtures.py` — `export` snapshots the served collection into `ci-fixtures/` (gzipped snapshot + gzipped `chunks.jsonl` + `MANIFEST.json` of SHA-256 hashes); `restore` verifies the hashes, loads the snapshot into a fresh Qdrant, checks the collection is green with the manifest's point count, and puts `chunks.jsonl` in place.
- `retrieval_gate.py` — `run` scores dense bge-large retrieval on the 150 dev positives against the restored index; `compare` diffs against `ci/baselines/retrieval.json` and exits 1 on failure; `baseline` rewrites the baseline (a deliberate, reviewable act).

## 2. Why this approach

**A gate's failure mode is passing when it should not.** So `evaluate` treats "no information" as failure: a gated metric missing or NaN/inf on either side, a baseline with a different corpus fingerprint, a different `n_queries`, or an empty spec list all fail. Only a clean in-band comparison passes. 13 tests pin each of these, plus the inclusive boundary and the lower-is-better direction (needed for the p95 gate in 16e).

**The fingerprint is derived from the verified file hashes**, so a report can never be attributed to a corpus other than the one that produced it; swapping the release asset changes the fingerprint and the comparison refuses rather than reporting a phantom regression.

**Queries are encoded live with the embedding cache bypassed.** The tracked `data/cache/embeddings.sqlite` already contains every golden query, so using it would make CI faster (~40 s) but would hide any regression in the encoder path (prefixing, truncation, model name) — exactly what a code change could break.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Relative tolerance (e.g. ±5%) | Scales with the metric | The metrics are bounded fractions with different sizes (recall 0.43, MRR 0.87); an absolute band means the same amount of ranking change for each, and is what the noise measurement below produces. |
| Gate on all five metrics | More coverage | Precision@10 and MAP@10 are the same signal as recall/NDCG on a fixed top-10 and would double-count one regression. Recorded but not gated. |
| Use the embedding cache in CI | 40 s faster | Hides encoder-path regressions (§2). |
| Commit the baseline from the first runner run only | Platform-true baseline | Needs the workflow to exist first; the M1-measured baseline is provisional and 16d re-checks it against the runner. |
| Skip the manifest, trust the release tag | Simpler | A partial download or re-uploaded asset would silently shift every number; verification costs one hash pass over 106 MB. |

## 4. How it works

```python
worse_by = -delta if spec.direction == "higher" else delta
if   worse_by > spec.tolerance + _EPS:  status = "fail"
elif -worse_by > spec.tolerance + _EPS: status = "improved"   # still passes, flagged for rebaselining
else:                                    status = "pass"
```

`_EPS = 1e-9` because 0.44 − 0.42 computes as 0.020000000000000018; a drop of exactly the tolerance is the largest drop called noise and must pass. `evaluate` returns `passed = no problems and ≥1 check and no check failed`.

Fixture round trip, measured locally: export 16 s → `collection.snapshot.gz` 102.9 MB + `chunks.jsonl.gz` 3.9 MB (106 MB total); restore into a fresh key-less Qdrant v1.18.3 in 5.8 s wall including hash verification; restored `chunks.jsonl` is byte-identical to the original (`cmp`). Fingerprint of this export: `e99f16de52c7d00f`.

## 5. What the numbers said

Gate run on the restored snapshot, M1 Pro CPU, 150 dev positives, queries encoded live. Two full runs: **every metric identical to all digits**.

| Metric | Value | In the gate |
| --- | --- | --- |
| recall@10 | 0.4317 | yes |
| MRR | 0.8674 | yes |
| NDCG@10 | 0.7100 | yes |
| MAP@10 / precision@10 | 0.3721 / 0.6667 | recorded only |
| search p95 | 17.5 ms / 17.0 ms | recorded only |
| encode | 0.271 s per query (CPU, bge-large) | recorded only |

**Choosing the tolerance** needed a noise measurement, and the first guess was wrong. The same snapshot gave recall@10 0.4439 earlier (query vectors read from the embedding cache) and 0.4317 now (encoded live on CPU): the vectors differ by at most 3.4e-7 per component, median 2.1e-7, min cosine 0.99999982 — and that flips near-tied top-10 boundaries on about 20 of 150 queries. Against that, the batch-size sweep (`16d_encoder_noise.py`, batch 1/4/16/32/64, vectors differing by ≤1.5e-7) moved nothing: spread 0.0000 on all three metrics. So the largest benign deltas observed are **recall@10 0.0122, MRR 0.0034, NDCG@10 0.0089**.

Tolerance = 2× the largest observed benign delta, rounded up to 0.005: **recall 0.025, MRR 0.01, NDCG 0.02**. The recall band must stay below the plan's own example of a regression the gate exists to catch — an 8% relative recall drop = 0.035 — and it does. Checked by exercising `compare` with a synthetic 8% drop: recall 0.3972 (Δ −0.0345) → `FAIL`, exit 1; the identical run → pass, exit 0.

## 6. What surprised me / what to watch for

- **The metrics respond to 1e-7 perturbations, in a threshold-like way.** 1.5e-7 of noise changed nothing; 3.4e-7 changed 0.012 of recall. In a duplicate-heavy corpus with an inaccurate HNSW graph (16b), near-tied candidates sit at the top-10 boundary, so a cross-architecture (M1 → x86 runner) delta could land anywhere from 0 to at least 0.012. This is why the tolerance is stated as provisional.
- **The tolerance guess was wrong before the measurement.** I first set 0.02 on all three metrics from the live-vs-snapshot gap (0.0067); the cached-vs-live gap was nearly twice that. Reported as found, then corrected.
- **The baseline here is provisional.** It comes from the M1; 16d compares it with the hosted runner's first run and widens the band with data only if the runner delta demands it.
- **Everything here is in-sample for the golden set's quirks** (16b): a gate on this set protects against *changes*, not against the set's own bias.

## 7. Where this fits and what's next

16d: the `eval.yml` workflow — download the release asset, restore, run `retrieval_gate`, post the diff — which needs you to upload `ci-fixtures/` as a release (command in the chat). Then 16e (LLM-backed checks with a per-run cost measurement first, and the p95 latency gate), 16f (PR comment), and 16g (the deliberate regression, screenshotted).
