# 16d — The `eval.yml` Workflow and the Three Runs It Took

**Phase:** 16 · **Date:** 2026-10-24 · **Files:** `.github/workflows/eval.yml`, `src/aegis/eval/ci_fixtures.py` (`wait_until_restored`), `tests/test_ci_fixtures.py`, `ci/baselines/retrieval.json`, release asset `ci-fixtures-v1`
**Topics from the course:** Evaluation strategies (system vs. component); Retrieval evaluation metrics; HNSW (`m`, `ef_construct`, and now the optimizer's effect on the graph).

## 1. What I built

A workflow that, on every PR, every push to `main`, and on demand: starts a Qdrant service, downloads the `ci-fixtures-v1` release, verifies its hashes, restores the snapshot, runs the 31 tenant-bypass tests on the real corpus (failing, not skipping, if the corpus is missing), scores dense retrieval on the 150 dev queries, and fails the build if recall@10, MRR or NDCG@10 leave their tolerance band around `ci/baselines/retrieval.json`. The comparison table goes to the job summary and the raw results are uploaded as an artifact. Branch pushes run only `test.yml`.

## 2. Why this approach

The restore and gate were rehearsed locally in 16b/16c. Nothing local could rehearse a 2-core x86 runner, and that gap is where two real defects were hiding. The three runs below are the evidence.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Accept `yellow` as ready (first fix idea) | One-line change | Yellow right after restore meant the optimizer was *rewriting* the index. Accepting it measured a different graph (run 2: recall@10 0.2687). |
| Wait for `green` (what I shipped in run 2) | Correct for a normal collection | With the optimizer already rebuilding, green arrives after the graph has changed. Right check, wrong layer. |
| Loosen the tolerance until run 2 passes | Green build | The tolerance is for noise (≤0.012); the failure was 0.16 recall. Widening it would have hidden a different index. |
| Rebuild HNSW and re-baseline on the runner | Matches whatever the runner builds | Unreproducible build (16b: two rebuilds agree on 54% of top-10s). |
| **Disable the optimizer on the CI Qdrant (chosen)** | Preserves the snapshot's graph exactly | One env var on the service; restore refuses to run without it. |
| Eval on every branch push | Faster feedback | ~11 min of runner time per push against 2,000 free minutes a month. |

## 4. How it works

`wait_until_restored` accepts the collection only when **all** hold, for 3 consecutive polls:

1. `max_optimization_threads == 0` (checked first — otherwise the opening poll can see the original 4 segments just before a merge begins);
2. `points_count == indexed_vectors_count == 19,236`;
3. `segments_count == manifest["segments_count"]` (4, recorded at export);
4. `optimizer_status == "ok"`; `red` fails at once.

With the optimizer disabled the status stays `yellow` permanently (merges pending that can never run), so "green" cannot be the criterion. The service sets `QDRANT__STORAGE__OPTIMIZERS__MAX_OPTIMIZATION_THREADS: "0"`; the manifest's fingerprint is unchanged because it derives only from the two file hashes.

## 5. What the numbers said

| Run | Result | What it showed |
| --- | --- | --- |
| 1 (2 m 24 s) | Restore failed | `status=yellow points=19236, expected green` — the check demanded green; instant on an M1, not on 2 cores |
| 2 (10 m 0 s) | Compare **failed**: recall@10 **0.2687**, MRR **0.5842**, NDCG@10 **0.4496** vs 0.4317 / 0.8674 / 0.7100 | Same fingerprint, same queries, same code. Qdrant had merged the segments on the 2-CPU host (yellow ~21 s; segments 4 → 5 → 2) and rebuilt HNSW |
| local, `--cpus=2`, optimizer on | 0.2859 / 0.6306 / 0.4842, segments 4 → 5 → 2 | Reproduced the failure on the M1 |
| local, `--cpus=2`, optimizer off | **0.4344 / 0.8674 / 0.7118**, segments stay 4 | The fix, within 0.003 of baseline |
| 3 (11 m 2 s) | **Green**: 0.4435 / 0.8640 / 0.7188 | x86 runner vs M1 baseline: **+0.0118 / −0.0033 / +0.0089**, i.e. 47% / 33% / 45% of the 0.025 / 0.010 / 0.020 bands |

The runner's delta equals the benign delta measured in 16c (0.0122 / 0.0034 / 0.0089) from a different cause (cached vs. live vectors on the same machine), so the bands set from that evidence stand: no widening was needed. The baseline was then **re-taken from the runner** (artifact of run 3, same fingerprint `e99f16de52c7d00f`, 150 queries): a baseline measured where the gate runs leaves the whole band for real change instead of spending ~half of it on platform offset.

Runner costs (run 3): restore 12 s, tenant-bypass tests 31 passed in 10.3 s, retrieval step **9 m 3 s** (encode 2.87 s/query ≈ 430 s; the M1 does 0.27 s), search p95 13.9 ms. Run 2's retrieval step took 7 m 41 s at 2.43 s/query, so the runner's encode speed varies about 18% between runs. The job saved the Hugging Face model cache this time; run 4 is the first to restore it.

## 6. What surprised me / what to watch for

- **A database optimizer can silently change what you are measuring.** A restored snapshot is not the same artifact as its source if the server is allowed to "help". Locally on 8 cores nothing happened, which is why 16b/16c looked fine.
- **Each fix was right for the symptom and wrong for the cause.** Run 1 → "wait for green" → run 2 showed green was reached *after* the rebuild. Reading the segment counts, not just the status, found it.
- **Unmeasured:** the gate has not yet been run twice on the runner on identical inputs. Local runs are exactly repeatable (16c); the runner's repeatability will show on the next run, which compares against the runner-taken baseline. If two runner runs differ, the band absorbs it but the cause would need finding.
- **CI minutes.** `test.yml` ≈ 3 min plus `eval.yml` ≈ 11 min per PR push is ~14 min; 2,000 free minutes a month is about 140 PR pushes. Encoding dominates and cannot use the embedding cache without hiding encoder regressions (16c).
- **A local developer running the gate needs the same Qdrant setting**, e.g. `docker run -e QDRANT__STORAGE__OPTIMIZERS__MAX_OPTIMIZATION_THREADS=0 …`; the restore will say so if they forget.

## 7. Where this fits and what's next

The retrieval half of the gate is live and green. Remaining Phase 16 work: 16e — the LLM-backed checks (router confusion matrix, escalation on negatives, a generation subset) with the real cost of one run measured before wiring, plus the p95 latency gate; 16f — results as a PR comment; 16g — the deliberate regression, screenshotted. Open design question for 16e: the plan's "p95 latency budget" gate needs an end-to-end number, and the Phase 13 p95 (5.6 s against the 3 s target) is still unmet, so the latency gate can only guard against getting worse, not assert the target is met.
