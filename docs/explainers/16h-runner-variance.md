# 16h — The Runner Is Not Numerically Constant

**Phase:** 16 · **Date:** 2026-10-24 · **Files:** `src/aegis/eval/gate.py` (`GateReport.info`, host line in `compare_files`), `src/aegis/eval/retrieval_gate.py` (`parse_cpuinfo`, `host_info`, query-vector hash), `tests/test_host_info.py`, `tests/test_gate.py`
**Topics from the course:** Retrieval evaluation metrics; Evaluation strategies. The practical lesson: an evaluation harness has to be treated as a measuring instrument with its own noise.

## 1. What I built

Provenance for every retrieval gate run: the host's CPU model, machine type and SIMD level (`avx`, `avx2`, `avx512f`), and a SHA-256 of the exact query vectors the encoder produced. It is stored in the report's non-gated `info` field and printed under the comparison table (so it also appears in the job summary and the PR comment). It never influences a verdict.

## 2. Why this approach

The recovery run of the revert test (run 40084720757) returned recall@10 0.431741731551329, MRR 0.867367724867725, NDCG@10 0.709988615910961, and **all five metrics were bit-identical to the M1 values** from 16c, not to the values the hosted runner had reproduced exactly on its previous three runs (0.4435 / 0.8640 / 0.7188; runs 40048175597, 40068902808, 40073021376). Same snapshot, same 150 queries, same code, same Qdrant image, same settings. What changed is not recorded anywhere, because no run recorded its hardware. That is the gap this unit closes: next time the numbers move, the report says whether the encoder's output changed (different vector hash) or only the search did (same hash, different metrics).

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Tighten the setup until the runner is deterministic (pin the CPU type) | Exact reproducibility | GitHub-hosted runners do not let you choose the CPU model; the only route is a self-hosted runner, which 16's own design rejected. |
| Re-baseline each time the numbers move | Always green | That turns the gate into a recorder: a regression and a hardware change would both be "absorbed". |
| Widen the bands | Fewer surprises | The bands were set from this exact kind of noise (16c, 2× the benign 0.0122) and held; widening would reduce sensitivity to real regressions (16g: a −0.04 recall drop is caught today). |
| Run the retrieval gate twice and take the worse | Smooths noise | Doubles the 10-minute cost to hide a measurement problem rather than see it. |
| **Record provenance, change nothing else (chosen)** | Diagnosis for free on every run | Cannot by itself prevent a false red; it makes one explainable. |

## 4. How it works

`parse_cpuinfo` reads `/proc/cpuinfo` text and returns the first `model name` and which of `avx`, `avx2`, `avx512f` appear in the first `flags` line; empty or odd input gives `("unknown", "")`, so provenance can never fail a run (3 tests). Off Linux it falls back to `platform.processor()`. `query_vectors_sha256` is the first 16 hex characters of the hash of the query-vector bytes. `GateReport.info` defaults to `{}`, so baselines written before it existed still load (tested), and `compare_files` prints the current run's `info` in a `<sub>` line.

## 5. What the numbers said

**Observed across five identical-input runner runs** (retrieval-gate, recall@10): 0.4435, 0.4435, 0.4435, (regression run, not comparable), **0.4317**. Two outcomes so far: A = 0.4435 / 0.8640 / 0.7188 (three runs, plus the cached-vector result of 16c: 0.4439 / 0.8640 / 0.7189) and B = 0.4317 / 0.8674 / 0.7100, bit-identical to the M1. The deltas between them are +0.0118 / −0.0033 / +0.0089, i.e. 47% / 33% / 45% of the bands, so the gate passed. Had the bands been 0.01 (a "tight" first guess) run 5 would have been a false red.

**A second, separately confirmed source: Qdrant's own search varies with its CPU limit.** On the M1, with the same live-encoded query vectors, an unconstrained Qdrant container gave 0.4317 / 0.8674 / 0.7100 (16c) and a `--cpus=2`, optimizer-off container gave 0.4344 / 0.8674 / 0.7118 (16d): +0.0027 recall, identical MRR, +0.0018 NDCG. The graph is the same in both; the search path (parallelism across segments, tie order among the 41.8% duplicate vectors) is not.

**What is not known:** which component produced runner outcome A versus B. Hardware difference is the leading suspect; I have no CPU data from those runs and so cannot say. The next runs will carry the CPU model, SIMD level and vector hash.

Verified locally: a real gate run on the M1 prints `machine: arm64 · cpu_model: arm · simd: · query_vectors_sha256: d8c00a8d4c279a23` under the table, and a baseline without `info` still compares. `uv run pytest -q`, ruff and mypy are clean.

## 6. What surprised me / what to watch for

- **I wrote "the runner is repeatable" in 16d and 16e, on three runs.** It is repeatable within a hardware class and not across them. Both records carry a correction note.
- **bit-identical to an ARM laptop.** Outcome B matching the M1 to 15 digits suggests the encoder and search numerics on that runner agreed with the M1 exactly; why a different architecture agrees with it while the other runs do not is unexplained.
- **The band absorbed it, but only just by design**: 47% of the recall band went to a hardware change. If a third numeric mode exists with a larger shift, the band could be exceeded with no code change.
- **Consequence for 16g:** the regression run was compared against outcome-A numbers. On outcome-B hardware its deltas would have been about 0.012 smaller in recall, still more than twice the band.

## 7. Where this fits and what's next

After a handful of runs the info lines tell us whether outcome A/B track the CPU model; if they do, the bands can be defended with that evidence, and if they don't the cause is elsewhere and is the next thing to chase. The remaining Phase 16 items are unchanged: generation-subset gate, escalation check, p95 gate.
