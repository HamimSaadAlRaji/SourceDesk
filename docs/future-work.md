# Future work

Ideas that are not in `plan.md` and therefore are not built (CLAUDE.md §9).

## Tooling / CI

- **Widen the mypy gate beyond the 18 core modules.** Full `src/` is 64 errors in 22 files (ruff 0.17.1 / mypy 2.4.0, 2026-10-22), mostly third-party protocol overrides in `eval/ragas_adapters.py` and `Optional` narrowing in the LLM glue (`eval/judge.py`). Added to the gate module by module as each is touched. Origin: design record 16a.
- **`ruff format` across the repo.** Would rewrite 142 of 186 files; do it as a single dedicated commit once no feature work is uncommitted, then add `ruff format --check` to `test.yml`.
- **Wider ruff rule set** (`RUF`, `UP`, `SIM`, `C4`): 103 findings under ruff 0.17.1's defaults vs. 45 under the current explicit list.
- **Pin GitHub Actions by commit SHA** instead of major tag (currently `actions/checkout@v7`, `astral-sh/setup-uv@v10.3.2`).

## Retrieval / evaluation validity (found in Phase 16, design record 16b)

- **The served HNSW graph returns only ~67% of the true cosine top-10** (ID overlap 0.671, score recall 0.682 vs exact, 150 dev queries, 2026-10-23), unchanged by `ef_search` (Phase 3 sweep). Hypothesis, untested: 41.8% of chunks are in exact-duplicate groups, which can make a node's neighbor list all copies of itself. Test: deduplicate before indexing and re-measure ANN recall.
- **The golden set's labels came from this specific index's candidates**, so exact cosine (the better retriever) scores recall@10 0.320 vs 0.437 live. Re-labeling against exact search would change every retrieval number in the project; for Phase 19's failure analysis to discuss, not to fix inside Phase 16.

## CI gate sensitivity (found in Phase 16, design record 16j)

- **The generation gate only sees gross regressions.** Four identical runs gave `valid_citation_rate` 0.6744 / 0.7339 / 0.8488 / 0.8710 (spread 0.197), so its band is 0.25. Tightening needs more samples: N = 60 would cut the spread by about 1.7x for ~$0.10 per run. Not done; the retrieval and router gates are much sharper.
- **No end-to-end p95 gate.** The 3 s target is unmet and unmeasured on the current stack (5.6 s from the qwen era); the confidence scorer's CPU cost makes any CI number a measurement of the runner. Needs a local Phase 13 re-run on gpt-5-mini.
- **Rewrite and entailment are unmeasured on gpt-4.1-mini** (16e); the generation gate's entailment calls are the first live use, with 0 unparseable verdicts over 48 calls (four runs).

## Escalation check follow-ups (Phase 16, design record 16i)

- **Verify the margin scores removed the CI miss.** The first CI run had 1 uncached pair (539/540 hits) on the AMD EPYC runner; the 347 margin pairs were scored afterwards (cache = 885 scores). If the next CI run still shows a miss, `rerank_cache_missed_keys` in the escalation job summary names it: look the key up in `ci-fixtures/rerank_pairs_export.json`, and if absent, add its (query, chunk) to the cache by running the gate once with `--save-cache`.
- **Cache refresh is manual.** After a deliberate reranker or chunking change, run the gate with `--save-cache --max-uncached 1000` and re-baseline; there is no automation, by design (a silent refresh would defeat the check).
- **tau itself is unchanged and unimproved**: 3 of 30 negatives still pass at tau 0.5; the gate pins it. Improving it is Phase 19 failure-analysis work.

## Orchestration (Phase 17, design records 17d-17f)

- **Graph path as a CI matrix leg** (user decision 2026-10-25: deferred). Run the pytest suite with `AEGIS_TEST_BACKEND=langgraph` as a second job; the suite is 862 tests / ~18 s locally, ~1 min on the runner. The four eval gates call components, not the orchestration, so they would not change. Cost: roughly 3 extra CI minutes per push against the 2,000-minute budget.
- **The end-to-end p95 is now measured: 11.8 s direct (n = 40, 35 on the generate path, 2026-10-25), against a 3 s target.** The Phase 13 attribution no longer describes the stack. On the generate path generation is 53% of mean time (p95 7.1 s, gpt-5-mini) and validation is 9% (p95 2.7 s: citation entailment is now an LLM call, 3 ms in Phase 13). A p95 gate is still not buildable in CI (confidence scorer CPU cost), so this needs a local fix, not a gate. Candidates, untested: streaming already hides generation time from the user; a lighter or non-reasoning generator; skipping tier-2 entailment when lexical overlap already passes.
- **The retry-on-low-confidence edge has only been exercised by unit tests.** On real data it needs reranker pairs outside the 885-pair cache (the 17d attempt was refused at 63 > 60 uncached pairs, ~150 pairs / ~3 min CPU in total), plus real reformulations (paid). Phase 8 measured the same idea as a retriever wrapper at -13.4% NDCG@5; whether a confidence-triggered version does better is untested.
- **Equivalence was proven under replay, not live-vs-live.** Two live runs would differ by model noise alone (Open Question #4); a repeated live comparison with a noise band would say how large that is for the end-to-end decisions.
