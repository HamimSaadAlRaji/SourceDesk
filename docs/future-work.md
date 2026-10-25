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
