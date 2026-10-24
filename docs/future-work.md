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
