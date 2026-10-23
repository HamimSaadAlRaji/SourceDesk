# Future work

Ideas that are not in `plan.md` and therefore are not built (CLAUDE.md §9).

## Tooling / CI

- **Widen the mypy gate beyond the 18 core modules.** Full `src/` is 64 errors in 22 files (ruff 0.17.1 / mypy 2.4.0, 2026-10-22), mostly third-party protocol overrides in `eval/ragas_adapters.py` and `Optional` narrowing in the LLM glue (`eval/judge.py`). Added to the gate module by module as each is touched. Origin: design record 16a.
- **`ruff format` across the repo.** Would rewrite 142 of 186 files; do it as a single dedicated commit once no feature work is uncommitted, then add `ruff format --check` to `test.yml`.
- **Wider ruff rule set** (`RUF`, `UP`, `SIM`, `C4`): 103 findings under ruff 0.17.1's defaults vs. 45 under the current explicit list.
- **Pin GitHub Actions by commit SHA** instead of major tag (currently `actions/checkout@v7`, `astral-sh/setup-uv@v10.3.2`).
