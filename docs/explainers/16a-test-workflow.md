# 16a — The `test.yml` Workflow

**Phase:** 16 · **Date:** 2026-10-22 · **Files:** `.github/workflows/test.yml`, `pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`, dev deps), `tests/security/test_attack_tenant_bypass.py`, `src/aegis/chunking/semantic.py`
**Topics from the course:** Evaluation strategies (system vs. component, code-based vs. LLM-judge vs. human feedback) — this unit is the *code-based, zero-LLM* tier; the LLM-judged tier is `eval.yml` (16b+).

## 1. What I built

A workflow that runs on every push and PR: install exactly `uv.lock`, `ruff check`, `mypy` on the core modules, then the full pytest suite against a real Qdrant service container. No secrets are exposed to it and no test can reach a paid provider, so it costs $0 and can run as often as GitHub schedules it. It is the cheap tier of the gate; the evaluation gate proper is `eval.yml`.

## 2. Why this approach

Three constraints from this repo shaped it:

- **Tests are all fakes except three.** On a clean copy of the tree (no `.env`, no `data/processed/`, no `data/raw/`, empty Hugging Face cache, offline) 743 of 746 passed. So the unit job needs no model download and no corpus. That is what makes "run on every push" affordable.
- **The three that need infrastructure need different things.** `test_delta_indexing` needs a live Qdrant (the "deleted document becomes unretrievable" regression, the Air Canada failure mode as a test) — handled by a service container pinned to the same image tag as `docker-compose.yml`. The two `test_attack_tenant_bypass` real-corpus cases need `data/processed/chunks.jsonl`, which is gitignored (27 MB) — they now **skip with a printed reason** on a bare checkout, and **fail** if `AEGIS_REQUIRE_CORPUS=1` is set. `eval.yml` will set it, so the check cannot pass there by not running.
- **The repo had never been linted or type-checked**, so "turn on a linter" is really "choose what to enforce". The user chose a narrow gate (see §3).

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Ruff with its default rules | Zero config | The default set is version-dependent: ruff 0.17.1 reports **103** findings on this repo with defaults (RUF059 ×22, C408 ×7, SIM117, UP*…), **45** with the explicit `E4,E7,E9,F,I` list. A gate whose meaning changes on a version bump isn't a gate, so the rule list is explicit and ruff is pinned (`==0.17.1`). |
| `ruff format --check` | Uniform style | Would rewrite **142 of 186** files. That diff buries `git blame` on every file in a repo whose Phase 15 work is still uncommitted, and gates style, not behavior. |
| mypy on all of `src/` | Strongest signal | **64 errors in 22 files** (110 files checked), mostly overrides of third-party protocols (`ragas_adapters.py`) and optional-narrowing in LLM glue. A day or more of cleanup in code that is not on the evidentiary spine. |
| Ratchet (fail only if error count rises) | Cheap to adopt | Leaves 100+ known errors in place and needs baseline-file bookkeeping; no tool here does it natively for mypy. |
| mypy on the core modules **(chosen)** | Covers where a silent bug corrupts every measurement (CLAUDE.md §6) | 18 files: `analysis/`, `chunking/`, `retrieval/fusion.py`, `guardrails/{confidence,citations}.py`, `router/policy.py`, `eval/metrics.py`. Remainder goes to `docs/future-work.md`. |
| Tests only, no service container (skip the Qdrant test) | Simpler YAML | The one test that proves a deleted doc is unretrievable would never run in CI — exactly the test worth gating. |
| Pin actions by commit SHA | Supply-chain hardening | Pinned by tag (`checkout@v7` major tag, `setup-uv@v10.3.2` exact) for now; SHA-pinning is noted in future work. The workflow has `contents: read` only and no secrets, which bounds the exposure. |

New dependencies (CLAUDE.md §2 requires justification): `ruff==0.17.1` and `mypy==2.4.0`, dev group only. Neither is something worth hand-rolling; both are pinned because their defaults and error sets move between releases.

## 4. How it works

```yaml
services:
  qdrant: { image: qdrant/qdrant:v1.18.3, ports: ["6333:6333"] }   # == docker-compose.yml
steps:
  - uv sync --frozen            # exact uv.lock; drift is a failure, not a re-resolve
  - ruff check src tests        # [tool.ruff.lint] select = E4,E7,E9,F,I
  - mypy                        # files = the 18 core modules in pyproject.toml
  - curl /healthz loop          # service containers report "started", not "ready"
  - pytest -q -rs               # -rs prints every skip with its reason
```

Cheap checks run before the 18 s test run so a lint error doesn't wait behind it. The skip/fail switch for the corpus tests:

```python
if not (settings.paths.data_processed / "chunks.jsonl").exists():
    if os.environ.get("AEGIS_REQUIRE_CORPUS"):
        pytest.fail(...)          # eval.yml: missing corpus = red
    pytest.skip(...)              # test.yml: visible in -rs, not an error
```

The first pass under the project config found 36: 33 fixed by `ruff --fix` (import order, 9 unused imports, one placeholder-less f-string), and 3 by hand — an ambiguous `l` variable (`tests/test_pipeline.py`) and two mid-file imports (`tests/test_pipeline_stream.py`). One more import re-sort followed the mypy edit below. The single mypy error in the core set was `_MODEL_CACHE: dict[str, object]` in `chunking/semantic.py` (`.embed` called on `object`); it is `dict[str, Any]` now because fastembed is imported lazily and its type can't be named at module level.

## 5. What the numbers said

| Check | Result | Command |
| --- | --- | --- |
| Local full suite | 746 passed, 18.4 s | `uv run pytest -q` |
| Clean-checkout rehearsal (fresh `.venv`, no `.env`/`data/processed`, key-less throwaway Qdrant v1.18.3 container) | **744 passed, 2 skipped**, 109.15 s; ruff and mypy clean | the workflow's own steps, run by hand |
| Same, empty HF cache + `HF_HUB_OFFLINE=1` | 743 passed, 3 errors (the 3 infra tests, then fixed/skipped as above) | pytest |
| Gate discriminates | probe file with an unused import and `def f(x: int) -> str: return x` → ruff `F401`, mypy `return-value` (2 and 1 errors); probe deleted afterwards | ruff, mypy |

**First green run on GitHub** (run 39984838107, `ubuntu-24.04`, cold cache, 2026-10-23): **3 m 27 s** total; Install 57 s (190 packages, CUDA torch wheels included -- acceptable, no CPU-only torch needed), ruff 3 s, mypy 13 s, Tests 62 s (**744 passed, 2 skipped, 57.4 s**; the skips are the corpus tests, reason printed), post-job cache save 47 s. The hand rehearsal's 109 s test time was a local-venv artifact, not a property of the workflow. Not yet measured: a warm-cache run (the post-step saved the cache; the next push shows the saving).

## 6. What surprised me / what to watch for

- **The local rehearsal took 109 s; the hosted runner took 57 s for the same 744 tests.** I never isolated the local cause (cold caches in a fresh venv is the likely candidate, unverified). On the runner it is a non-issue; CI minutes are still the budget on a private repo.
- **ruff's defaults changed under me.** 103 vs 45 findings is the same code with different rules. That is the whole argument for the explicit list.
- **`known-first-party = ["aegis"]` changed the count**: 45 findings with the explicit list under `--isolated`, 36 once the project config (which includes it) was in effect. I did not isolate which files moved; I set it so `aegis` sorts as first-party consistently in `src/` and `tests/`.
- **CI only sees committed code.** 119 files are uncommitted at this moment, including all of Phase 15. The first push will lint and type-check code no human has looked at through CI before; expect the first run to find something the local run didn't (a platform difference, a dependency drift).

- **First real GitHub run (2026-10-23, run 39984201410) failed in 7 s, before any step ran:** `Unable to resolve action astral-sh/setup-uv@v10`. I had taken `v10` from the latest release name (`v10.3.2`); that repo publishes only exact version tags (v10.0.1 … v10.3.2), no floating `v10`, whereas `actions/checkout` does publish `v7`. `actionlint` and the local rehearsal cannot catch this (they never resolve action refs). Fixed by pinning `v10.3.2`. Lesson: the rehearsal covers the commands, not the workflow plumbing; only a real run does.

## 7. Where this fits and what's next

`test.yml` is the always-on, $0 floor. Next is 16b: what `eval.yml` can run on a hosted runner (restoring the index from `vectors_bge_large.npy` + `chunks.jsonl`, rebuilding BM25, measuring query-encode and rerank time on CPU first) and what thresholds the Phase 6 retrieval baseline justifies. The LLM-backed checks follow, capped at about $0.15 per PR run, with the real cost of one run measured and reported before anything is wired into a workflow.
