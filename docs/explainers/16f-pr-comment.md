# 16f — Metric Diffs as a PR Comment

**Phase:** 16 · **Date:** 2026-10-24 · **Files:** `src/aegis/eval/pr_comment.py`, `tests/test_pr_comment.py`, `report` job in `.github/workflows/eval.yml`
**Topics from the course:** Evaluation strategies; Logging, monitoring & observability (a gate nobody reads changes nothing).

## 1. What I built

A third job in `eval.yml`, `report`, that runs after both gates even when they fail (`if: always()`), and on same-repo PRs only. It downloads each gate's result tables, builds one markdown comment (a verdict headline, the retrieval and router tables diffed against their baselines, the commit, the run link and the OpenAI spend), and **creates or updates a single sticky comment** found by a hidden first-line marker, so a PR keeps one comment however many times it is pushed to.

## 2. Why this approach

- **The failing gate is when the comment matters.** `needs:` alone skips a job when a dependency fails; `always()` is what makes the table of a red run reach the reviewer.
- **The builder is stdlib-only and run by path**, without `uv sync`. The report must not be able to fail because the project's dependencies did; a missing numpy must never hide a failing recall.
- **The verdict is stated, not assumed.** A gate that was skipped (fork PR has no secrets), cancelled, or crashed before writing its table is reported as exactly that. Only real failures are red; "1 gate not run" is disclosed in the headline rather than folded into "passed".

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| A marketplace "sticky PR comment" action | Zero code | Another third-party action with `pull-requests: write` on a repo that holds an API-key secret; 30 lines of `gh api` do the same. |
| A new comment on every push | Simplest | Ten pushes, ten comments; the diff you want is the latest. |
| `gh pr comment --edit-last` | One command | Edits the last comment by the same user, not the one with our marker; a bot comment from another tool would be overwritten. |
| Job summary only | Already works | Nobody opens the Actions tab on a PR; it is where the numbers are, not where reviewers look. |
| Comment on fork PRs too | Coverage | The token is read-only there; posting would fail. Numbers still appear in the job summaries. |

## 4. How it works

Verdict logic, in order: any `failure` → "❌ Evaluation gate failed: <names>"; else any `cancelled` → "⚠️ incomplete"; else any `skipped` → "✅ No gate failed (N gates not run)"; else "✅ All evaluation gates passed". A gate with no table gets a one-line explanation ("Gate failure; no result was produced"). Sections are capped at 25,000 characters and the body at 60,000 (GitHub's limit is 65,536), with an explicit truncation notice. Spend is the sum of `cost_usd` in the run's uploaded `ci_spend.sqlite`.

The update step finds the comment with `startswith("<!-- aegis-eval-gate -->")` and PATCHes it, else POSTs. `startswith`, not `contains`: a human quoting the marker in a reply must not be overwritten.

## 5. What the numbers said

Nothing is measured here beyond correctness. 8 tests pin the verdicts (all-green, failed gate, failure with no table, skipped, cancelled, failure beating cancelled), the size cap and the spend read. A comment rendered from the artifacts of the real green run 40068902808 shows both tables at Δ +0.0000 and "OpenAI spend this run: $0.0154". The jq filter was checked against sample comment JSON: it selects the bot comment (id 222) and ignores both an unrelated comment and a human comment that quotes the marker. **Not yet verified:** posting to a real PR. That needs a PR; the first one is the next step.

## 6. What surprised me / what to watch for

- **Artifacts from a gate that died early don't exist**, and `download-artifact` fails when it can't find one. Both downloads are `continue-on-error`; without that, the report would vanish exactly when a gate crashed.
- **`pull_request` runs the workflow from the merge ref**, so the `report` job only exists for PRs opened after the workflow reached `main`.
- **PR-supplied text never reaches a shell here.** Only the PR number and head SHA (fixed formats) go through `env:`; titles and branch names are not used.

## 7. Where this fits and what's next

Next: a PR to watch the comment appear and update in place, then 16g, the deliberate regression on a branch, where the comment shows the red table. The remaining LLM tiers (generation subset, escalation, p95) follow; the comment already has the structure to take more sections.
