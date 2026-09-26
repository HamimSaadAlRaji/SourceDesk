# Judge Rubric Fix — Full Re-score (N=26 Groq-side, Gemini still blocked)

Real re-score of Unit 4's already-generated 29 real answers under the fixed `JUDGE_PROMPT_TEMPLATE`
(design record 11f), using `experiments/11g_judge_rubric_rescore.py` — generation and RAGAS reused
unchanged, only the two judges re-run. Raw: `docs/results/judge-agreement-batch-raw-rubric-fix.json`.

**Groq (`gpt-oss-120b`): 26/29 succeeded** (3 hit its 200,000 TPD cap again, re-exhausted by this very
re-score). **Gemini: 0/29 succeeded** — its 20-requests/day cap did not clear during this run despite
resetting once earlier the same session (docs/results/judge-rubric-fix-spot-check.md); no paired data
exists yet, so real inter-judge kappa under the fixed rubric is still not computable.

## The refusal-specific fix: confirmed at fuller scale

All 5 refusal-like answers among the 19 rows with both an old and new groq score moved up, matching or
converging toward Gemini's original stance on the same rows:

| query | old groq | new groq |
|---|---|---|
| `golden::issue::3212` | 1 | 5 |
| `golden::issue::753` | 1 | 5 |
| `golden::issue::15764` | 1 | 5 |
| `golden::issue::1767` | 2 | 5 |
| `golden::issue::12965` | 2 | 4 |

Consistent with the spot-check (`judge-rubric-fix-spot-check.md`) and the diagnosed root cause: the
targeted fix works on every real refusal case it was built for.

## A new, real observation: the fix also raised non-refusal scores overall

| | old | new |
|---|---|---|
| n | 20 | 26 |
| mean | 3.10 | 4.31 |
| distribution | `{4:9, 1:5, 5:3, 2:3}` | `{4:12, 5:12, 3:1, 1:1}` |

24 of 26 new scores are 4 or 5 — a real compression toward the top of the scale, not just on the 5 refusal
cases. Two non-refusal answers moved *down* (`golden::issue::760`: 5->4, `golden::issue::58`: 5->4) --
real evidence the "reserve 5, name what a 4 version would be missing" anchor is doing its intended job of
demanding a concrete justification before the top score, not just inflating everything. But at least one
non-refusal answer moved up without an obvious refusal-related explanation (`golden::issue::2884`: 4->5).

**That last case is also a real non-determinism finding, not just an ambiguous score change**: the earlier
spot-check (`judge-rubric-fix-spot-check.md`) scored this exact same (query, answer, context) triple under
the identical new prompt and got **4->4 (unchanged)**. This run, same inputs, same model, same
`temperature=0.0` (`JudgeSettings.temperature`), got **4->5**. `gpt-oss-120b` is not perfectly
deterministic across repeated real calls on identical input, even at temperature 0 -- a real, disclosed
limitation for interpreting any single kappa measurement as if it were a fixed ground truth rather than one
draw from a distribution.

## What this means, honestly

The refusal-penalizing bug is fixed — confirmed on every real case that exposed it, at N=26 not just the
original 4-case spot-check. But the rubric fix has a real, disclosed side effect worth watching: overall
groq-side scores are now more concentrated at the top of the scale than before (24/26 at 4-5, vs. 12/20
previously) — a milder version of the same ceiling-effect shape Gemini already had, not the opposite
problem. Whether this actually helps or hurts real inter-judge kappa is not yet knowable: if both judges
converge toward "usually a 4 or 5," kappa could still be low (both judges now have compressed variance,
not just one), or agreement could improve if they're now converging on the *same* cases. This can only be
answered with real paired data from both judges, which still doesn't exist.

## Next

Both providers are quota-exhausted again as of this run (Groq `gpt-oss-120b`: re-hit its TPD cap; Gemini:
0/29, still fully blocked). No further live judge testing is possible today. Next session: re-run
`experiments/11g_judge_rubric_rescore.py` again once both reset — it's checkpoint-resumable, so it will
only fill in the missing `judge_gemini` scores (and the 3 missing `judge_groq` rows) rather than re-paying
for the 26 real groq scores already collected.
