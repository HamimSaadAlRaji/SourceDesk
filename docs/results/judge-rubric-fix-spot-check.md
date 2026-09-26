# Judge Rubric Fix — Real Spot-Check (Phase 11, post-Unit 6)

Real, live re-scoring of the fixed `JUDGE_PROMPT_TEMPLATE` (`src/aegis/eval/judge.py`) against the exact
answers already on disk from Unit 4's batch (`docs/results/judge-agreement-batch-raw.json`) — not a fresh
generation run, just the judges re-run on the same real (query, answer, context) triples with the new
rubric. Groq (`openai/gpt-oss-120b`) quota allowed 7 real calls before hitting its 200,000 TPD cap again
(199,157/200,000 used); Gemini's 20-requests/day cap (cumulative across the whole session, including
earlier quota-check calls) allowed exactly 1 before `429 RESOURCE_EXHAUSTED`.

## The 4 real disagreement cases (all honest refusals, |old diff| >= 2)

| query | old groq | old gemini | **new groq** | reasoning excerpt |
|---|---|---|---|---|
| `golden::issue::3212` | 1 | 5 | **5** | "According to the critical rule..." |
| `golden::issue::12965` | 2 | 5 | **5** | "...aligns with the critical rule that an honest refusal is a correct, h[elpful answer]" |
| `golden::issue::15764` | 1 | 5 | **5** | correctly identifies the context lacks the specific regression info |
| `golden::issue::5857` | 1 | 4 | **4** | "According to the critical rule, such a refusal is considered helpful and should be scored 4 or 5" |

New groq scores: `[5, 5, 5, 4]`. Old Gemini scores on the same 4 rows: `[5, 5, 5, 4]` — **exact match**,
where the old rubric produced `[1, 2, 1, 1]` from groq against the same 4 answers. Real, direct evidence
the targeted fix (moving the honest-refusal rule to a CRITICAL, pre-scale position and explicitly naming
the override language the model had been using) worked on the exact failure mode it targeted. Gemini
itself was not re-tested on these 4 (quota) — no evidence yet on whether the ceiling-effect fix changed its
side.

## Non-refusal cases (checking the fix didn't overcorrect toward blanket high scores)

| query | old groq | new groq |
|---|---|---|
| `golden::issue::2689` | 4 | 4 |
| `golden::issue::2884` | 4 | 4 |
| `golden::issue::219` | 5 | 5 |

All three unchanged — the fix didn't inflate scores on answers outside the refusal case it targeted.
2 more non-refusal rows were queued but hit `openai/gpt-oss-120b`'s TPD wall before running.

## What this does and doesn't confirm

**Confirmed, real:** the specific, diagnosed groq-side failure (penalizing honest refusals despite an
existing-but-buried instruction) is fixed on the 4 real cases that exposed it, without disturbing 3 real
non-refusal scores.

**Not yet confirmed:** whether Gemini's ceiling-effect anchoring instruction ("reserve 5...") changes its
real scoring behavior — only 1 Gemini call succeeded before its daily quota (cumulative across this whole
session) ran out. Whether the fix holds at N=30 scale and produces a real κ >= 0.6 — needs a full batch
re-run, itself now blocked again on both providers' quotas (groq's `gpt-oss-120b`: ~33 min TPD partial
refill per its own error message; Gemini: next daily reset).

## Next

Re-run `experiments/11e_judge_validation_batch.py` against a **fresh** raw-results file (not overwriting
`judge-agreement-batch-raw.json`, which is Unit 4's real shipped artifact) once both providers' quotas
allow it, to get a real N=30 (or as far as quota allows) κ number under the fixed rubric.
