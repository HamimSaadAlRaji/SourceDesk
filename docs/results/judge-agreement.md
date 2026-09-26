# Judge Agreement — Unit 4 (Phase 11)

**Methodology.** plan.md Phase 11 step 3 asks for judge-vs-hand-label Cohen's κ; the user declined to
hand-label any answers (2026-09-25, progress.md Decision Log), so this project measures **inter-judge
agreement** instead — two independent LLM judges (`eval/judge.py`, Unit 3) scoring the same generated
answers, agreement computed with `eval/agreement.cohen_kappa` (Unit 2). This is a real, disclosed
substitution, not an equivalent one: it answers "do two judges agree with each other," not "does a judge
agree with a human." Treat every number below with that limitation in mind.

N=30 dev-positive golden queries (`random_seed=20260802`), real retrieval (Phase 6 winning config), real
Groq generation (`qwen/qwen3.8-27b`), real RAGAS scoring, real judge scoring by both `gemini-3.5-flash`
and `openai/gpt-oss-120b`. Script: `experiments/11e_judge_validation_batch.py`. Raw data:
`docs/results/judge-agreement-batch-raw.json`.

## The headline result

| | value |
|---|---|
| Cohen's κ (unweighted) | **-0.0161** |
| Cohen's κ (quadratic-weighted) | **0.0906** |
| Raw agreement rate | 0.2222 |
| N paired (both judges succeeded) | 9 / 29 |
| plan.md's bar | ≥ 0.6 |

Both κ values are at or below chance level. **plan.md's own framing applies directly: "If κ < 0.6 the
judge is unreliable and every metric built on it is decoration."** This is a real, honest negative
result, not a bug to explain away — see "What this means" below for why it's actually a valuable one.

## Why: two distinct, real judge failure modes, not just noise

### 1. Gemini has almost no discriminative variance

| judge | n succeeded | score distribution | mean |
|---|---|---|---|
| `gemini-3.5-flash` | 11 | five 5's, one 4 — nothing else | 4.91 |
| `openai/gpt-oss-120b` | 20 | 4×9, 1×5, 2×3, 5×3 | 3.10 |

Gemini scored 10 of 11 real answers a perfect 5/5. A judge whose scores barely move regardless of input
carries very little information — mechanically, Cohen's κ cannot register real agreement against a rater
whose marginal distribution is nearly degenerate, independent of whatever the "true" quality differences
are. This is a real, distinct LLM-judge failure mode (a leniency/ceiling-effect bias), not one of the
three plan.md names as examples (position bias, verbosity bias, self-preference) but the same category of
problem.

### 2. The two judges disagree systematically, not randomly, on honest refusals

All 4 of the paired disagreements with `|diff| >= 2` are the same case: the generated answer is an honest
"the context doesn't contain enough information to answer this" refusal.

| query | groq (`gpt-oss-120b`) | gemini | groq's reasoning (abridged) | gemini's reasoning (abridged) |
|---|---|---|---|---|
| `golden::issue::3212` | 1 | 5 | "does not help the user solve their problem" | "this honest refusal is the most helpful and accurate response possible" |
| `golden::issue::12965` | 2 | 5 | "does not provide any concrete guidance... leaves the user's [problem unresolved]" | "avoids making unsupported assumptions, which is the most helpful response given the limited context" |
| `golden::issue::15764` | 1 | 5 | "offers no solution, explanation, or guidance... fails to help the user resolve their issue" | "correctly and honestly identifies that the provided context does not contain information" |
| `golden::issue::5857` | 1 | 4 | (garbled — judge output referenced "the evaluator's request" rather than the actual answer, a real judge-quality issue on its own) | "the assistant correctly states that the context does not contain enough info[rmation]" |

The rubric (`JUDGE_PROMPT_TEMPLATE`, design record 11c) explicitly instructs: *"An answer that correctly
says it doesn't have enough information to answer, when the context genuinely lacks the answer, should
score high (4-5), not low."* Gemini visibly follows this instruction every time. `gpt-oss-120b` visibly
does not — it scores every one of these refusals low on pure task-completion grounds, effectively
overriding the explicit instruction.

**This is the more consequential of the two failure modes.** If `gpt-oss-120b` were used alone to
score or tune this system, its scores would create a direct incentive to hallucinate a confident answer
instead of escalating — the exact Air Canada failure mode this entire project is built to prevent, being
actively rewarded by the metric meant to catch it.

## Secondary results: RAGAS metrics (fuller coverage, not judge-quota-limited)

| metric | n | mean |
|---|---|---|
| `faithfulness` | 20/29 | 0.7977 |
| `response_relevancy` | 27/29 | 0.5662 |
| `id_based_context_precision` | 29/29 | 0.5828 |
| `id_based_context_recall` | 29/29 | 0.4103 |

`faithfulness` (plan.md's exit criterion: ≥ 0.90) is real and below target. Not chased further this
session — a real root-cause pass (is this a generation-grounding problem, a RAGAS-scoring-model problem,
or a golden-set-ground-truth-permissiveness problem, per Open Question #8) is real, separate work, and
every model used to measure it today is now quota-exhausted. Flagged as open, not silently accepted.

## What actually broke: every model used for judging hit a real, hard quota wall

| model | role | what happened |
|---|---|---|
| `qwen/qwen3.8-27b` | generation | Real input-tokens/minute (ITPM) cap hit once (7000 limit) — a *new* constraint, on the input side, not the output-side OTPM cap found in Unit 1 |
| `openai/gpt-oss-120b` | judge | Real 200,000 tokens/day (TPD) cap, ~99% consumed by query ~10 of 30 (cumulative with this session's earlier testing) |
| `openai/gpt-oss-20b` | RAGAS scoring | Same 200,000 TPD cap, independently exhausted by query ~29 of 30 |
| `gemini-3.5-flash` | judge (+ generation fallback) | Real 20-requests/day cap, fully exhausted partway through the batch |

One query (`golden::issue::2600`) failed generation entirely — both Groq (ITPM wall) and Gemini (daily
quota) were down simultaneously at that moment, a real demonstration of `FallbackLLMClient`'s fallback
boundary being necessary but not sufficient when *both* providers are constrained at once.

## What this means

The inter-judge-agreement substitution (forced by the no-hand-labeling decision) turned out to be more
informative than a clean pass/fail would have been. A single judge's raw score, taken at face value,
would have looked like a normal eval number — nothing here would have looked wrong from either judge
alone. It's only the *comparison* that surfaces that neither judge is trustworthy on its own: one has
almost no discriminative range, the other actively penalizes the exact safe behavior this system is
designed to produce. This is close to word-for-word the scenario plan.md's design-record ask anticipates
("where the judge disagreed with you and who was right") — here, arguably, *neither* judge was fully
right, and knowing that is the real deliverable.

## Not yet done (next session's first task)

- Iterate the judge rubric — likely strengthen or reposition the honest-refusal instruction, and consider
  a lower-variance-inducing rubric structure for Gemini specifically — then re-run once quota resets.
- A root-cause pass on `faithfulness=0.7977` (below the 0.90 target).
- Every model used here is quota-exhausted as of this session; nothing above can be re-tested until reset.
