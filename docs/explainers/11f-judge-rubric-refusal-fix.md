# 11f — Judge Rubric Fix: Refusal-Scoring Divergence

**Phase:** 11 · **Date:** 2026-09-25, updated 2026-09-26 · **Files:** `src/aegis/eval/judge.py`
(`JUDGE_PROMPT_TEMPLATE`), `tests/test_judge.py`, `experiments/11g_judge_rubric_rescore.py`
**Topics from the course:** LLM-as-judge failure modes (leniency/ceiling-effect bias, instruction-adherence
failure) and rubric iteration — plan.md Phase 11 step 3's own framing: "iterate the rubric... until it
agrees with you"

## 1. What I built

A revised `JUDGE_PROMPT_TEMPLATE` targeting the two real, diagnosed failure modes from Unit 4
(`docs/results/judge-agreement.md`): `openai/gpt-oss-120b` penalizing honest refusals despite an existing
instruction, and Gemini's near-zero score variance. Same `HelpfulnessJudgment` shape (1-5 score + required
reasoning) — no changes needed anywhere else (`agreement.py`, `custom_metrics.py`, `scorecard.py` all
consume the same interface unchanged).

## 2. Why this approach

Both failure modes are prompt-structure problems, not model-capability problems — `gpt-oss-120b`'s own real
reasoning text ("does not provide any concrete guidance", "leaves the user's problem unresolved") shows it
correctly identified the answer as a refusal, then applied a task-completion heuristic that overrode the
one-sentence instruction sitting after the scale definition. The fix keeps the single-holistic-score rubric
shape (design record 11c already rejected multi-dimensional scoring, for reasons that still hold — adding
axes adds places for two judges to disagree over wording, not substance) and instead restructures *where*
and *how forcefully* the two problem instructions are stated, informed directly by the real override
language each judge used.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ----------------- |
| Switch to a different second judge model instead of fixing the prompt | Sidesteps whatever is specific to `gpt-oss-120b`'s training | The failure is a demonstrated instruction-adherence gap on a rubric the project controls, not a capability ceiling — a different model would need the same investment (measure it, might have its own biases) with no guarantee of a better outcome, and throws away Unit 4's real, already-diagnosed evidence about *this* model's specific failure shape. |
| Go multi-dimensional (separate grounding/clarity/tone scores) to force more discriminative output from Gemini | Might naturally break the ceiling effect by giving Gemini more axes to vary independently | Design record 11c already measured-through this tradeoff for the original rubric design and rejected it (more axes = more surface for disagreement over wording, not substance) — nothing about the ceiling-effect finding changes that argument; the ceiling effect is about score anchoring, not dimensionality. |
| Add explicit few-shot examples (a worked 2, a worked 4, a worked 5) instead of the "reserve 5, name what a 4 would be missing" anchoring instruction | Few-shot calibration is a standard, often stronger LLM-judge fix | Real corpus answers vary enough in shape (refusals, code-heavy, prose-only) that any single set of worked examples risks anchoring judges to *that example's* specific pattern rather than the general principle — the counterfactual-justification instruction ("what would a 4 version be missing") asks for the same discriminating behavior without hardcoding what a 4 looks like. Worth revisiting if the anchoring instruction alone doesn't move Gemini's real variance enough once re-tested. |

## 4. How it works

Two structural changes, both grounded in the real disagreement evidence:

1. **The honest-refusal rule moved from a trailing sentence to a `CRITICAL RULE` block at the very top**,
   before the numeric scale, and now explicitly names and preempts the override language `gpt-oss-120b`
   actually used: *"Do NOT score it low with reasoning like 'does not provide concrete guidance', 'leaves
   the problem unresolved', or 'offers no solution'..."* — quoting the real failure back at the model as
   exactly what not to do, rather than trusting a generic instruction to out-compete the model's default
   task-completion framing from a weaker position in the prompt.
2. **A ceiling-effect anchor**: *"Most real answers are NOT a 5 — reserve 5 for an answer you cannot think
   of any concrete way to improve... Before giving a 5, identify what a 4 version of this same answer would
   be missing; if you can't name anything, it's a 5."* Forces a counterfactual check before defaulting to
   the top score, targeting Gemini's real 10/11-answers-scored-5 pattern without adding rubric dimensions.

A third, smaller change: the reasoning field is now explicitly asked to state up front whether the answer
*is* a refusal case, making future disagreement review (and this fix's own verification) a grep instead of
a full re-read.

## 5. What the numbers said

**Update, 2026-09-26 — updated after a fuller re-score, see below.** Two real measurements now exist, at
increasing scale:

**Spot-check (4 disagreement + 3 non-refusal cases, `docs/results/judge-rubric-fix-spot-check.md`):** all 4
real refusal-disagreement cases moved from `[1, 2, 1, 1]` to `[5, 5, 5, 4]` — an exact match to Gemini's
original scores on the same 4 rows; 3 non-refusal cases were unchanged (`4→4`, `4→4`, `5→5`).

**Full re-score (26 of 29 real rows, groq side only, `docs/results/judge-rubric-fix-rescore.md`):**
confirms the refusal fix holds at fuller scale — all 5 real refusal cases among the 19 rows with both an
old and new score moved up, matching Gemini's stance. But it also surfaced something the small spot-check
didn't show: **mean moved 3.10→4.31 and the distribution compressed toward the top** (24/26 now score 4-5,
vs. 12/20 before) — a milder version of the same ceiling-effect shape Gemini already had, not the opposite
problem. Two non-refusal answers moved down (5→4, the anchor working as intended), but one moved up with no
clear refusal-related cause — and that case (`golden::issue::2884`) also revealed **real judge
non-determinism**: the identical (query, answer, context, prompt) at `temperature=0.0` scored 4 in the spot-
check and 5 in the full re-score.

**Still not confirmed**: Gemini's own side of the fix — 0 of 29 Gemini calls have succeeded under the new
rubric across two separate attempts (both quota-blocked), so real paired inter-judge κ under the fixed
rubric remains completely unmeasured. Whether the groq-side distribution compression helps or hurts real
agreement with Gemini is unknown until that data exists.

## 6. What surprised me / what to watch for

At small scale (the 4-case spot-check), the fix's effect looked binary — every refusal case jumping
straight to what Gemini had already been scoring it, with zero effect on 3 non-refusal cases. **The fuller
26-row re-score complicated that picture**: the refusal-specific correction still holds cleanly, but the
overall distribution also compressed toward the top of the scale, something invisible at N=7. A real
lesson for this project's own practice, not just this fix: a small, targeted spot-check is good for fast,
cheap confirmation of a specific hypothesis, but it can miss a broader side effect that only shows up at
scale — the same reason Unit 4 needed N=30 (not N=1) to find the original problem in the first place.

Also real: `gpt-oss-120b` is not perfectly deterministic across repeated calls at `temperature=0.0` on
identical input (`golden::issue::2884`: 4 in the spot-check, 5 in the re-score) — worth remembering when
interpreting any single judge score, or any single kappa measurement, as more than one noisy draw.

Verifying this fix at all cost real quota on both providers, on the same day quota had just reset — this
project's free-tier operating margin for *any* judge-related iteration is thin enough that even a targeted,
7-call spot-check exhausted `gpt-oss-120b`'s daily budget again, and the fuller 26-call re-score exhausted
it a second time the same day. Worth naming directly: real judge iteration on a free tier is not a "test,
adjust, retest same day" loop, it's closer to one real iteration per quota window.

## 7. Where this fits and what's next

The groq-side refusal fix is real and verified against the exact cases that motivated it, at both small and
fuller scale. What's still open: (a) Gemini's own side of the fix has zero live evidence after two separate
attempts (both quota-blocked) — real inter-judge κ under the fixed rubric is completely unmeasured; (b) the
groq-side distribution compression is a new, undiagnosed signal that needs Gemini's paired data to interpret
at all. `experiments/11g_judge_rubric_rescore.py` is checkpoint-resumable and just needs both providers'
quotas to allow the remaining calls — it won't re-pay for the 26 real groq scores already collected. That
re-run writes to a fresh raw-results file, not overwriting `judge-agreement-batch-raw.json` (Unit 4's real,
already-published artifact) — so today's negative result
stays reproducible and comparable to whatever the fixed rubric produces.
