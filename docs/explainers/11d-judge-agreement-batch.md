# 11d — Judge Agreement Batch (Unit 4)

**Phase:** 11 · **Date:** 2026-09-25 · **Files:** `experiments/11e_judge_validation_batch.py`,
`src/aegis/config.py` (`JudgeSettings.ragas_llm_model` split), `src/aegis/eval/ragas_adapters.py`
(`GroqRagasLLM.max_tokens`)
**Topics from the course:** Evaluation strategies (system vs. component, code-based vs. LLM-judge vs.
human feedback); Evaluating the LLM's performance (RAGAS: Response Relevancy, Faithfulness); Choosing an
LLM & reading benchmarks

## 1. What I built

`experiments/11e_judge_validation_batch.py` — the real N=30 batch: generation → RAGAS (Faithfulness,
ResponseRelevancy, IDBasedContextPrecision/Recall) → both judges (`eval/judge.py`, Unit 3) → inter-judge
Cohen's κ (`eval/agreement.py`, Unit 2). Field-level resumable by design (each query's row is checkpointed
after every sub-step; a re-run only computes what's missing or errored) — not a nice-to-have, a direct
response to the real quota risk flagged before this ran (design record 11c §7). That risk materialized in
full; see §5.

## 2. Why this approach

The resumable design is the one decision here worth defending on its own. A non-resumable script that
re-runs everything from scratch on any failure would have been simpler to write, but every real external
constraint this project has hit in Phase 11 (Groq model retirement, Gemini transient 503s, now four
separate real quota walls in this unit alone) has been the *same* shape: partial, recoverable failure
under a scarce, shared, real-money-adjacent resource. Re-spending already-successful LLM calls to retry
one failed field would have been a real, avoidable cost, not just an engineering inconvenience — this
project's whole free-tier operating model depends on not doing that.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ----------------- |
| Non-resumable, whole-query retry on any failure | Simpler script, fewer edge cases | Real quota is the scarcest resource in this phase (four separate hard walls hit in one run) — re-spending it to retry one already-succeeded field is the exact waste the design specifically exists to avoid |
| Wait out the ~18-24 minute quota-refill windows before relaunching | Fewer failed attempts logged, marginally cleaner run | User's explicit call (`AskUserQuestion`, "resume now, accept stalls") — the resumable design makes the cost of relaunching immediately low (fast-failing 429s, not long hangs), so waiting idle traded real time for a benefit that didn't materialize: the walls were hit again regardless, because "already used 98.8% of a shared daily bucket" doesn't meaningfully un-happen in 20 minutes |
| Route the RAGAS-scoring role and the judge role onto Gemini instead of splitting across two Groq models (`gpt-oss-120b`/`gpt-oss-20b`) | One fewer model to manage | Gemini's 20-requests/day cap is *tighter*, not looser, than Groq's per-model daily caps — moving load onto it would have made the collision worse, not better. Confirmed by what actually happened: Gemini independently exhausted its own quota well before either Groq model did |

## 4. How it works

Three real bugs surfaced by running this at real scale, each fixed in sequence, each documented at the
point it was found (`docs/results/groq-model-replacement.md`, this file's own history):

1. `Faithfulness` failing with `LLMDidNotFinishException` on `qwen/qwen3.8-27b` at
   `max_tokens=1000` (its real OTPM ceiling, Unit 1) — fixed by decoupling the RAGAS-scoring LLM's
   `max_tokens` from `GroqSettings.max_tokens` entirely (`GroqRagasLLM.max_tokens`, now an independent
   field defaulting to `JudgeSettings.ragas_llm_max_tokens=4096`).
2. `openai/gpt-oss-120b` — the fix for (1) — turned out to have its *own* 8,000 tokens/minute cap
   (combined prompt+completion, a real 413 hit testing an intermediate value), and worse, a 200,000
   tokens/day cap that collapsed under the combined load of RAGAS scoring (5 calls/query) *and* the judge
   role (1 call/query) sharing one bucket. Fixed by splitting the two roles onto two different models —
   `openai/gpt-oss-20b` for RAGAS scoring, `openai/gpt-oss-120b` for the judge role only
   (`JudgeSettings.ragas_llm_model`).
3. Even after the split, `gpt-oss-20b`'s *own* 200,000 TPD cap was hit by query ~29 of 30 — the split
   fixed the collision, not the fact that this session's cumulative real testing (Units 1, 3, and this
   unit's dry runs) had already spent a meaningful fraction of each model's daily budget before the real
   batch even started.

## 5. What the numbers said

Full detail: `docs/results/judge-agreement.md`. Headline: **inter-judge κ = -0.0161 (unweighted), 0.0906
(quadratic), N=9 paired** — at or below chance agreement, well under plan.md's 0.6 bar. Two distinct,
real mechanisms, not one: Gemini's scores have almost no variance (10 of 11 real successes were a 5/5);
`gpt-oss-120b` systematically scores honest "I don't have enough information" refusals low (1-2) despite
an explicit rubric instruction to score them high, while Gemini follows that same instruction every time.
RAGAS `faithfulness` mean 0.7977 (n=20/29) — real, below the 0.90 target, not chased further this session.

## 6. What surprised me / what to watch for

**The judge-agreement pivot (forced by the no-hand-labeling decision) produced a more informative result
than the originally-planned judge-vs-human comparison plausibly would have.** Neither judge looks
obviously broken in isolation — a single judge's scores, read alone, would pass a casual sanity check.
It's specifically the *disagreement* that exposes that neither is fully trustworthy: one has a ceiling
effect, the other has a real instruction-adherence failure on exactly the behavior (honest abstention)
this project's confidence scorer (Phase 10) is built to produce. If this system's future prompt or model
tuning had used `gpt-oss-120b` alone as its quality signal, it would have been optimizing directly against
its own North Star.

**Every model this unit depended on hit a real quota wall in one sitting, in four structurally different
ways** (input-tokens/minute, output-tokens/minute, tokens/day, requests/day) **across two providers.**
This isn't a one-off Phase 11 inconvenience — Phase 16 (the CI gate) and Phase 19 (failure analysis) will
both need real generation/judge calls at some volume, and "the free tier survives N=30 in one sitting" is
now a falsified assumption, not an open question. Open Question #12 (progress.md) already named the
*shape* mismatch between Groq's token-based and Gemini's request-based quotas; this session adds that
Groq's own token-based caps aren't even uniform across roles or time windows (ITPM, OTPM, TPM, TPD, all
independently enforced, on top of standard per-account limits).

## 7. Where this fits and what's next

Unit 5 (custom metrics) is already done, independently of this unit — see design record for `eval/
custom_metrics.py` in progress.md's session log (no separate explainer file; the module's own docstring
and this session's progress.md entry cover the design choices, which were small enough not to warrant a
dedicated record). Unit 6 (`make eval` scorecard) is next, but the more urgent open item is fixing the
judge rubric's refusal-scoring divergence and re-running this batch once quota resets — without that, any
`make eval` scorecard built on today's judge numbers would be reporting a known-unreliable signal as if it
were a settled one.
