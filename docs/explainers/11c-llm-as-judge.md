# 11c — LLM-as-Judge (Inter-Judge Agreement)

**Phase:** 11 · **Date:** 2026-09-25 · **Files:** `src/aegis/eval/judge.py`, `src/aegis/config.py`
(`JudgeSettings`), `tests/test_judge.py`, `experiments/11d_judge_smoke_test.py`
**Topics from the course:** Evaluation strategies (system vs. component, code-based vs. LLM-judge vs.
human feedback); Advanced prompting & reasoning (reasoning models); RAG vs. fine-tuning boundary N/A

## 1. What I built

`Judge` — wraps one LLM in an `.judge(query, answer, context_block) -> HelpfulnessJudgment` call, where
`HelpfulnessJudgment` is a Pydantic model (`score: int` 1-5, `reasoning: str`), returned as real
structured output via `instructor`, not parsed from free text. Two factory functions,
`build_groq_judge`/`build_gemini_judge`, build one against `openai/gpt-oss-120b` (Groq) and one against
Gemini (`GeminiSettings.model`), reusing the `InstructorLLM`-based builders Unit 1 wrote and correctly
re-scoped for exactly this (`build_groq_judge_llm`/`build_gemini_judge_llm` in `ragas_adapters.py`).

## 2. Why this approach

plan.md's Phase 11 step 3 asks for "rubric-based, structured output" to validate against hand-labels.
2026-09-25's scope change (progress.md Decision Log — the user declined to hand-label) means this
project validates the judge against a *second, independent judge* instead, via inter-judge Cohen's kappa
(`eval/agreement.py`, Unit 2). That reframing drove two real design choices here:

**A single holistic score, not a multi-dimensional rubric.** A richer rubric (grounding, tone,
completeness, conciseness as separate scored dimensions) is more thorough on paper, but every dimension
adds a place for two judges to disagree over wording rather than substance, which would muddy exactly the
signal this unit exists to produce — is the judge (in aggregate) reliable. Grounding is already covered by
RAGAS's `Faithfulness` (Unit 1); citation mechanics are already covered by the citation validator (Phase
9) and will get an aggregated custom metric in Unit 5. What's left uncovered — does the answer actually
resolve the user's real problem — is a single question, so it gets a single score.

**Two architecturally distinct models, chosen so neither judge scores its own generation.** The primary
generator is `qwen/qwen3.8-27b` (`GroqSettings.model`, Unit 1's decision). Judge A is `openai/gpt-oss-120b`
— same host (Groq), different model family, never the generator. Judge B is Gemini — different vendor
entirely, and already the project's committed fallback-generation provider, but not the primary path for
most of an N=30 batch dominated by Groq successes. This preserves the same self-preference-bias avoidance
logic the 2026-08-18 session used to pick Gemini as *a* judge in the first place, just applied to both
judges now that there are two.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ----------------- |
| Multi-dimensional rubric (grounding, clarity, tone, completeness scored separately) | More granular signal; closer to what a production eval framework (e.g. a full RAGAS-style suite) would ship | Overlaps metrics already built elsewhere (RAGAS Faithfulness, the citation validator) and adds surface area for two judges to disagree on wording rather than substance — the wrong kind of noise for a unit whose entire purpose is measuring whether disagreement reflects real quality differences |
| Reuse `qwen/qwen3.8-27b` (the same model already wired for everything else in `GroqSettings`) as one of the two judges, since it's already verified and fast | One fewer model to reason about; consistent with Unit 1's "one verified model for every Groq role" simplification | Would make one judge architecturally identical to the primary generator for most of the batch's answers — the exact self-preference-bias risk this project has avoided since 2026-08-18's original judge decision. `openai/gpt-oss-120b`'s reasoning-token tax and slow latency (disqualifying for *generation*, `docs/results/groq-model-replacement.md`) are irrelevant for an offline judge call, so there's no real cost to using it here instead |
| Free-text judge output, parsed with a regex/JSON-mode prompt (same pattern as the citation validator's tier-2 entailment check) | No new dependency category beyond what's already used elsewhere in this project | `instructor`'s structured-output path (already added for Unit 1, now for the actual purpose it was originally scoped for) eliminates an entire class of parsing failure the citation validator's simpler yes/no JSON does have to guard against (`json.JSONDecodeError`/`KeyError` fallbacks in `_default_entailment_fn`). A 1-5 int plus a reasoning string is exactly the shape `instructor`/Pydantic validation was built for |

## 4. How it works

```python
class HelpfulnessJudgment(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str

def _default_judge_fn(llm: _InstructorLLM) -> JudgeFn:
    def judge(query, answer, context_block) -> HelpfulnessJudgment:
        prompt = JUDGE_PROMPT_TEMPLATE.format(context_block=context_block, query=query, answer=answer)
        return llm.generate(prompt, response_model=HelpfulnessJudgment)
    return judge
```

`llm.generate(prompt, response_model=HelpfulnessJudgment)` is the `InstructorBaseRagasLLM` contract Unit
1 identified as the *wrong* one for `Faithfulness`/`ResponseRelevancy` but the *right* one here — the
same class (`InstructorLLM`), used for the use case it was actually built for. `Field(ge=1, le=5)` isn't
decorative: `instructor` retries the LLM call automatically if the returned JSON fails Pydantic
validation, so an out-of-range score triggers a real retry against the model rather than silently passing
through — free input-sanitization from the structured-output contract, confirmed by the two red-phase
tests (`test_helpfulness_judgment_rejects_score_below_one`/`_above_five`).

One prompt instruction is load-bearing enough to have its own regression test
(`test_judge_prompt_template_does_not_penalize_honest_refusal`): the rubric explicitly says an honest "I
don't have enough information" should score 4-5, not low. Without this, a judge could systematically
penalize exactly the behavior Phase 10's abstention gate exists to produce — this project's own North
Star failure mode (a confident, uncited wrong answer), scored backwards by the very component meant to
help validate against it.

## 5. What the numbers said

One real query (`golden::issue::2689`, the same one used in `docs/results/ragas-smoke-test.md`), one real
generated answer: `openai/gpt-oss-120b` scored it 4/5 (1,647ms), Gemini scored it 5/5 (17,225ms) — both
real structured-output calls, both real reasoning text, genuinely close but not identical scores. Full
detail: `docs/results/judge-smoke-test.md`. This is N=1 — not a real agreement measurement (kappa is not
meaningful at N=1), just confirmation the wiring produces real, sane output from both providers.

## 6. What surprised me / what to watch for

**Two more real, fixable bugs surfaced by just running it**, continuing this phase's pattern (Units 1-2
each found one): `instructor` needed its `google-genai` extra (`jsonref`) installed, and it was a
transitive-only dependency the same way `scikit-learn` was before Unit 2 — declaring both explicitly now
closes a real "works today because something else happens to need it" fragility. Second, a real transient
Gemini `503 UNAVAILABLE` on the first two full-script runs after the dependency fix — already a documented
failure mode (Phase 9's bake-off), not new, but a live reminder that any real Unit 4 batch run needs to
tolerate Gemini's occasional overload without losing the whole run, the same way `FallbackLLMClient`
already tolerates it for generation.

**Worth watching in Unit 4, not yet a finding:** design record 11b §6 flagged the risk that two LLM
judges might share more systematic bias with each other than an LLM and a human would (both drifting to
a "safe middle" score, or sharing a verbosity preference). This session's single data point (4 vs. 5, both
high) is consistent with that risk but far too small to say anything real — Unit 4's N=30 run is where
this either becomes a real, reportable pattern or doesn't.

## 7. Where this fits and what's next

Unit 4 (N=30 batch run: generation → RAGAS → both judges → inter-judge kappa) is next, and is where this
unit's numbers stop being N=1 sanity checks and start being the actual measurement DoD tracker item #8
(kappa ≥ 0.6) needs. Unit 4 should budget for Gemini's real quota (20 requests/day/model, `docs/results/
llm-bakeoff.md` context) and occasional transient 503s across a 30-query judge pass on top of the N=30
generation calls already planned.
