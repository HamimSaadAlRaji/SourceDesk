# 15g2 — Prompt Fit: a Stricter Citation Prompt Made Things Worse

**Phase:** 15 (provider migration, step 1b) · **Date:** 2026-10-19 · **Files:** `src/aegis/generation/prompts/v2c.py`, `src/aegis/generation/prompts/__init__.py`, `tests/test_prompts.py`, `experiments/15g2_prompt_fit.py`, `docs/results/prompt-fit-openai-raw.json`
**Topics from the course:** Prompt engineering for grounded generation (Module 4); evaluation design: dev/test separation (Module 4)

## 1. What I built

Prompt **v2c**: v2 plus three rules aimed at the two OpenAI failure patterns 15g1 found. If a factual sentence has no citation, leave it out. Cite a block only if it states the claim. No speculative advice, and the "context doesn't cover this" sentence carries no citation. I tested it on 10 seeded dev queries, disjoint from the 20 comparison queries, with gpt-4.1-mini and gpt-4.1-nano. It **failed the dev check**, so it was never run on the comparison set.

## 2. Why this approach

15g1 left one untested explanation for the OpenAI models' gap: prompt v2 was written against Groq models' failure pattern, so maybe it just doesn't fit OpenAI models. The user chose to test that (option C) before picking a generator. The design choices:

- **The same rendering as v2.** Only the system prompt changes (test: user turn byte-identical to v2's), so a difference comes from the wording alone.
- **A disjoint dev set as the go/no-go check.** I wrote v2c from failure patterns observed on the comparison 20 in 15g1. That's a leak the dev set can't undo, so it's disclosed here. Testing on 10 queries the prompt never saw catches a prompt that only fits the comparison set, and it costs half as much as a full run.
- **Track volume as well as rate.** An answer with no checkable claim scores `valid_citation_rate = 1.0`, so a prompt that makes the model say less can look better. The script reports supported-claim counts, refusal-language answers and output length alongside the rate.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
|---|---|---|
| Run v2c straight on the comparison 20 | One run, one number | The prompt was designed by looking at those 20 answers, so a gain there could be overfitting. The dev check costs $0.044 and catches that. |
| Iterate prompts on the dev set until one wins | Could find a working wording | 10 queries, single runs: repeated attempts on the same 10 queries turn the dev set into a training set, and each round costs about $0.044. One principled attempt, honestly reported, beats a searched-for win. |
| Run the eval stage anyway after the dev failure | Completes the protocol | v2c lost on every rate cell and cut supported claims by 40–57%. Spending $0.044 to confirm a prompt already shown to be worse buys nothing. |
| Change the validator to exempt "context does not cover X" sentences | v2c tells the model to write exactly those, and the validator penalizes them | It changes the metric for every model mid-comparison, including qwen's 0.7296. That's a separate, explicit decision (Open Question #17), not something to slip in to rescue a prompt. |

## 4. How it works

`experiments/15g2_prompt_fit.py --stage dev` draws 10 dev positives with `Random(seed + 1)` after removing the 20 comparison ids. It generates v2 and v2c answers × mini and nano (40 calls) and validates each under both 15g1 judges, using the imported 15g1 entailment wrapper (retries 429s, never fails closed). There's a hard stop at $0.15 of spend per invocation. `--stage eval` exists but was not run.

## 5. What the numbers said

Dev set, 10 queries, **qwen judge** (nano judge in brackets):

| Model | Prompt | rate | supported | unsupported | missing | wrong-cite share | refusal-language answers | mean output tokens |
|---|---|---|---|---|---|---|---|---|
| gpt-4.1-mini | v2 | 0.409 (0.413) | 31 | 14 | 24 | 0.311 | 3/10 | 265 |
| gpt-4.1-mini | v2c | **0.305** (0.294) | 18 | 5 | 29 | 0.217 | 9/10 | 171 |
| gpt-4.1-nano | v2 | 0.495 (0.554) | 30 | 21 | 7 | 0.412 | 2/10 | 232 |
| gpt-4.1-nano | v2c | **0.371** (0.343) | 13 | 10 | 10 | 0.435 | 9/10 | 125 |

Spend: **$0.043726** (estimate for the full plan was $0.092; the eval stage, ≈$0.044, was not run). 57 entailment calls per judge, 0 API errors, 0 unparseable.

- **The rate fell for both models under both judges** (mini −0.10, nano −0.12, qwen judge).
- **The models mostly said less:** supported claims −42% (mini) and −57% (nano), refusal-language answers 2–3 → 9 of 10.
- **"Delete uncited sentences" didn't work:** mini still produced ~23 substantive uncited sentences (24 under v2).
- **The one improvement:** mini's wrong citations fell 14 → 5 (share 0.31 → 0.22). nano's didn't move.
- **The dev set is harder than the comparison set.** v2 baselines are mini 0.409 and nano 0.495 here, against 0.449 and 0.667 on the comparison 20, and nano's wrong-cite share is 0.41. So nano's decorative citing isn't specific to the comparison set.

## 6. What surprised me / what to watch for

- **The validator penalizes the behavior the prompt asks for.** About 6 of mini's 29 and 6 of nano's 10 v2c "missing" sentences are the instructed uncited "the context does not cover X" statements. The validator's `_requires_citation` treats them as factual claims, and it already did so for qwen in 15g1 (9 such sentences). It's the same family as Open Question #10. **An honest gap statement costs the same as a hallucinated uncited claim on this metric**, which runs against the North Star. Logged as Open Question #17 and deliberately not changed here.
- **Telling a model to delete uncited sentences made it refuse more but didn't make it cite better.** On these models, strictness turned into abstention, not discipline.
- **10 dev queries, one run each, no CI.** The direction is consistent (4/4 rate cells down, both supported-claim counts down sharply), but it's not a fine-grained measurement.

## 7. Where this fits and what's next

The prompt-fit explanation isn't supported by this attempt: a stricter prompt doesn't close the OpenAI models' gap. v2 stays the served prompt, and v2c stays in the registry, unserved, as the documented negative result. The generator decision returns to the user (A: qwen + nano fallback, B: nano, or a further model test), then step 2 (router re-eval).
