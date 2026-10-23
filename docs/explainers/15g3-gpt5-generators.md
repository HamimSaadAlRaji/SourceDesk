# 15g3 — gpt-5-mini Is the First OpenAI Generator to Match qwen

**Phase:** 15 (provider migration, step 1c) · **Date:** 2026-10-19 · **Files:** `experiments/15g1_generation_bakeoff.py` (`--extra-models`, truncation/visible-token tracking), `docs/results/generation-bakeoff-openai{.md,-table.md,-raw.json}`
**Topics from the course:** Choosing an LLM (Module 3); faithfulness and citation grounding (Module 4)

## 1. What I built

The 15g1 bake-off extended to gpt-5-mini and gpt-5-nano, the user's option D after the prompt-fit test failed (15g2). Same 20 queries, prompt v2, 400-token cap and two judges. The result decides the generator recommendation.

## 2. Why this approach

15g1 showed gpt-4.1-mini/4o-mini under-cite and gpt-4.1-nano cites decoratively, and 15g2 showed a stricter prompt doesn't fix either. My hypothesis was that qwen's advantage came from reasoning before answering, and gpt-5 models are OpenAI's reasoning models, both already priced in the ledger. Keeping the 400 cap (which on reasoning models limits hidden reasoning and the answer together) holds the protocol fixed. Truncation is reported rather than avoided.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
|---|---|---|
| `reasoning_effort="low"` with a larger cap (e.g. 1,200) | Would actually test "reasoning helps" | It breaks parity with every other row (different cap) and costs about 2× (~$0.08 vs ≤$0.042 approved). It's worth doing only if gpt-5-mini at minimal effort had lost. |
| gpt-4.1 (full) | Strongest non-reasoning OpenAI model | Not priced in the ledger, and about 5× gpt-4.1-mini's price. At the project's $10 cap, it would make the step-4 re-measurement and 15h meaningfully more expensive. |
| Stop and pick A (qwen) | Best measured grounding | The quota limits that motivated the migration come back. D cost $0.03 to rule this in or out. |

## 4. How it works

`--extra-models` appends to `MODELS`. The run is resumable, so the 60 existing answers and 176 existing judge verdicts are reused untouched (stage 0 re-reads its stored 0.7296). New per-answer fields: `visible_tokens` (tiktoken of the answer text), compared with billed `output_tokens` to expose hidden reasoning, and the table's "answers at cap" column. Before the run, a one-query smoke test through the ledger (purpose `smoke`) read `finish_reason` and `reasoning_tokens` directly. An empty answer from reasoning eating the cap would have scored a perfect 1.0, so this was checked before spending on 40 calls.

## 5. What the numbers said

Full tables: `docs/results/generation-bakeoff-openai.md` (addendum).

| Generator | rate (qwen / nano judge) | Δ vs qwen, qwen judge [CI] | wrong-cite share | refusal answers | at cap | cost/answer | p50 / 19th-of-20 |
|---|---|---|---|---|---|---|---|
| qwen | 0.7296 / 0.6329 | — | 0.076 | 10/20 | 0 | free (TPD) | — |
| **gpt-5-mini** | **0.7158 / 0.6242** | −0.014 [−0.148, +0.111], W/L 10/5 | 0.128 | 4/20 | 2 | $0.001173 | 2,426 / 4,374 ms |
| gpt-5-nano | 0.4217 / 0.3717 | −0.308 [−0.465, −0.163] | 0.163 | 7/20 | 7 | $0.000261 | 2,116 / 3,275 ms |
| gpt-4.1-nano | 0.6665 / 0.6288 | −0.063 [−0.209, +0.072] | 0.243 | 4/20 | 1 | $0.000434 | 1,884 / 2,685 ms |

Spend $0.031937 (estimate ≤ $0.042); total $0.135486 of $10. **Reasoning tokens: 0** at minimal effort (smoke test), and the billed-minus-visible gap is ~10 tokens on average.

## 6. What surprised me / what to watch for

- **The hypothesis I proposed D on wasn't what got tested.** At `minimal` effort, the gpt-5 models produce no reasoning tokens. gpt-5-mini matched qwen anyway, so its gain isn't from reasoning at this setting. Why gpt-5-mini cites well and gpt-4.1-mini doesn't is unexplained.
- **Claim count isn't comparable across models.** gpt-5-mini writes long multi-fact sentences (59 claims, 162 citation markers; qwen 111 claims, 168 markers). The validator judges at sentence level, so one long sentence counts as one claim. The rate is coarser for this model, and a sentence that's half supported can pass lexically. The faithfulness re-measurement in step 4 (RAGAS, statement-level) is the check that doesn't have this bias.
- **Latency tail:** gpt-5-mini's 19th/20 total is 4.4 s, against 3.9 s (4.1-mini) and 2.7 s (4.1-nano). Because answers stream, TTFT (1.1 s mean) is what the user waits for, but Phase 13's 3 s p95 target is end to end. This makes that harder, not easier.
- **gpt-5-nano is verbose**: 7/20 at the cap. A larger cap might lift its score, but it would still be measured at a different cap from every other row.

## 7. Where this fits and what's next

The recommendation is gpt-5-mini as the generator, with qwen on Groq as first fallback (free, measured best) and Gemini after. That needs the user's OK before `OpenAISettings.generation_model` and `provider_order` change. The 15h v2-vs-v3 comparison and step 4 then run on the chosen generator. Next is step 2, the router re-eval.
