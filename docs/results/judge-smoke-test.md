# LLM-as-Judge Smoke Test — Unit 3 (Phase 11)

One real generated answer, scored by both real judges, before trusting the wiring for Unit 4's N=30
batch. Same query as `docs/results/ragas-smoke-test.md` (`golden::issue::2689`) for direct comparability.
Script: `experiments/11d_judge_smoke_test.py`.

## Results

| judge | score (1-5) | latency (ms) | reasoning |
|---|---|---|---|
| `openai/gpt-oss-120b` (Groq) | 4 | 1,647 | "...addresses the user's concern about the root_validator running twice, confirming that this behavior cannot be avoided because it's needed for security and response consistency. While it doesn't provide a workaround (since none exists), it directly resolves the user's confusion..." |
| `gemini-3.5-flash` | 5 | 17,225 | "...correctly and clearly explains why the validation runs twice, referencing the design of FastAPI's `response_model`... directly addresses the user's observation of redundant validation and explains that this is expected behavior by design." |

`[N=1 sanity check only]`: `raw_agreement_rate=0.0`, `cohen_kappa=0.0` — both degenerate at N=1 with any
disagreement (a single differing pair scores as 0% raw agreement, and kappa is not a meaningful statistic
at this sample size regardless). Printed only to confirm `eval/agreement.py`'s functions run against real
judge output without erroring — the real inter-judge agreement measurement is Unit 4's job, at N=30.

## Two real bugs found and fixed before this worked

1. **Missing `jsonref` dependency.** First attempt: `ConfigurationError("The 'jsonref' package is
   required for Gemini structured output. Install it with: pip install 'instructor[google-genai]'")`.
   `instructor` itself was only a transitive dependency (pulled in via `ragas`), never declared directly
   despite `ragas_adapters.py` importing it by name — same gap `scikit-learn` had before Unit 2. Fixed:
   `uv add "instructor[google-genai]"`.
2. **Transient Gemini `503 UNAVAILABLE`** ("high demand"), on the first two full-script attempts after
   the dependency fix. Same real failure mode already documented in the Phase 9 provider bake-off
   (`docs/results/llm-bakeoff.md`: "3 transient `503` ... a different and much rarer failure mode").
   Not a code bug — a direct, isolated retry (`build_gemini_judge().judge(...)`) succeeded on a later
   attempt with no code change.

## What this confirms

- `build_groq_judge`/`build_gemini_judge` (Unit 1's `InstructorLLM`-based builders, renamed and correctly
  re-scoped this session specifically for this use) work end-to-end with real structured output —
  `HelpfulnessJudgment(score, reasoning)` parsed correctly from both providers, no manual JSON handling.
- `GeminiRagasLLM`'s sibling path (`build_gemini_judge_llm`, same underlying `instructor.from_genai`
  client construction) getting a real successful call here is indirect evidence the Gemini side of the
  Unit 1 RAGAS adapters is plumbed correctly too, though `GeminiRagasLLM` itself (the `BaseRagasLLM`,
  not the judge's `InstructorBaseRagasLLM`) still hasn't been smoke-tested directly — still open per
  `docs/results/ragas-smoke-test.md`.
- Latency gap between judges (1.6s vs. 17.2s) matches the pattern already measured for these two models
  in their *generation* roles (`docs/results/groq-model-replacement.md`, `docs/results/llm-bakeoff.md`) —
  expected, and acceptable here since judge calls in Unit 4 run offline against a batch, not on any
  user-facing request path.
