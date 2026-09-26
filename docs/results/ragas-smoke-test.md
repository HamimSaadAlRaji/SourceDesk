# RAGAS Smoke Test — Unit 1 (Phase 11)

Phase 11 Unit 1's actual goal, blocked since 2026-08-18 by the Groq model-retirement problem
(`docs/results/groq-model-replacement.md`) and, once that cleared, by a real API-protocol mismatch in
the installed `ragas==0.3.9` (design record 11 §3–4). One real dev-positive golden query, real dense
retrieval, real `qwen/qwen3.8-27b` generation, real `Faithfulness`/`ResponseRelevancy`/
`IDBasedContextPrecision`/`IDBasedContextRecall` scoring. Script: `experiments/11c_ragas_smoke_test.py`.

Query: `golden::issue::2689` ("execution optimization, redundant validation" — the `response_model`
double-validation question also used in the Phase 9 bake-off). 10 ground-truth chunk IDs.

## Results

| metric | score | latency (ms) | LLM calls | quota cost |
|---|---|---|---|---|
| `faithfulness` | 0.8750 | 16,157 | 2 (statement generation + NLI entailment) | Groq |
| `response_relevancy` | 0.7820 | 55,487 | 3 (strictness=3, one hypothetical-question generation call each) | Groq (LLM) + local (embeddings, `Encoder`, zero quota) |
| `id_based_context_precision` | 0.8000 | 0.9 | 0 | none — pure ID-set arithmetic |
| `id_based_context_recall` | 0.8000 | 0.1 | 0 | none — pure ID-set arithmetic |

Generation itself (not a RAGAS metric, but part of the same real call chain): `provider=groq`,
`latency_ms=868–935` across runs, `valid_citation_rate=1.0`.

## Per-query cost projection for Unit 4's N=30 batch

`response_relevancy` is the expensive metric — 55.5s here, though that includes a one-time embedding
model weight-load (`Loading weights: 100%|...`) that won't repeat across queries in the same process.
`faithfulness` is a steadier 16.2s (2 fixed LLM calls regardless of query). Rough per-query budget for
Units 1+4 combined at N=30: **faithfulness ≈ 8 min, response_relevancy ≈ high (weight-load is one-time,
but 3 LLM calls/query at real Groq latency adds up)** — real per-query numbers once weights are warm are
Unit 4's job to measure properly across the full batch, not extrapolated from N=1 here. This smoke test's
job was proving the wiring works end-to-end, which it now does.

## Why this took two real bug fixes, not just the model swap

Getting here required correcting `eval/ragas_adapters.py` twice after the Groq model was already fixed:

1. `Faithfulness`/`ResponseRelevancy` (built on `PydanticPrompt`) need a `BaseRagasLLM` (`generate_text`/
   `agenerate_text` → `LLMResult`). The adapter built last session (`InstructorLLM`, what `llm_factory`
   itself also returns) is a *different* protocol — `generate(prompt, response_model) -> T`, for
   structured-output calls — and gets silently misrouted by `PydanticPrompt`'s own
   `hasattr(llm, "agenerate") and not hasattr(llm, "run_config")` LangChain-detection heuristic, producing
   an `AttributeError` on a method (`agenerate_prompt`) `InstructorLLM` doesn't have, not a clean type
   error.
2. Same split on the embeddings side: `ResponseRelevancy` calls `.embed_query` (the legacy,
   `langchain-core`-`Embeddings`-derived `BaseRagasEmbeddings`, plural), not `.embed_text` (the modern
   `BaseRagasEmbedding`, singular, this file originally built against).

Full mechanism, code, and the reasoning behind hand-writing `BaseRagasLLM` subclasses instead of adding
`langchain-groq`/`langchain-google-genai`: design record 11.

## What's still open

- This is N=1. `faithfulness=0.875` and `response_relevancy=0.782` are real numbers but say nothing yet
  about whether the ≥0.90 faithfulness target (plan.md exit criterion) holds at N=30 — that's Unit 4.
- `GeminiRagasLLM` (the Gemini-side `BaseRagasLLM`) is written and follows the same pattern as
  `GroqRagasLLM`, but hasn't been smoke-tested with a real call yet — only the Groq path has a real
  result so far. Worth a real check before Unit 4 needs it (e.g. if Groq quota runs out mid-batch and
  RAGAS scoring itself needs a fallback, not just generation).
