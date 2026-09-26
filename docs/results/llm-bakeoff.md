# LLM Provider Bake-off — Groq vs. Gemini

Phase 9, step 6. Real run: `experiments/09b_llm_bakeoff.py`, Phase 6's winning retrieval config (dense,
heading-aware, bge-large, no enrichment), prompt v1, temperature=0.0, top_p=1.0, `max_tokens=1024`
(shared cap, both providers), 15 sampled dev-positive golden queries, retrieval computed once per query
and shared across both providers.

## Results

| provider | model | N succeeded | N failed | mean valid_citation_rate | mean latency (ms) |
|---|---|---|---|---|---|
| groq | `llama-3.3-70b-versatile` | 2 | 13 | 0.1538 | 1,549 |
| gemini | `gemini-3.5-flash` (thinking disabled) | 12 | 3 | 0.7014 | 21,437 |

**This is not a fair, equal-N quality comparison, and shouldn't be read as one.** Groq's 13 failures were
all the same real `RateLimitError` (100,000 tokens/day quota, already most of the way exhausted by the
sampling sweep earlier in this session) — an availability failure, not a quality signal. N=2 is far too
few to trust as "Groq's citation quality." Gemini's 3 failures were transient `503 UNAVAILABLE` ("high
demand") server errors, a different and much rarer failure mode in this session's experience.

## What is a fair comparison: the two queries both providers actually answered

| query | groq valid_citation_rate | gemini valid_citation_rate |
|---|---|---|
| `golden::issue::2689` (redundant response_model validation) | 0.3077 | 1.0000 |
| `golden::issue::11104` (python-multipart 0.0.7 upload error) | 0.0000 | 0.0000 |

On the one query where they diverged, Gemini's answer stayed tightly grounded in the single most relevant
context block (`[1]`, repeated 6 times across a focused, structured answer) while Groq's answer hedged
across several less-relevant blocks (`[1][4][7]`) without clearly grounding each sentence. On the query
neither could answer, both correctly declined rather than fabricating — the context genuinely didn't
contain a resolution for that specific issue, and both providers behaved appropriately given that.

## The real, load-bearing finding: latency, not quality

Gemini's mean latency (21.4s) is roughly **14x** Groq's (1.5s) in this run — every single Gemini call
exceeded the project's 3-second end-to-end latency budget
(`RerankSettings.pipeline_latency_budget_ms`) on its own, before retrieval, budgeting, or citation
validation are even added. This is the concrete, measured justification for plan.md's architecture
("Groq primary, Gemini secondary"): Groq's custom inference hardware is the reason it can be the
*primary* provider at all under this budget; Gemini's role has to be "an answer eventually, when Groq
can't," not "an equally-fast alternative." A quality-only reading of this bake-off (Gemini "wins" on
`valid_citation_rate`) would miss the more decision-relevant number entirely.

## What this means for "choosing an LLM"

Under real operating conditions in this session, the two providers failed in different, informative ways:
Groq failed on a **predictable, quota-shaped** axis (hard daily ceiling, then works normally again);
Gemini failed on an **unpredictable, load-shaped** axis (transient server overload, no quota signal).
`FallbackLLMClient`'s narrow pre-first-token fallback boundary (design record 09 §4) is built for exactly
this kind of failure — both are the "auth/rate-limit/model-unavailable errors surface before any token is
produced" case it targets, not the harder mid-stream case.

Not measured here (real gap, not glossed over): cost. Both providers were used on free tiers this
session; a production cost-per-query comparison needs real pricing data once either tier is exceeded,
which is Phase 12's job (`docs/results/` cost tables), not this phase's.
