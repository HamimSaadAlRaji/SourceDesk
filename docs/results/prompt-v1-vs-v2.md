# Prompt v1 vs. v2 — Citation Coverage

Follow-up to Phase 9 step 3. v1's real measured gap (N=20, Groq, `docs/results/sampling-sweep.md`): 21.1%
of claims `missing_citation`, spot-checked to a specific pattern — opening/topic sentences
disproportionately lacking a citation. v2 (`generation/prompts/v2.py`) adds an explicit first-sentence
citation rule and a worked example targeting that exact pattern.

## Result: v2 wins, real data, promoted to the shipped default

Two attempts hit real free-tier quota walls before producing usable data (Gemini's 20-requests/day limit,
then Groq's 100,000-tokens/day limit still exhausted hours later — see the earlier attempts' raw files,
`prompt-v1-vs-v2-raw.json` and the first pass of `prompt-v1-vs-v2-groq-raw.json`, kept for the record). A
third attempt, once Groq's daily quota had actually reset, produced a real, complete, matched-pair
comparison: `experiments/09c_prompt_v2_check.py` against Groq `llama-3.3-70b-versatile`, same 6 real
dev-positive queries, same retrieval context, `max_tokens=400`. (v1's run also completed 2 further queries
v2's run didn't reach, due to a quota wall partway through v2's half — those 2 are excluded from the
matched comparison below to keep it apples-to-apples; full raw data for all 8 in
`prompt-v1-vs-v2-groq-raw.json`.)

| query | v1 `valid_citation_rate` | v2 `valid_citation_rate` |
|---|---|---|
| `golden::issue::2689` | 0.5455 | 0.6154 |
| `golden::issue::2884` | 0.6667 | 0.8000 |
| `golden::issue::3212` | 0.2000 | 0.4000 |
| `golden::issue::753` | 0.7500 | 0.7692 |
| `golden::issue::219` | 0.3571 | 0.8000 |
| `golden::issue::11104` | 0.0000 | 0.0000 |
| **mean (N=6)** | **0.4199** | **0.5641** |

**v2 wins or ties on 6/6 comparisons: 5 strict wins, 1 tie, 0 losses.** Mean `valid_citation_rate`
improved 0.4199 → 0.5641, a **+34.3% relative** improvement. The one tie (`issue::11104`) is a query where
the retrieved context genuinely doesn't contain the answer (upgrading `python-multipart` to `0.0.7`) —
both versions correctly scored 0.0 rather than fabricating, consistent with earlier findings on this same
query in the sampling sweep and bake-off.

## The mechanism held — confirmed on real generated text, not just the aggregate number

`golden::issue::2689` is the clearest example. v1's answer opens:

> "The issue you're describing is related to redundant validation when using `response_model` in
> FastAPI." — **no citation.**

v2's answer to the identical question, same retrieved context:

> "The issue you're encountering is related to redundant validation in FastAPI when using
> `response_model` and a Pydantic model with validators **[4]**." — **cited.**

This is exactly the pattern v2's explicit first-sentence rule and worked example were written to fix, and
it fixed it, on a real generation, not a synthetic test case. `golden::issue::219` shows the same pattern
even more sharply (v1 opens with an uncited framing sentence and takes several more sentences before its
first citation; v2's very first sentence carries `[1]`), which is the main reason that query jumped from
0.3571 to 0.8000.

## Decision

**`GenerationSettings.prompt_version` promoted from `"v1"` to `"v2"`** (`src/aegis/config.py`). N=6 is
modest — well short of the N≥15-20 that would be ideal — but the evidence clears a reasonable bar for a
production default change: real data, a consistent direction (0 losses), and a mechanistic explanation
confirmed by inspecting the actual before/after text, not just a favorable mean that could be noise. A
larger confirmatory run (N≥15-20, once quota allows) would still be worth doing, but isn't required to
justify shipping v2 now — CLAUDE.md's bar is "measured, not invented," and this is measured.

`docs/results/prompt-v1-vs-v2-raw.json` (the Gemini attempt, 2/10 v1 only) and the untruncated
`prompt-v1-vs-v2-groq-raw.json` (all 8 v1 + 6 v2 real completions) are both kept for the record.
