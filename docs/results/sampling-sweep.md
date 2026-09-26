# Sampling Sweep — Temperature x top_p vs. Faithfulness Proxy

Phase 9, step 5. Re-scoped and re-run 2026-09-26 after `llama-3.3-70b-versatile` (the model used for the
original attempts below) was retired entirely (Open Question #15) — this run uses `qwen/qwen3.8-27b`
(`GroqSettings.model` as of Phase 11 Unit 1). Real run against Phase 6's winning retrieval config (dense,
heading-aware, bge-large, no enrichment), prompt v1, N=20 dev-positive golden queries. Faithfulness proxy:
`CitationValidator.valid_citation_rate` — fraction of citation-requiring claims that end up supported
(lexically or via entailment). Not RAGAS faithfulness (Phase 11, `docs/results/faithfulness-root-cause.md`);
named a proxy deliberately.

## What got measured this run

| temperature | top_p | N succeeded | N failed | mean valid_citation_rate | mean latency (ms) | claim status counts |
|---|---|---|---|---|---|---|
| 0.0 | 0.5 | 20 | 0 | 0.7296 | 23,847 | supported_lexical=59, supported_entailment=14, exempt=10, missing_citation=22, unsupported=6 |
| 0.0 | 1.0 | 20 | 0 | 0.7296 | 26,411 | supported_lexical=59, supported_entailment=14, exempt=10, missing_citation=22, unsupported=6 |
| 0.5 | 1.0 | 0 | 20 | — | — | — |
| 1.0 | 1.0 | not attempted | — | — | — | — |

**The two temperature=0.0 configs produced identical results** — same mean rate, same latency-adjacent
range, same exact claim-status counts, across all 20 real queries. This is the expected, mechanistically
correct result, not a bug: at `temperature=0.0`, decoding is effectively greedy (always the single
highest-probability token), so `top_p` — which only changes the *sampling pool* when there's genuine
randomness above the top logit — has nothing left to affect. plan.md's step 5 explicitly asks to "measure
it anyway" rather than assume; this is that measurement, confirmed real rather than assumed.

**Both non-zero-temperature configs did not complete.** `qwen/qwen3.8-27b` hit a real 200,000 tokens/day
(TPD) cap partway into the third config — the *same* real cap magnitude Phase 11 Unit 4 found on
`openai/gpt-oss-120b` and `openai/gpt-oss-20b`, now confirmed on this model too (a new, disclosed finding —
previously only its output-tokens/minute and input-tokens/minute caps were known, Phase 11 Units 1 and 4).
The fourth config was never attempted; the run was stopped once the TPD wall made every further call a
guaranteed, fast 429.

## What this means for the exit criterion

The temperature x top_p grid plan.md asks for is still **not fully complete** — real data exists for 2 of 4
planned configs, both at temperature=0.0. What is real and confirmed:

- A full pipeline run at real corpus scale, under the current (non-retired) model, works end to end.
- The temperature=0.0 result is real, well-powered (N=20), and mechanistically explained (top_p's real lack
  of effect at greedy decoding, not a measurement gap).
- A new, real, disclosed quota constraint specific to this model (200,000 TPD) is now documented, adding to
  Phase 11's growing real picture of how granular Groq's free-tier caps actually are (Open Question #12 in
  `progress.md`).

## Historical: the original two attempts (now-retired model, not comparable)

Kept for the record, not reused — mixing two different models within one "temperature x top_p" sweep would
confound the variable under test, so these are historical only, from the original Phase 9 session:

**Attempt 1** (`llama-3.3-70b-versatile`, temperature=0.0, top_p=0.5, N=20, `max_tokens=1024`): mean
`valid_citation_rate`=0.6634, mean latency 15,522ms. Console-observed; the model was subsequently retired.

**Attempt 2** (same model, 4-config grid, N=6/config, `max_tokens=400`): quota was already exhausted;
only 2/6 queries completed for the first config (`valid_citation_rate`=0.2500, too small a sample to trust).

## What's next

**Deferred to the user**, to run once `qwen/qwen3.8-27b`'s TPD quota resets: `uv run python
experiments/09a_sampling_sweep.py`. The script is now config-level resumable (added 2026-09-26, after this
run) — it reads any existing `sampling-sweep-raw.json`, skips configs already at `n_succeeded == N_QUERIES`
(the 2 temperature=0.0 configs above), and only runs the 2 remaining ones (temperature=0.5 and 1.0, both
top_p=1.0), so it's now safe to just re-run the same command rather than needing to preserve today's data
by hand. Each config used roughly 40,000-50,000 of the 200,000 TPD budget, so a fresh day's quota should
comfortably cover both remaining configs in one sitting.
