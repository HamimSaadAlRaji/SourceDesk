# Sampling Sweep — Temperature x top_p vs. Faithfulness Proxy

Phase 9, step 5. Re-scoped and re-run 2026-09-26 after `llama-3.3-70b-versatile` (the model used for the
original attempts below) was retired entirely (Open Question #15) — this run uses `qwen/qwen3.8-27b`
(`GroqSettings.model` as of Phase 11 Unit 1). Real run against Phase 6's winning retrieval config (dense,
heading-aware, bge-large, no enrichment), prompt v2 (corrected 2026-10-19 from "v1": v2 was already the default, `git show 7e3c6c6^`), N=20 dev-positive golden queries. Faithfulness proxy:
`CitationValidator.valid_citation_rate` — fraction of citation-requiring claims that end up supported
(lexically or via entailment). Not RAGAS faithfulness (Phase 11, `docs/results/faithfulness-root-cause.md`);
named a proxy deliberately.

## Grid complete — 2026-10-18 (all 4 configs)

The remaining two configs were run with the same resumable script (the two temperature=0.0 configs were reused, not re-run). The machine-generated table now lives in `sampling-sweep-table.md`; this file is the analysis.

| temperature | top_p | N succeeded | N failed | mean valid_citation_rate | mean latency (ms) | claim status counts |
|---|---|---|---|---|---|---|
| 0.0 | 0.5 | 20 | 0 | 0.7296 | 23,847 | supported_lexical=59, supported_entailment=14, exempt=10, missing_citation=22, unsupported=6 |
| 0.0 | 1.0 | 20 | 0 | 0.7296 | 26,411 | supported_lexical=59, supported_entailment=14, exempt=10, missing_citation=22, unsupported=6 |
| 0.5 | 1.0 | 20 | 0 | 0.7350 | 25,402 | supported_lexical=55, supported_entailment=13, exempt=7, missing_citation=13, unsupported=11 |
| 1.0 | 1.0 | **17** | 3 | 0.6546 | 25,293 | supported_lexical=41, supported_entailment=8, exempt=3, missing_citation=14, unsupported=7 |

The 3 temperature=1.0 failures were `qwen/qwen3.8-27b`'s **daily** cap (`Limit 200000, Used 197018–197056`). They were the last 3 queries of the run, so that row is n=17, not 20. Its mean is over 17 queries, and a different 17 from the other rows' 20.

**Reading, with the sample size in mind (N=20, one run per config, no CI):**

- **temperature 0.5 ≈ 0.0 on the proxy:** 0.7350 vs 0.7296, a +0.0054 difference that n=20 can't distinguish from noise. The claim mix moved, though: `unsupported` 6 → 11 (claims carrying a citation whose source doesn't back them) and `missing_citation` 22 → 13. The two shifts roughly cancel in the rate. Same rate, different failure mode: at 0.5 the model cites more often, and more often wrongly. For this project's North Star, a wrongly cited claim is worse than an uncited one, because it *looks* grounded.
- **temperature 1.0 is worse:** 0.6546, −0.075 vs 0.0. Also fewer claims overall (73 counted vs 111 at 0.0, partly the n=17), consistent with shorter, looser answers. Not significance-tested.
- **Latency does not move with temperature** (23.8–26.4 s mean across all four). The ~25 s means are dominated by pacing and rate-limit behavior in this script, not by decoding, so they say nothing about serving latency (Phase 13/14 measure that: ~1 s generation).

**Decision unchanged: temperature=0.0** (`GenerationSettings`). No config beat it on the proxy beyond noise, 0.5 shifts errors toward the more dangerous kind, and 1.0 is measurably worse. top_p is irrelevant at 0.0 (identical results, explained below).

## The earlier partial run (2026-09-26)

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

*(As of 2026-09-26; superseded by the full grid above.)*

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

Nothing for this sweep: the grid is complete as of 2026-10-18 (the 2026-09-26 "deferred to the user" note is closed). If revisited: one run per config at N=20 gives no confidence intervals. A repeat at temperature 0.5 would be the useful one, to check whether the `unsupported` 6 → 11 shift is real.
