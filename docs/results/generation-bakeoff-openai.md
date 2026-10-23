# Generation Bake-off — OpenAI vs qwen (Phase 15, step 1)

**Run:** 2026-10-19, `uv run python experiments/15g1_generation_bakeoff.py`. Raw: `generation-bakeoff-openai-raw.json`
(every answer, every claim with its status, per-answer tokens/cost/latency). Generated table: `generation-bakeoff-openai-table.md`.
Design record: `docs/explainers/15g1-generation-bakeoff.md`.

**Protocol (identical to the Phase 9 sweep's temperature=0.0 config):** the same 20 seeded dev positives (asserted
against `sampling-sweep-raw.json`), dense bge-large top-10 with `exclude_doc_id`, 4,096-token packing, prompt v2,
temperature 0.0, max_tokens 400. qwen's 20 answers are the stored Phase 9 answers, not regenerated.

**Judge held fixed.** Every answer set is validated under both entailment judges: `qwen` (qwen/qwen3.8-27b on Groq, the judge
behind 0.7296) and `nano` (gpt-4.1-nano, the served default). **Harness check: qwen's stored answers re-validated with
the qwen judge reproduce 0.7296 exactly.** No entailment API errors and no unparseable verdicts (88 calls per judge).

## Results

| Generator | valid_citation_rate (qwen judge) | Δ vs qwen, 95% paired-bootstrap CI | W/L/T vs qwen | (nano judge) | Δ vs qwen (nano judge), CI |
|---|---|---|---|---|---|
| qwen/qwen3.8-27b (Groq) | **0.7296** | — | — | 0.6329 | — |
| gpt-4.1-mini | 0.4486 | −0.281 [−0.417, −0.140] | 5/15/0 | 0.4361 | −0.197 [−0.335, −0.062] |
| gpt-4o-mini | 0.3458 | −0.384 [−0.517, −0.253] | 2/17/1 | 0.3264 | −0.307 [−0.431, −0.187] |
| gpt-4.1-nano | 0.6665 | −0.063 [−0.209, +0.072] | 8/11/1 | 0.6288 | −0.004 [−0.137, +0.117] |

Claim status counts (qwen judge):

| Generator | claims | supported | unsupported | missing (meta/refusal · substantive) | wrong-cite share of cited claims | answers with refusal language |
|---|---|---|---|---|---|---|
| qwen | 111 | 73 | 6 | 22 (9 · 13) | 0.076 | 10/20 |
| gpt-4.1-mini | 138 | 56 | 16 | 47 (11 · 36) | 0.222 | 4/20 |
| gpt-4o-mini | 126 | 39 | 7 | 68 (21 · 47) | 0.152 | 5/20 |
| gpt-4.1-nano | 129 | 81 | 26 | 17 (2 · 15) | 0.243 | 4/20 |

"Meta/refusal" uses a regex heuristic ("the context does not contain…", "I cannot…", "let me know"). It's not hand-labelled.
Rate with those sentences excluded (qwen judge): qwen 0.793, 4.1-mini 0.519, 4o-mini 0.419, nano 0.664, so the ranking holds.

Latency and cost (sequential, non-streaming wall time from this laptop; billed tokens from the ledger):

| Generator | mean TTFT | p50 total | 19th/20 total | mean output tokens | cost / answer | total (20) |
|---|---|---|---|---|---|---|
| gpt-4.1-mini | 1,060 ms | 2,690 ms | 3,887 ms | 260 | $0.001765 | $0.035297 |
| gpt-4o-mini | 799 ms | 2,937 ms | 3,501 ms | 186 | $0.000618 | $0.012359 |
| gpt-4.1-nano | 746 ms | 1,884 ms | 2,685 ms | 242 | $0.000434 | $0.008684 |

Billed input = 67,475 tokens per model, exactly the local tiktoken count made before the run.
Answers that hit the 400-token cap: 4.1-mini 3, nano 1, 4o-mini 0.

## Reading

1. **gpt-4.1-mini (the planned generator) and gpt-4o-mini are clearly worse than qwen on this proxy.** The CIs exclude
   zero under both judges, and they lose 15/20 and 17/20 queries. The failure is mostly **substantive uncited
   claims** (36 and 47 against qwen's 13), not citation formatting: every model uses `[n]`, which the parser handles.
2. **gpt-4.1-nano ties qwen on the rate** (CI spans zero under both judges) **but gets there in a different way:** it cites
   heavily (most supported claims, 81) and wrongly (26 unsupported; 24% of its cited claims are unsupported against qwen's 7.6%).
   A spot check of those 26 found speculation, generic claims and one refusal cited to all 10 blocks, so these are real
   decorative citations, not validator strictness. That makes an answer look grounded when it isn't, which is the failure
   this project's North Star ranks worst.
3. **qwen abstains more.** 10/20 qwen answers contain refusal language, against 4–5 for every OpenAI model. On these
   20 (many issue-derived, partially answerable) questions, the OpenAI models answer where qwen declines, and their extra
   claims land as missing or unsupported.
4. **The judge alone moves the metric.** The same qwen answers score 0.7296 (qwen judge) and 0.6329 (nano judge): −0.097,
   almost all from entailment verdicts (supported_entailment 14 → 8). Qwen-era numbers and nano-judged numbers are not comparable.

**Caveats:** N=20, one run per model, a citation-validity *proxy* (not RAGAS faithfulness, not correctness), and prompt v2
was developed on Groq models (llama, then qwen). A prompt-fit effect is possible and **not tested**.

## Addendum 2026-10-19 — gpt-5-mini and gpt-5-nano (option D, design record 15g3)

Same run, same 20 queries, same prompt v2, same 400 cap (`--extra-models gpt-5-mini,gpt-5-nano`). Earlier answers were reused, not regenerated.
`reasoning_effort="minimal"`: a one-query smoke test showed **0 reasoning tokens** for both models (usage `reasoning_tokens=0`),
and across the run output tokens exceed the visible answer by only ~10–12 on average. So this tests the gpt-5 models as
generators, **not** "does reasoning help". Spend: $0.031937 (smoke test included; estimate ≤ $0.042).

| Generator | rate qwen judge | Δ vs qwen [95% CI] | W/L vs qwen | rate nano judge | Δ [CI] | claims | supported | unsupported | missing | wrong-cite share | refusal answers | at 400 cap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen | 0.7296 | — | — | 0.6329 | — | 111 | 73 | 6 | 22 | 0.076 | 10/20 | 0 |
| **gpt-5-mini** | **0.7158** | −0.014 [−0.148, +0.111] | 10/5 | 0.6242 | −0.009 [−0.164, +0.142] | 59 | 41 | 6 | 6 | 0.128 | 4/20 | 2 |
| gpt-5-nano | 0.4217 | −0.308 [−0.465, −0.163] | 3/15 | 0.3717 | −0.261 [−0.404, −0.118] | 79 | 36 | 7 | 36 | 0.163 | 7/20 | 7 |
| gpt-4.1-nano (from above) | 0.6665 | −0.063 [−0.209, +0.072] | 8/11 | 0.6288 | −0.004 [−0.137, +0.117] | 129 | 81 | 26 | 17 | 0.243 | 4/20 | 1 |

| Generator | mean TTFT | p50 total | 19th/20 total | mean output tokens | cost / answer |
|---|---|---|---|---|---|
| gpt-5-mini | 1,137 ms | 2,426 ms | 4,374 ms | 184 | $0.001173 |
| gpt-5-nano | 881 ms | 2,116 ms | 3,275 ms | 249 | $0.000261 |

**Reading:**
- **gpt-5-mini ties qwen on the rate under both judges and wins more queries than it loses (10/5).** Its unsupported count equals qwen's (6), but over fewer cited claims (share 0.128 vs 0.076), and it has far fewer than gpt-4.1-nano (26). Of everything tested, it's the only OpenAI model that matches qwen without gpt-4.1-nano's decorative citing.
- **Fewer claims, same citation density.** gpt-5-mini's 59 claims vs qwen's 111 is mostly sentence length: about 28% less text (838 vs 1,166 mean chars) but about the same number of `[n]` markers (162 vs 168), with several facts per sentence and several citations each. Caveat: the validator scores at sentence level, so a long multi-fact sentence passes or fails as one claim. For this model the rate is a coarser measurement than for qwen.
- **gpt-5-nano loses clearly**, partly through verbosity: 7/20 answers hit the 400-token cap, and a truncated last sentence has no citation (missing 36).
