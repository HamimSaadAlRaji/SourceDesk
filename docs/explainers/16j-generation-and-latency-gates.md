# 16j — The Generation Gate, Latency Metrics, and What They Can't See

**Phase:** 16 · **Date:** 2026-10-25 · **Files:** `src/aegis/eval/generation_gate.py`, `src/aegis/eval/router_gate.py` (`router_p95_ms`), `tests/test_generation_gate.py`, `ci/baselines/generation.json`, `ci/baselines/router.json`, `generation-gate` job in `.github/workflows/eval.yml`, `GateSettings` in `src/aegis/config.py`
**Topics from the course:** Evaluating the LLM's performance (RAGAS: Response Relevancy, Faithfulness) and Handling hallucinations; Evaluation strategies (code-based vs. LLM-judge). This tier is code-based on top of LLM output: the project's own citation validator, not a judge model.

## 1. What I built

A `generation-gate` job that generates answers for the same seeded 20 dev positives used in Phase 9 and the 15g bake-off, with the served generator (`gpt-5-mini`) and the served citation validator (tier-2 entailment on `gpt-4.1-mini`), over the restored index, and gates on `valid_citation_rate`, `cited_answer_rate`, `empty_answer_rate` and generation p95 latency. The router job gained `router_p95_ms`. The PR comment grew a third section. Worst-case spend per PR run is split between the two LLM jobs (router job capped at $0.05, generation job at $0.10) so the approved $0.15 per run holds.

## 2. Why this approach

- **A silent hole in the metric.** `valid_citation_rate` is the fraction of citation-*requiring* claims that are supported, and is a vacuous 1.0 for an answer with nothing to check, so an empty answer, or a model that stops citing, can leave it flat or push it up. The gate scores an empty answer 0.0 and tracks `cited_answer_rate` and `empty_answer_rate` as separate metrics (7 tests).
- **Entailment must not fail closed while measuring.** The served validator turns an API error into "unsupported". That is right for serving and would score an outage as bad citations here, so a persistent failure raises and fails the job (15g1's rule).
- **OpenAI-only generator in CI.** The served fallback chain would quietly measure Groq or Gemini if OpenAI failed, a different model from the one the baseline describes.
- **Retrieval is not under test here**, so query vectors come from the tracked embedding cache (180/180 present in the committed copy): no 1.3 GB model download, no CPU encoding.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| RAGAS faithfulness + judge model | The Phase 11 headline metrics | The judge's κ is 0.13 against a 0.6 target and the judge/RAGAS roles are not wired through the ledger; a gate on an unvalidated judge is a gate on noise. |
| Gate `code_syntax_valid_rate` | In the Phase 11 scorecard | All 20 sampled answers contain zero Python code blocks (`code_blocks_checked: 0`), so it has no support. Recorded, not gated; a metric with no data must not pass or fail. |
| One end-to-end p95 gate | The plan's "p95 latency budget" | The confidence scorer's cross-encoder alone costs 4.09 s per query on an M1 and the hosted runner is slower; an end-to-end number in CI would measure the runner's CPU lottery (16h), not the code. |
| N = 40-60 queries for a tighter band | Less noise | ~$0.07-0.10 per run for √2-√3 better power; not enough to change what the gate can detect (see §5). Left as future work. |
| Fallback chain as served | "Test what ships" | Silently changes the model under test (above). |

## 4. How it works

`score_generation` aggregates per-answer rows; the band for each metric is 2× the largest deviation from the 3-run mean, rounded up (the same rule as retrieval and router) except latency, which is a judgment value because three samples cannot describe an API's heavy tail. The baseline is the mean of three identical-input runs. A metric that is undefined (NaN: no code blocks) is omitted from the report, so a *gated* metric going missing fails the gate.

## 5. What the numbers said

Four runs on identical inputs (same 20 queries, same prompt v3, same generator):

| Run | valid_citation_rate | cited_answer_rate | p95 (ms) | Cost |
| --- | --- | --- | --- | --- |
| 1 | 0.6744 | 0.80 | 6,661 | $0.0339 |
| 2 | 0.8488 | 0.90 | 5,851 | ~$0.034 |
| 3 | 0.8710 | 1.00 | 6,591 | ~$0.034 |
| 4 (CI-style restored index) | 0.7339 | 0.85 | 6,179 | $0.0295 |

**The noise is large**: `valid_citation_rate` spans 0.197 and `cited_answer_rate` 0.20 across runs that differ in nothing the code controls (gpt-5-mini has no temperature control; n = 20 is small). Baseline = mean of runs 1-3 (0.7981 / 0.90 / 0.00 / 6,368 ms); bands: citation **0.25** (2 × 0.124), cited-answer **0.20** (2 × 0.10), empty-answer **0.0**, p95 **2,000 ms** (judgment). All four runs pass; a synthetic "citations broke" result (0.20 / 0.30) fails both citation metrics.

**What this gate can and cannot see.** With a band of 0.25 it catches gross breakage (citations disappearing, answers coming back empty, latency doubling) and *not* a quiet regression: a 10-point citation drop sits inside the noise. The plan's "8% silent drop" is visible to the retrieval gate and not to this one. That is a measured limitation, not a bug.

Router latency (3 runs, 8 concurrent calls, p95): 2,100 / 2,282 / 2,157 ms (mean 2,180, spread 182). Router accuracy was 0.7586 on all three runs here, against 0.7759 on earlier days: the same one-query drift as 16e, inside its 0.035 band. Router p95 band 1,000 ms (judgment). Real spend this unit: **$0.176** on the project ledger (4 generation runs, 3 router runs); local ledger now $0.607 of $10, plus about $0.066 spent by the four CI runs so far (each run's own ledger, from its artifact), so the project total is about **$0.673**.

## 6. What surprised me / what to watch for

- **20% of answers carried no supported citation in run 1** (cited_answer_rate 0.80), and 0% in run 3. The pre-existing citation quality is far less stable than any single number from earlier phases suggested; Phase 9/11 reported single-run figures.
- **A gate's band is its sensitivity.** Setting it from the measured noise is the honest choice; it also means "all green" here is weak evidence for small changes. The retrieval and router gates are much sharper (their noise is ~0.01).
- **A CI cost estimate was 2.4× too high** ($0.08 predicted, $0.034 measured) because gpt-5-mini sees ~1.7K tokens a query and the reasoning effort is minimal; estimates should be measured on the first run, not assumed.
- **The latency baseline is a weather report**: p95 of 20 calls is effectively the second-slowest call. Expect the band to be exceeded by a slow API day, in which case the right response is to re-run, and if it recurs to look at the provider, not widen blindly.

## 7. Where this fits and what's next

Plan items done: LLM-backed gate tiers (router, generation), per-stage latency (router, generation, search recorded). **The 3 s end-to-end p95 target remains unmet and ungated in CI** (5.6 s from the qwen era, never re-measured on gpt-5-mini): it needs a local measurement, a Phase 13 re-run, not a CI gate. The escalation check (16i) is the remaining tier.
