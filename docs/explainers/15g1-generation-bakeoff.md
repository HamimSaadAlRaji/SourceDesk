# 15g1 — Generation Bake-off: the Planned Generator Lost

**Phase:** 15 (provider migration, step 1) · **Date:** 2026-10-19 · **Files:** `experiments/15g1_generation_bakeoff.py`, `docs/results/generation-bakeoff-openai{.md,-table.md,-raw.json}`, `experiments/09a_sampling_sweep.py` (label fix), `docs/results/sampling-sweep{,-table}.md` (label fix)
**Topics from the course:** Choosing an LLM / provider bake-off (Module 3); faithfulness and citation grounding (Module 4); evaluating the evaluator (Module 4, LLM-as-judge)

## 1. What I built

A measurement, not a component. Three OpenAI generators (gpt-4.1-mini, gpt-4o-mini, gpt-4.1-nano) answer the 20 Phase 9 sweep queries under the exact Phase 9 protocol. Each answer is scored by the citation validator under two entailment judges and compared with qwen/qwen3.8-27b's stored answers. Output: a quality/latency/cost table that decides which model generates answers. Spend: $0.059732.

## 2. Why this approach

The user approved gpt-4.1-mini as the generator on 2026-10-18 on price and reputation, provisional until measured. Its baseline is qwen's 0.7296. Two design choices keep that comparison honest:

- **Hold the judge fixed.** `valid_citation_rate` depends on an LLM entailment judge for claims that lexical overlap can't settle. 0.7296 was judged by qwen, and the served default is now gpt-4.1-nano. Rerunning naively would change the generator and the judge at once. So every answer set, including qwen's stored 20, is scored by both judges.
- **Prove the harness first.** Stage 0 re-validated qwen's stored answers with qwen's judge before any paid call, and refused to continue unless it reproduced 0.7296. It did, exactly. Any later difference comes from the generator, not from a changed retrieval, packing or validator.

A third choice is specific to measurement. **Entailment does not fail closed here.** The served code turns an API error into "unsupported", which is right for serving. In a measurement, a Groq 429 would silently lower the score. The experiment's wrapper retries rate limits and raises on persistent failure (0 errors occurred).

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
|---|---|---|
| Rerun with the served nano judge only | Simplest; it's the judge in production | Confounds the comparison. This run shows the judge alone moves qwen's answers from 0.7296 to 0.6329 (−0.097), larger than nano-vs-qwen's generator gap. |
| Regenerate qwen's answers too | Same-day conditions for all four | Costs about 90k of Groq's 200k daily tokens to recreate data already on disk. Temperature 0.0 plus the exact reproduction of 0.7296 shows the stored answers still score as before. |
| Use the served `CitationValidator()` as is | No experiment-specific code | It fails closed on API errors, so quota trouble would show up as worse grounding. The wrapper keeps the same prompt and parsing and only changes error handling. |
| Larger N (e.g. 150 dev positives) | Tighter CIs | About 7.5× the cost (~$0.45) and Groq-quota-bound for the qwen judge. N=20 already separates mini and 4o-mini from qwen (CIs exclude 0), and nano's tie is the only open question. |

## 4. How it works

```
stage 0  qwen stored answers --qwen judge--> 0.7296?   (else exit before spending)
stage 1  for model in [4.1-mini, 4o-mini, 4.1-nano]:
             20 × GenerationPipeline.generate(on_token=...)   # TTFT + total wall time
             cost ← the ledger row the call settled (purpose=bakeoff_generation)
stage 2  for judge in [qwen, nano]: for each of 4 answer sets: CitationValidator.validate
```

`valid_citation_rate = supported / (claims − exempt)`, where supported means `supported_lexical` or `supported_entailment`. It's averaged per query and then across the 20 queries, the same as Phase 9. The paired bootstrap resamples the 20 per-query differences (model − qwen) 10,000 times (seed 42).

## 5. What the numbers said

Full tables: `docs/results/generation-bakeoff-openai.md`.

| Generator | rate, qwen judge | Δ vs qwen [95% CI] | rate, nano judge | wrong-cite share | cost/answer | p50 total |
|---|---|---|---|---|---|---|
| qwen (Groq) | 0.7296 | — | 0.6329 | 0.076 | free (200k TPD) | n/a here |
| gpt-4.1-mini | 0.4486 | −0.281 [−0.417, −0.140] | 0.4361 | 0.222 | $0.001765 | 2,690 ms |
| gpt-4o-mini | 0.3458 | −0.384 [−0.517, −0.253] | 0.3264 | 0.152 | $0.000618 | 2,937 ms |
| gpt-4.1-nano | 0.6665 | −0.063 [−0.209, +0.072] | 0.6288 | 0.243 | $0.000434 | 1,884 ms |

- **The approved generator, gpt-4.1-mini, lost to qwen** by 0.28 (qwen judge) and 0.20 (nano judge), with both CIs excluding zero and 15/20 queries lost. The cause is **substantive uncited claims** (36 vs 13), not formatting.
- **gpt-4.1-nano ties qwen on the rate but cites decoratively.** 26 of its cited claims are unsupported (24%) against qwen's 6 (7.6%). A spot check found speculation and generic statements carrying citations, and one refusal citing all 10 blocks.
- **The judge swap alone costs qwen 0.097**, through 14 → 8 entailment-supported claims. Nano is the stricter judge on these claims, and qwen-era and nano-era numbers can't share a column.
- **Ledger check:** billed input was 67,475 tokens per model, exactly the pre-run tiktoken count, so the reservation's input side is exact.

## 6. What surprised me / what to watch for

- **The bigger, pricier model scored worse than the smallest one.** I expected 4.1-mini ≥ nano. The data says nano cites everything and mini cites less. The proxy rewards citing *something*, so nano's rate is flattered, and the wrong-cite share is the number that exposes it. **One metric alone would have picked nano and hidden its failure mode.**
- **qwen abstains about twice as often** (10/20 answers with refusal language vs 4–5). On partially answerable issue questions, the OpenAI models answer anyway. Phase 10's confidence gate runs *before* generation, so it can't catch a generator that pushes past thin context.
- **The served default is currently the losing model.** 15g set `generation_model = "gpt-4.1-mini"` before this measurement, and it stays that way until the user decides.
- **Untested confound: prompt fit.** v2 was written against Groq models' failure pattern (uncited opening sentences). OpenAI models may need different instructions. I haven't measured this, and it's the cheapest open question (~$0.04 per model per prompt variant on these 20).
- Found while setting up: `09a`'s report template hard-coded "prompt v1", although the run used v2. Corrected in the script and both sweep result files.

## 7. Where this fits and what's next

The generator choice is **open, pending the user's decision** (options and recommendation in `progress.md`). It feeds 15h's v2-vs-v3 comparison, which should use the chosen generator, and step 4's re-measurement. Next is step 2, the router re-eval (gpt-4.1-nano vs gpt-4.1-mini, ~$0.01).
