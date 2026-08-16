# Query-understanding ablation

**Phase:** 8 · Base retriever: Phase 6's winning config (dense-only, heading-aware chunking, `bge-large-en-v1.5`, no contextual enrichment), unchanged.

Four techniques, each A/B'd against that base retriever. Raw numbers: `query-understanding-ablation-raw.json`. Script: `experiments/08c_query_understanding_ablation.py` (expansion/HyDE/iterative, full 150-query dev set) + `experiments/08b_generate_rewrite_eval_set.py` + the same script's rewrite section (24-item synthetic multi-turn set — see below for why rewriting needs its own eval set).

## Summary

| Technique | NDCG@5 | Δ vs. baseline | Recall@10 | Δ vs. baseline | Latency (mean / p95) | Decision |
|---|---|---|---|---|---|---|
| Baseline (none) | 0.7587 | — | 0.4435 | — | 8.5ms / 10.9ms | — |
| Multi-query expansion | 0.3839 | **−49.4%** | 0.2136 | **−51.8%** | 5,526ms / 13,457ms | **Off** |
| HyDE | 0.5795 | **−23.6%** | 0.3180 | **−28.3%** | 1,597ms / 6,727ms | **Off** |
| Iterative retrieval | 0.6570 | **−13.4%** | 0.3734 | **−15.8%** | 574ms / 3,092ms | **Off** |
| Query rewriting | see below (own eval set) | | | | 1,542ms / 2,463ms | **On** |

Three of four techniques regress retrieval quality and are disabled. Only query rewriting — the one technique solving a genuinely different problem (recovering a follow-up query's missing context, not re-ranking or re-expressing an already-complete one) — measurably pays for itself, on the eval set built for exactly that problem.

*Baseline note:* this run's baseline (NDCG@5 0.7587, recall@10 0.4435) differs slightly from Phase 7's reported no-rerank baseline on the identical config (0.7550, 0.4324, `rerank-tradeoff.md`) — same query set, same collection, same code path. Not investigated further (the gap doesn't change any conclusion here), but plausibly HNSW's approximate search returning a marginally different candidate set query-to-query, or minor floating-point differences between this run's CPU-based encoding (switched from the configured MPS default after a repeat of the Phase 3/7 MPS-hang failure mode, see §6 of the design record) and whichever device produced the cached query vectors Phase 7 read. Flagged rather than silently reconciled.

## 1. Multi-query expansion — off

`ExpandedRetriever`: generate 3 paraphrases (`QueryExpander`, `llama-3.1-8b-instant`), retrieve each independently, fuse with RRF (Phase 5's `reciprocal_rank_fusion`, unweighted).

**NDCG@5 dropped 49.4%, the worst regression measured anywhere in this project so far** (worse than Phase 7's reranking regression at any pool size). Investigated directly rather than just reported (CLAUDE.md: a result this far outside expectation demands investigation, not a shrug). Two compounding mechanisms found, both real and reproducible against live data:

**Mechanism 1 — the small model sometimes hallucinates answers instead of rephrasing the question.** Example, `golden::issue::2618` ("Generate auto-document when wrapping all response into a custom object with error code, message, and data"): the three "paraphrases" `llama-3.1-8b-instant` returned were not restatements of the question at all —

```
You can use a generic response model with type hints to automatically generate the correct response structure...
You can leverage the `Generic` class from Pydantic to create a generic response model...
You can use a metaclass or a decorator to automatically generate the wrapper models...
```

Despite the system prompt's explicit "Do not answer the question," the model answered it — three different, invented, non-existent solutions. Embedding these retrieves whatever real chunks happen to share their invented vocabulary, not chunks relevant to the real question.

**Mechanism 2 — even when paraphrases are reasonable, unweighted RRF across 4 lists structurally favors breadth over peak relevance.** Same example: the original query's top result (`doc::advanced/additional-responses.md::heading_aware::6`, a real ground-truth chunk, rank 2 in the original-only list) never appears in any of the three hallucinated-paraphrase lists, so it earns exactly one RRF vote (`1/(60+2) ≈ 0.0159`). Meanwhile `issue::2382::heading_aware::5` — not in ground truth at all — appears in all three paraphrase lists (ranks 1, 1, 3) purely from superficial "generic model" / "Pydantic" wording overlap, earning three votes (`≈ 0.0487`) and out-scoring the actually-correct chunk by 3x. RRF's `k=60` was tuned in Phase 6 for exactly **two** sources (dense + sparse) of comparable reliability; nothing validated it for 4-way fusion where 3 of the 4 sources can be actively wrong. A quick 20-query spot check (not the full run, just this investigation) confirmed the pattern generalizes: mean NDCG@5 0.868 → 0.440, only 2/20 queries unchanged, **zero improved**.

Not fixed here — reweighting RRF for multi-source fusion or validating paraphrase quality before fusing would be new scope beyond what Phase 8 asks for, and the measured answer (off) is already actionable. Left as a concrete idea in `docs/future-work.md` territory if revisited.

## 2. HyDE — off

`HydeRetriever`: generate a hypothetical answer (`llama-3.3-70b-versatile`, not the small model — see `config.py`, this one directly determines embedding quality), embed it instead of the query.

NDCG@5 dropped 23.6% — real, but a different and much less catastrophic mechanism than expansion. An 8-query spot check (temperature=0.0, `llama-3.3-70b-versatile`) found most queries **completely unaffected** (5/8 identical scores) or **improved** (1/8) — the regression is concentrated in specific cases, not universal like expansion's. The clearest case, `golden::issue::2342` ("Bug: Breaking change in new release of Starlette"): the hypothetical answer was specific, accurate-sounding, and genuinely FastAPI-literate —

> "The removal of `UJSONResponse` in Starlette 0.14.1 introduces a breaking change that affects FastAPI's `JSONResponse` implementation. To mitigate this, you can update your FastAPI application to utilize the `ORJSONResponse` instead..."

— not hallucinated garbage. But it committed to one specific, narrow interpretation (the `UJSONResponse`/`ORJSONResponse` angle) of a question whose real ground truth spans a broader set of chunks about the general Starlette-upgrade breaking change. This is the **same mechanism Phase 7 found for the cross-encoder reranker** (`docs/explainers/07-reranking-and-encoder-architectures.md` §6): the golden set's ground truth (Phase 4's token-overlap auto-labeling) is a *broad, "on-topic" bar* (median 17 ground-truth chunks across 12 documents per query), and any technique that commits to *specific, narrow* relevance — a sharp cross-encoder, or here a HyDE answer that picks one concrete angle — is structurally disadvantaged against it, independent of whether that technique is actually working correctly. This is now the second independently-built technique to hit the identical labeling-methodology wall (see Open Question #8, progress.md) — strong enough of a pattern that it should weigh on how much Phase 11's absolute numbers (not just relative comparisons) can be trusted, not just a one-off excuse.

## 3. Iterative retrieval — off

`IterativeRetriever`: on low confidence (top-1 fused score below a threshold), reformulate the query once and retry (`QueryReformulator`, capped at 1 retry per plan.md). Threshold calibrated as the real 25th percentile of the baseline run's own top-1 score distribution (0.7747) — not a guess.

38 of 150 queries (25.3%, by construction of the p25 threshold) triggered a retry; the other 112 are provably identical to baseline (the retry path is never entered for them). The entire measured regression (NDCG@5 −13.4% over the full 150) is therefore concentrated entirely within those 38 already-hardest queries — reformulation made the hardest queries harder on average, rather than recovering them. This is consistent with the confidence *proxy* being weak, not necessarily the retry *pattern* being wrong: top-1 fused score alone conflates "genuinely ambiguous/hard query" with "good top-1 chunk that just doesn't dominate by a wide margin," and reformulating the latter risks drifting away from an already-adequate query. Phase 10's real confidence scorer (reranker score + rank-1/rank-2 gap + inter-chunk agreement + score-distribution shape) is a strictly richer signal than this provisional single-score proxy and may change this result — this measurement is about the proxy-plus-retry combination available *in Phase 8*, not a final verdict on the iterative pattern itself.

## 4. Query rewriting — on

`QueryRewriter`: resolve pronouns/ellipsis against conversation history. Can't be measured on the 150-query dev set directly — it needs `(history, follow-up)` pairs, and this corpus (FastAPI docs + closed issues, both single-turn artifacts) has none. Uses the 24-item LLM-generated synthetic multi-turn set (`data/golden/query_rewrite_eval_v1.jsonl`, `experiments/08b_generate_rewrite_eval_set.py`, sampled 4/stratum, decided with the user 2026-08-12) instead, scored against each item's *original* golden query's real `ground_truth_chunk_ids`.

| Condition | NDCG@5 | Recall@10 | MRR |
|---|---|---|---|
| turn2 alone (context-free) | 0.1995 | 0.0897 | 0.2861 |
| turn2 rewritten with turn1 as history | **0.3045** | **0.1203** | **0.4573** |
| original full query (oracle upper bound) | 0.7405 | 0.4486 | 0.8531 |

Rewriting recovers a real, meaningful fraction of the context-free query's lost signal: **+52.7% NDCG@5, +34.2% recall@10, +59.8% MRR** relative to doing nothing. It doesn't close the gap to the oracle (0.3045 vs. 0.7405) — expected, since a synthetic 1-sentence follow-up genuinely carries less information than the original multi-paragraph issue text, and the rewriter can only resolve what the history actually implies, not invent detail neither turn stated. Latency (p50 1,542ms) is real but this is the only technique in this ablation that clears its own bar: it is solving a problem (context-free follow-ups are close to unanswerable, 0.0897 recall@10) that the other three techniques don't have, rather than trying to out-perform an already-strong single-turn baseline (0.4435 recall@10) that has much less room to improve.

## What this changes

`QuerySettings.rewrite_enabled = True`; `expansion_enabled`, `hyde_enabled`, `iterative_retrieval_enabled` stay `False`. Three of four plan.md Phase 8 techniques are honest negative results, consistent with the phase's own framing ("At least one probably won't pay for itself — report that honestly") — in this case three did, not one, and each for a distinct, investigated reason rather than the same excuse repeated.
