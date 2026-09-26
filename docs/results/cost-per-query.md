# Cost Per Query, Per Configuration (Phase 12 Unit 6)

Every number below is real and already published elsewhere in this project (sources inline); this table only assembles them into one per-query total. **Tokens and latency, not dollars** -- this project has run entirely on Groq/Gemini free tiers, so the real cost that has actually constrained this project is quota (TPD/RPM caps), not billed dollars; there is no real dollar figure to report.

| configuration | components (ms) | total latency | completion tokens |
|---|---|---|---|
| Cache miss, no rerank (shipped default) | retrieval=6.7, confidence scoring=4,088.9, generation=8,692.0 | 12,787.5 ms | 254.7 |
| Cache miss, with rerank (pool=10, not shipped) | retrieval=6.7, full rerank (pool=10)=1,560.0, confidence scoring=4,088.9, generation=8,692.0 | 14,347.5 ms | 254.7 |
| Semantic cache hit | exact-tier miss check=0.0, semantic-tier hit=0.7 | 0.8 ms | 0.0 |
| Exact cache hit | exact-tier hit=0.0 | 0.0 ms | 0.0 |
| Escalation (partial -- see note) | retrieval=6.7, confidence scoring=4,088.9 | 4,095.5 ms | 0.0 |

## Sources for each component

- **retrieval** (6.66 ms): `docs/results/ann-recall-vs-latency.md`, the real shipped HNSW config (m=16, ef_construct=100, ef_search=100) -- measured on `bge-small` (the dev workhorse), not `bge-large` (the shipped encoder); not re-measured for `bge-large` here.
- **full rerank, pool=10** (1560.0 ms p50, 5227.0 ms p95): `docs/results/rerank-tradeoff.md` -- the cheapest swept pool, and the one that would ship if reranking were ever turned on (`RerankSettings.enabled=False` in the shipped config; every larger pool measured worse on both latency and NDCG@5).
- **confidence scoring** (4088.9 ms): design record 10 §6's real 180-query dev sweep wall time (736s / 180 queries), at the shipped `rerank_pool_size=3` -- paid by every non-cache-hit query, unconditionally, to decide escalate-vs-generate (`guardrails/confidence.py::handle`). Deliberately not a per-pair estimate multiplied out: design record 10 §6 already found a smaller probe's per-pair mean undersold this real number by roughly 4x.
- **generation** (8692.0 ms, 254.7 completion tokens): `docs/results/groq-model-replacement.md`'s real full 6/6-succeeded run at the current shipped config (`qwen/qwen3.8-27b`, `max_tokens=1000`).
- **exact-tier check/hit** (0.014 ms) and **semantic-tier hit** (0.741 ms): `docs/results/cache-hit-rate-simulation.md` (Phase 12 Unit 3), mean across the 3 real Zipf skews measured there.

## What's missing, disclosed rather than guessed at

**Escalation's total cost is a real, partial lower bound, not a complete number.** `guardrails/confidence.py::handle` structurally cannot call both `generate_fn` and `escalate_fn` -- confirmed by its own test, a spy that raises if invoked -- so escalation never pays the generation cost above it, real savings. But `escalate_fn` itself (whatever actually happens on a real escalation -- writing to a queue, notifying a human agent, rendering the escalation UI) has no real implementation yet anywhere in this codebase; only `EscalationPayload`'s construction exists (`guardrails/confidence.py::build_escalation_payload`), which is already included in the measured confidence-scoring cost above. Phase 14 (API/serving) is where a real handler would be built. The escalation row above is real and correct as far as it goes, not a placeholder number, but it doesn't yet include whatever that eventual handler costs.

**Rerank's effect on generation cost is assumed unchanged, not separately measured.** Reranking changes which chunks end up in the context, not how many tokens `context_budget_tokens` (4096) allows into the prompt, so the "with rerank" row reuses the same real generation number as the no-rerank row. A real difference in output length is plausible but unmeasured -- disclosed as an assumption, not reported as if it were its own measured figure.
