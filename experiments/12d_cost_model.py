"""Phase 12 Unit 6 (plan.md step 6): cost per query per configuration --
cache miss with rerank, without, semantic hit, exact hit, escalation.

Zero new measurements. Every number here is already real and published
elsewhere in this project; this script's only job is to assemble them
into one per-query total without hand-arithmetic (a citation error is
easy, an addition error in a markdown table is easier) and make every
component's source explicit.

Deliberately reports **tokens and latency, not dollars**: this project has
run entirely on Groq/Gemini free tiers all session, so the real, load-
bearing cost dimension has been quota (the TPD/RPM/daily caps that have
repeatedly blocked real work -- Phase 9/11's deferred-to-the-user reruns
exist because of exactly this) and latency, not billed dollars. A
fabricated "$/query" from a hypothetical paid-tier price sheet would be
exactly the invented number CLAUDE.md forbids -- there is no real dollar
figure this project has ever paid, so none is reported.
"""

from __future__ import annotations

from aegis.config import REPO_ROOT

RESULTS_MD = REPO_ROOT / "docs" / "results" / "cost-per-query.md"

# Real dense retrieval latency at the actual shipped HNSW config (m=16,
# ef_construct=100, ef_search=100) -- docs/results/ann-recall-vs-latency.md.
# That sweep used bge-small (the dev workhorse), not bge-large (the
# shipped encoder) -- disclosed, not re-measured here.
RETRIEVAL_LATENCY_MS = 6.66

# Real full-rerank latency at pool=10 -- the cheapest swept pool, and the
# one that would actually ship if reranking were ever turned on, since
# every larger pool measured *worse* on both latency and NDCG@5
# (docs/results/rerank-tradeoff.md). Reranking is RerankSettings.enabled=
# False in the shipped config; this row is the real cost if it weren't.
RERANK_LATENCY_P50_MS = 1560.0
RERANK_LATENCY_P95_MS = 5227.0

# Real confidence-scoring cost -- every non-cache-hit query pays this,
# unconditionally, to decide escalate-vs-generate (guardrails/confidence.py
# ::handle), whether or not the full RerankedRetriever stage above ran.
# The real number is the full 180-query dev sweep's measured wall time
# (design record 10 §6: "the real run took 736s (12.3 minutes)" at
# rerank_pool_size=3), not a per-pair estimate multiplied out -- a smaller
# 6-query probe's per-pair mean (~320ms) would have projected ~960ms/query
# here, which design record 10 §6 already flagged as a real, measured
# undersell of the full run by roughly 4x.
CONFIDENCE_SCORING_LATENCY_MS = 736_000.0 / 180

# Real generation cost at the current shipped Groq config
# (qwen/qwen3.8-27b, max_tokens=1000), the full real 6/6-succeeded run
# (docs/results/groq-model-replacement.md's "Full 6-query set at 1000").
GENERATION_LATENCY_MS = 8_692.0
GENERATION_COMPLETION_TOKENS = 254.7

# Real cache-hit latencies, mean across the 3 real Zipf skews measured in
# Unit 3 (docs/results/cache-hit-rate-simulation.md) -- both tiers are
# checked in the real architecture's order (exact first, semantic on an
# exact miss), so a semantic hit's total cost includes paying for the
# exact-tier miss check first.
EXACT_HIT_LATENCY_MS = 0.014
SEMANTIC_HIT_LATENCY_MS = 0.741  # mean of 0.796/0.791/0.635 across the 3 skews


def main() -> None:
    configs = [
        {
            "name": "Cache miss, no rerank (shipped default)",
            "components": [
                ("retrieval", RETRIEVAL_LATENCY_MS),
                ("confidence scoring", CONFIDENCE_SCORING_LATENCY_MS),
                ("generation", GENERATION_LATENCY_MS),
            ],
            "tokens": GENERATION_COMPLETION_TOKENS,
        },
        {
            "name": "Cache miss, with rerank (pool=10, not shipped)",
            "components": [
                ("retrieval", RETRIEVAL_LATENCY_MS),
                ("full rerank (pool=10)", RERANK_LATENCY_P50_MS),
                ("confidence scoring", CONFIDENCE_SCORING_LATENCY_MS),
                ("generation", GENERATION_LATENCY_MS),
            ],
            "tokens": GENERATION_COMPLETION_TOKENS,
        },
        {
            "name": "Semantic cache hit",
            "components": [
                ("exact-tier miss check", EXACT_HIT_LATENCY_MS),
                ("semantic-tier hit", SEMANTIC_HIT_LATENCY_MS),
            ],
            "tokens": 0.0,
        },
        {
            "name": "Exact cache hit",
            "components": [("exact-tier hit", EXACT_HIT_LATENCY_MS)],
            "tokens": 0.0,
        },
        {
            "name": "Escalation (partial -- see note)",
            "components": [
                ("retrieval", RETRIEVAL_LATENCY_MS),
                ("confidence scoring", CONFIDENCE_SCORING_LATENCY_MS),
            ],
            "tokens": 0.0,
        },
    ]

    for c in configs:
        c["total_latency_ms"] = sum(v for _, v in c["components"])

    lines = [
        "# Cost Per Query, Per Configuration (Phase 12 Unit 6)",
        "",
        "Every number below is real and already published elsewhere in this project (sources inline); "
        "this table only assembles them into one per-query total. **Tokens and latency, not dollars** -- "
        "this project has run entirely on Groq/Gemini free tiers, so the real cost that has actually "
        "constrained this project is quota (TPD/RPM caps), not billed dollars; there is no real dollar "
        "figure to report.",
        "",
        "| configuration | components (ms) | total latency | completion tokens |",
        "|---|---|---|---|",
    ]
    for c in configs:
        components_str = ", ".join(f"{name}={value:,.1f}" for name, value in c["components"])
        lines.append(f"| {c['name']} | {components_str} | {c['total_latency_ms']:,.1f} ms | {c['tokens']:.1f} |")

    lines += [
        "",
        "## Sources for each component",
        "",
        f"- **retrieval** ({RETRIEVAL_LATENCY_MS} ms): `docs/results/ann-recall-vs-latency.md`, the real "
        "shipped HNSW config (m=16, ef_construct=100, ef_search=100) -- measured on `bge-small` (the dev "
        "workhorse), not `bge-large` (the shipped encoder); not re-measured for `bge-large` here.",
        f"- **full rerank, pool=10** ({RERANK_LATENCY_P50_MS} ms p50, {RERANK_LATENCY_P95_MS} ms p95): "
        "`docs/results/rerank-tradeoff.md` -- the cheapest swept pool, and the one that would ship if "
        "reranking were ever turned on (`RerankSettings.enabled=False` in the shipped config; every larger "
        "pool measured worse on both latency and NDCG@5).",
        f"- **confidence scoring** ({CONFIDENCE_SCORING_LATENCY_MS:.1f} ms): design record 10 §6's real "
        "180-query dev sweep wall time (736s / 180 queries), at the shipped `rerank_pool_size=3` -- paid by "
        "every non-cache-hit query, unconditionally, to decide escalate-vs-generate "
        "(`guardrails/confidence.py::handle`). Deliberately not a per-pair estimate multiplied out: design "
        "record 10 §6 already found a smaller probe's per-pair mean undersold this real number by roughly "
        "4x.",
        f"- **generation** ({GENERATION_LATENCY_MS} ms, {GENERATION_COMPLETION_TOKENS} completion tokens): "
        "`docs/results/groq-model-replacement.md`'s real full 6/6-succeeded run at the current shipped "
        "config (`qwen/qwen3.8-27b`, `max_tokens=1000`).",
        f"- **exact-tier check/hit** ({EXACT_HIT_LATENCY_MS} ms) and **semantic-tier hit** "
        f"({SEMANTIC_HIT_LATENCY_MS:.3f} ms): `docs/results/cache-hit-rate-simulation.md` (Phase 12 Unit 3), "
        "mean across the 3 real Zipf skews measured there.",
        "",
        "## What's missing, disclosed rather than guessed at",
        "",
        "**Escalation's total cost is a real, partial lower bound, not a complete number.** "
        "`guardrails/confidence.py::handle` structurally cannot call both `generate_fn` and `escalate_fn` -- "
        "confirmed by its own test, a spy that raises if invoked -- so escalation never pays the generation "
        "cost above it, real savings. But `escalate_fn` itself (whatever actually happens on a real "
        "escalation -- writing to a queue, notifying a human agent, rendering the escalation UI) has no real "
        "implementation yet anywhere in this codebase; only `EscalationPayload`'s construction exists "
        "(`guardrails/confidence.py::build_escalation_payload`), which is already included in the measured "
        "confidence-scoring cost above. Phase 14 (API/serving) is where a real handler would be built. The "
        "escalation row above is real and correct as far as it goes, not a placeholder number, but it "
        "doesn't yet include whatever that eventual handler costs.",
        "",
        "**Rerank's effect on generation cost is assumed unchanged, not separately measured.** Reranking "
        "changes which chunks end up in the context, not how many tokens `context_budget_tokens` (4096) "
        "allows into the prompt, so the \"with rerank\" row reuses the same real generation number as the "
        "no-rerank row. A real difference in output length is plausible but unmeasured -- disclosed as an "
        "assumption, not reported as if it were its own measured figure.",
    ]

    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {RESULTS_MD}")
    for c in configs:
        print(f"{c['name']:<45} {c['total_latency_ms']:>10,.1f} ms   {c['tokens']:>6.1f} tokens")


if __name__ == "__main__":
    main()
