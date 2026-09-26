# 12f — Cost Per Query, Per Configuration

**Phase:** 12 · **Date:** 2026-09-26 · **Files:** `experiments/12d_cost_model.py`
**Topics from the course:** Cost vs response quality (topic 34) — the whole phase's throughline (Units 1-2's per-request tradeoff, Unit 3's aggregate savings, Unit 5's size/latency curve) closed out with the one number an interviewer actually asks for: what does a query cost, by configuration.

## 1. What I built

`experiments/12d_cost_model.py`: assembles five per-query cost profiles (cache miss with rerank, cache miss without rerank, semantic cache hit, exact cache hit, escalation) out of real numbers already measured and published across this project — zero new measurements, zero new LLM calls. Output: `docs/results/cost-per-query.md`.

## 2. Why this approach

Every component this table needs was already measured somewhere else in this project: retrieval latency (Phase 3), rerank latency (Phase 7), confidence-scoring latency (Phase 10), generation latency and tokens (Phase 9/11), cache-hit latency (Phase 12 Units 1-3). The genuinely new work in this unit isn't measurement, it's assembly — and assembly done by hand in a markdown table is exactly where a real arithmetic or citation error would hide silently. A short script that names each source explicitly and adds the components programmatically is cheap insurance against that, and gives future units (or the user) one place to update if any upstream number changes.

Reporting tokens and latency rather than dollars is a deliberate, disclosed scope decision, not an oversight: this project has never left the Groq/Gemini free tiers, so there is no real dollar figure to report, and inventing one from a hypothetical paid-tier price sheet would be exactly the fabricated number CLAUDE.md forbids. Tokens are the honest stand-in — they're what has actually run out and blocked real work this session (Phase 9/11's deferred-to-the-user reruns are a direct consequence of hitting real token quota walls), so they're the currency that would actually matter if this system had a real user base on a metered plan.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ----------------- |
| Assign a hypothetical $/token rate (e.g., a public API price sheet) and report dollars | Dollars are the unit people expect in a cost table | Would be a number this project never measured or paid — exactly the "never invent a metric value" line CLAUDE.md draws. Tokens are real; a dollar conversion of them wouldn't be |
| Derive confidence-scoring cost from the per-pair mean (~320ms) x `rerank_pool_size` (3) = ~960ms | Simpler, uses a number already in `config.py`'s own comments | Design record 10 §6 already measured the real full-sweep number (4,088.9 ms/query) and found the per-pair estimate undersells it by roughly 4x — using the smaller estimate here would re-introduce a mistake this project already caught and documented once |
| Treat escalation's cost as "free" (no generation call, so it's just the savings) | Simpler headline | Confidence scoring still runs unconditionally before the escalate/generate fork — reporting escalation as free would hide a real, measured 4.1s cost that exists whether or not generation ever happens |

## 4. How it works

Each configuration is a list of named `(component, latency_ms)` pairs, summed once, so the total is never hand-added twice in two different documents:

```python
configs = [
    {"name": "Cache miss, no rerank (shipped default)",
     "components": [("retrieval", RETRIEVAL_LATENCY_MS),
                     ("confidence scoring", CONFIDENCE_SCORING_LATENCY_MS),
                     ("generation", GENERATION_LATENCY_MS)],
     "tokens": GENERATION_COMPLETION_TOKENS},
    ...
]
for c in configs:
    c["total_latency_ms"] = sum(v for _, v in c["components"])
```

The one non-trivial mechanism worth stating plainly: confidence scoring is not an escalation-only cost. `guardrails/confidence.py::handle` runs the confidence scorer unconditionally on every non-cache-hit query — generation and escalation are what happens *after* that shared cost, not alternatives to paying it. Both "cache miss" rows and the "escalation" row include it for that reason.

## 5. What the numbers said

| configuration | total latency | completion tokens |
|---|---|---|
| Cache miss, no rerank (shipped default) | 12,787.5 ms | 254.7 |
| Cache miss, with rerank (pool=10, not shipped) | 14,347.5 ms | 254.7 |
| Semantic cache hit | 0.8 ms | 0.0 |
| Exact cache hit | 0.0 ms | 0.0 |
| Escalation (partial — see design record and results doc for what's not yet included) | 4,095.5 ms | 0.0 |

Full component breakdown and sources: `docs/results/cost-per-query.md`.

## 6. What surprised me / what to watch for

**Escalation isn't just cheaper than generating an answer — it's roughly 3x faster, and it's the only non-cache-hit path with zero token cost.** 4.1s vs. 12.8s is a bigger gap than "no generation call" alone would suggest, because confidence scoring (the cost escalation *does* pay) is itself dominated by three small local cross-encoder calls, not a network round trip to a rate-limited provider — real, local compute is cheap and fast next to a real Groq call, even one that's the fastest LLM path this project measured. Worth remembering for the North Star framing: the system's honest "I don't know" path being both fast and quota-free removes any cost-based argument for ever preferring a confident guess over abstention.

**The confidence-scoring number (4,088.9 ms) is bigger than a reasonable-looking back-of-envelope estimate (960 ms) by 4x, and this project already has a documented reason why** (design record 10 §6: per-pair latency variance doesn't average out over a small probe). Reusing the real full-sweep number instead of re-deriving a smaller, wrong one from `config.py`'s own per-pair comment was a deliberate choice after almost making exactly that mistake while drafting this unit — a good reminder that a number living in a comment isn't automatically the most authoritative real number available for a given question.

## 7. Where this fits and what's next

Closes plan.md's Phase 12 step 6 exit criterion (cost-per-query table) and the phase's last open build item. `docs/results/cache-performance.md` (the phase-level synthesis the exit criteria also name) ties this together with Units 1-5's results. Phase 13 (observability) is next — its per-stage latency attribution will be the first place these numbers get checked against real traced requests instead of assembled from separate experiments.
