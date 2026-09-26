"""Phase 9 sampling sweep (plan.md Phase 9, step 5): temperature x top_p
against faithfulness. "Expect low temperature to win; measure it anyway so
you can say so with data."

Faithfulness proxy: the citation validator's `valid_citation_rate` (fraction
of citation-requiring claims that end up supported, lexically or via
entailment). Real RAGAS faithfulness is Phase 11's job and needs a judge
model validated against hand labels first (plan.md Phase 11) -- this phase
doesn't have that yet, so it measures the signal Phase 9's own component
(the citation validator) actually produces, named explicitly as a proxy so
it's never confused with the Phase 11 number later.

Retrieval is computed once per query (Phase 6's winning config: dense,
heading-aware, bge-large, no enrichment) and reused across every sampling
config -- only generation varies with temperature/top_p, so re-retrieving
per config would waste compute measuring nothing new.

Cost control: a fixed sample of `N_QUERIES` dev positives (seeded,
reproducible), not the full 150 -- benchmarked first (single real call,
~0.77s generation + ~0.27s/entailment-check) before committing to a grid
size. The *first* attempt at this sweep used a 6-config x 20-query grid at
`max_tokens=1024` and hit a real wall mid-run: Groq's free tier caps
`llama-3.3-70b-versatile` at 100,000 tokens/day, and one config (temperature
0.0, top_p 0.5) alone used 95,944 of it (real ~4096-token contexts plus
completions that regularly used most of a 1024-token cap for a detailed
technical answer). Grid and `max_tokens` cut down here in response to that
measurement, not guessed in advance -- see design record 09 §5-6. The
entailment tier's model (`llama-3.1-8b-instant`) has a *separate* Groq
quota bucket and was never the constraint.

**2026-09-26 re-scope:** `llama-3.3-70b-versatile` is now retired entirely
(Open Question #15, Phase 11 Unit 1) -- `GroqSettings.model` (read here via
`GroqProvider()`'s default) is now `qwen/qwen3.8-27b`. Its real free-tier
caps are *per-minute*, not per-day (Phase 11 Unit 4: 1000 output-tokens/min,
7000 input-tokens/min), a fundamentally more forgiving shape for a sweep
like this than a once-a-day hard wall -- re-scoped `N_QUERIES` back up to 20
(matching the original v1 baseline's scale and Attempt 1's real N) and
added a fixed inter-call `SLEEP_SECONDS` pause to keep the real aggregate
input-tokens/minute comfortably under the 7000 ITPM cap given this corpus's
real per-chunk token stats (p50=79 tokens/chunk x 10 chunks ~= 790 median
case, but `context_budget_tokens=4096` is the real worst case). Because the
model changed, this run is **not** a continuation of Attempt 1's (0.0, 0.5)
number (0.6634, measured under the now-dead model) -- mixing two different
models within one "temperature x top_p" sweep would confound the variable
under test, so all 4 configs are re-measured fresh here.

Device forced to "cpu" for the retrieval encoder -- this project has hit
the MPS-under-memory-pressure hang three times already (Phases 3, 7, 8);
not repeating it a fourth time.

Each config's results are written to `RESULTS_RAW` incrementally, and a
per-query call is wrapped so one transient provider failure doesn't lose
an otherwise-real run -- the same "don't lose already-collected data to a
later crash" lesson `experiments/07b_kaggle_rerank_sweep.py` already
learned the hard way.
"""

from __future__ import annotations

import json
import random
import time

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.generation.llm.fallback import FallbackLLMClient
from aegis.generation.llm.groq_provider import GroqProvider
from aegis.generation.pipeline import GenerationPipeline
from aegis.guardrails.citations import CitationValidator
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, GoldenQuery, Query, Retrieved

# Cut from 20 queries x 6 configs (temp x top_p full grid) x max_tokens=1024
# after the first config alone used 95,944 of Groq's 100,000 tokens/day
# limit on llama-3.3-70b-versatile. 4 configs isolate both variables (temp
# 0.0/0.5/1.0 at top_p=1.0, plus top_p=0.5 at temp=0.0) at a token cost this
# session's remaining daily budget can actually clear.
N_QUERIES = 20  # restored to Attempt 1's scale -- qwen3.8-27b's per-minute caps allow it (see docstring)
CONFIGS = [(0.0, 0.5), (0.0, 1.0), (0.5, 1.0), (1.0, 1.0)]  # (temperature, top_p)
MAX_TOKENS = 400  # down from GroqSettings' default 1024 -- also a more realistic support-answer length
SLEEP_SECONDS = 8  # real ITPM-safety pacing -- see 2026-09-26 docstring note
RESULTS_MD = REPO_ROOT / "docs" / "results" / "sampling-sweep.md"
RESULTS_RAW = REPO_ROOT / "docs" / "results" / "sampling-sweep-raw.json"


def load_chunks() -> list[Chunk]:
    path = settings.paths.data_processed / "chunks.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def load_sample_queries(n: int) -> list[GoldenQuery]:
    path = settings.paths.data_golden / "golden_set_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        queries = [GoldenQuery(**json.loads(line)) for line in f]
    positives = [q for q in queries if not q.is_negative]
    return random.Random(settings.golden.random_seed).sample(positives, n)


def load_existing_results() -> tuple[list[dict], dict[tuple[float, float], dict]]:
    """Resumable by config, added 2026-09-26 -- a config already measured in
    full (`n_succeeded == N_QUERIES`) is real, already-paid-for data; without
    this, re-running the script to pick up the remaining configs would
    silently overwrite it, the exact "lose already-collected data" mistake
    this file's own docstring already names as a lesson learned elsewhere.
    """
    if not RESULTS_RAW.exists():
        return [], {}
    data = json.loads(RESULTS_RAW.read_text())
    raw = data.get("raw", [])
    by_config = {(s["temperature"], s["top_p"]): s for s in data.get("summaries", [])}
    return raw, by_config


def main() -> None:
    chunks = load_chunks()
    queries = load_sample_queries(N_QUERIES)
    print(f"{len(chunks)} chunks, {len(queries)} sampled dev positives")

    cache = EmbeddingCache(settings.paths.embedding_cache)
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=cache, device="cpu")
    dense_index = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)

    print("Retrieving once per query (shared across every sampling config)...")
    retrieved_by_query: dict[str, list[Retrieved]] = {}
    for q in queries:
        results = dense_index.search(q.text, top_k=settings.generation.retrieval_top_k, exclude_doc_id=q.source_issue)
        retrieved_by_query[q.id] = [Retrieved(chunk=c, score=s, rank=i + 1) for i, (c, s) in enumerate(results)]

    llm = FallbackLLMClient([GroqProvider()])  # single-provider on purpose -- isolate the sampling-param effect
    validator = CitationValidator()

    raw_results, existing_by_config = load_existing_results()
    config_summaries: list[dict] = list(existing_by_config.values())
    RESULTS_RAW.parent.mkdir(parents=True, exist_ok=True)

    for temperature, top_p in CONFIGS:
        existing = existing_by_config.get((temperature, top_p))
        if existing is not None and existing.get("n_succeeded", 0) >= N_QUERIES:
            print(f"temperature={temperature:.1f} top_p={top_p:.1f}: already complete (n_succeeded={existing['n_succeeded']}), skipping")
            continue
        if existing is not None:
            # A prior partial attempt at this exact config -- redoing it fresh
            # rather than accumulating, so drop its stale rows/summary first.
            raw_results[:] = [r for r in raw_results if not (r["temperature"] == temperature and r["top_p"] == top_p)]
            config_summaries[:] = [s for s in config_summaries if (s["temperature"], s["top_p"]) != (temperature, top_p)]

        pipeline = GenerationPipeline(
            llm=llm, validator=validator, temperature=temperature, top_p=top_p, max_tokens=MAX_TOKENS
        )
        rates: list[float] = []
        latencies_ms: list[float] = []
        claim_counts: dict[str, int] = {}
        failures = 0

        for q in queries:
            start = time.perf_counter()
            try:
                result = pipeline.answer(Query(id=q.id, text=q.text), retrieved_by_query[q.id])
            except Exception as exc:  # noqa: BLE001 -- log and continue, don't lose an otherwise-real run
                failures += 1
                print(f"  query {q.id} failed: {exc!r}")
                time.sleep(SLEEP_SECONDS)
                continue
            latencies_ms.append((time.perf_counter() - start) * 1000)
            time.sleep(SLEEP_SECONDS)
            rates.append(result.validation.valid_citation_rate)
            for claim in result.validation.claims:
                claim_counts[claim.status.value] = claim_counts.get(claim.status.value, 0) + 1

            raw_results.append(
                {
                    "query_id": q.id,
                    "temperature": temperature,
                    "top_p": top_p,
                    "valid_citation_rate": result.validation.valid_citation_rate,
                    "tier2_calls": result.validation.tier2_calls,
                    "latency_ms": latencies_ms[-1],
                    "answer_text": result.text,
                }
            )

        if rates:
            mean_rate = sum(rates) / len(rates)
            mean_latency = sum(latencies_ms) / len(latencies_ms)
        else:
            mean_rate = mean_latency = float("nan")
        print(
            f"temperature={temperature:.1f} top_p={top_p:.1f}  "
            f"mean_valid_citation_rate={mean_rate:.4f}  mean_latency_ms={mean_latency:.0f}  "
            f"failures={failures}  claims={claim_counts}"
        )
        config_summaries.append(
            {
                "temperature": temperature,
                "top_p": top_p,
                "mean_valid_citation_rate": mean_rate,
                "mean_latency_ms": mean_latency,
                "n_succeeded": len(rates),
                "n_failed": failures,
                "claim_status_counts": claim_counts,
            }
        )

        # Incremental write after every config -- a later config's failure
        # (e.g. hitting the daily token limit again) shouldn't cost the
        # configs already measured.
        RESULTS_RAW.write_text(json.dumps({"summaries": config_summaries, "raw": raw_results}, indent=2))

    print(f"\nRaw results written to {RESULTS_RAW}")

    write_report(config_summaries, len(queries))
    print(f"Report written to {RESULTS_MD}")


def write_report(config_summaries: list[dict], n_queries: int) -> None:
    lines = [
        "# Sampling Sweep — Temperature x top_p vs. Faithfulness Proxy",
        "",
        f"Phase 9, step 5. {n_queries} sampled dev-positive golden queries, Phase 6's winning retrieval config "
        f"(dense, heading-aware, bge-large, no enrichment), prompt v1, Groq `{settings.groq.model}` only "
        "(single-provider, to isolate the sampling-parameter effect from provider choice — that's step 6).",
        "",
        "Faithfulness proxy: `CitationValidator.valid_citation_rate` — fraction of citation-requiring claims "
        "that end up supported (lexically or via entailment). Not RAGAS faithfulness (Phase 11); named as a "
        "proxy deliberately.",
        "",
        "| temperature | top_p | mean valid_citation_rate | mean latency (ms) | claim status counts |",
        "|---|---|---|---|---|",
    ]
    for s in sorted(config_summaries, key=lambda x: (x["temperature"], x["top_p"])):
        counts = ", ".join(f"{k}={v}" for k, v in sorted(s["claim_status_counts"].items()))
        lines.append(
            f"| {s['temperature']:.1f} | {s['top_p']:.1f} | {s['mean_valid_citation_rate']:.4f} | "
            f"{s['mean_latency_ms']:.0f} | {counts} |"
        )
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
