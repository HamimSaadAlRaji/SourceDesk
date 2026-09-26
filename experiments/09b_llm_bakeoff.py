"""Phase 9 provider bake-off (plan.md Phase 9, step 6): "the concrete
version of 'choosing an LLM'" -- Groq (`llama-3.3-70b-versatile`) vs.
Gemini (`gemini-2.5-flash`) head to head on the same real retrieved context
for the same real golden queries, at the sampling sweep's best config
(temperature=0.0, per `docs/results/sampling-sweep.md` -- "expect low
temperature to win," and 0.0 is the only config that sweep got real,
well-powered data for).

Run once `GEMINI_API_KEY` landed (results: `docs/results/llm-bakeoff.md`).
Groq's daily quota was still mostly exhausted from the sampling sweep
earlier in the same session, so Groq only succeeded on 2/15 queries (13
real `RateLimitError`s) -- not a fair equal-N quality sample, but Gemini's
12/15 real successes, and the two queries both providers actually
answered, still produced a genuine, decision-relevant finding: Gemini's
mean latency (~21s) was ~14x Groq's, which is the real justification for
"Groq primary, Gemini secondary" rather than the reverse.

Same structure as `09a_sampling_sweep.py`: retrieval computed once per
query and shared across both providers, so only generation varies.
"""

from __future__ import annotations

import json
import random
import time

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.generation.llm.fallback import FallbackLLMClient
from aegis.generation.llm.gemini_provider import GeminiProvider
from aegis.generation.llm.groq_provider import GroqProvider
from aegis.generation.pipeline import GenerationPipeline
from aegis.guardrails.citations import CitationValidator
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, GoldenQuery, Query, Retrieved

N_QUERIES = 15  # sized against the sampling sweep's real measured per-call cost, not guessed
TEMPERATURE = 0.0  # sampling-sweep's only well-powered (N=20) config
TOP_P = 1.0
RESULTS_MD = REPO_ROOT / "docs" / "results" / "llm-bakeoff.md"
RESULTS_RAW = REPO_ROOT / "docs" / "results" / "llm-bakeoff-raw.json"


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


def run_provider(name: str, provider, queries, retrieved_by_query, validator) -> tuple[list[dict], dict]:
    pipeline = GenerationPipeline(
        llm=FallbackLLMClient([provider]), validator=validator, temperature=TEMPERATURE, top_p=TOP_P
    )
    raw: list[dict] = []
    rates: list[float] = []
    latencies_ms: list[float] = []
    failures = 0

    for q in queries:
        start = time.perf_counter()
        try:
            result = pipeline.answer(Query(id=q.id, text=q.text), retrieved_by_query[q.id])
        except Exception as exc:  # noqa: BLE001 -- log and continue, don't lose an otherwise-real run
            failures += 1
            print(f"  [{name}] query {q.id} failed: {exc!r}")
            continue
        latency = (time.perf_counter() - start) * 1000
        latencies_ms.append(latency)
        rates.append(result.validation.valid_citation_rate)
        raw.append(
            {
                "provider": name,
                "query_id": q.id,
                "valid_citation_rate": result.validation.valid_citation_rate,
                "tier2_calls": result.validation.tier2_calls,
                "latency_ms": latency,
                "answer_text": result.text,
            }
        )

    summary = {
        "provider": name,
        "n_succeeded": len(rates),
        "n_failed": failures,
        "mean_valid_citation_rate": sum(rates) / len(rates) if rates else float("nan"),
        "mean_latency_ms": sum(latencies_ms) / len(latencies_ms) if latencies_ms else float("nan"),
    }
    print(f"[{name}] {summary}")
    return raw, summary


def main() -> None:
    chunks = load_chunks()
    queries = load_sample_queries(N_QUERIES)
    print(f"{len(chunks)} chunks, {len(queries)} sampled dev positives")

    cache = EmbeddingCache(settings.paths.embedding_cache)
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=cache, device="cpu")
    dense_index = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)

    print("Retrieving once per query (shared across both providers)...")
    retrieved_by_query: dict[str, list[Retrieved]] = {}
    for q in queries:
        results = dense_index.search(q.text, top_k=settings.generation.retrieval_top_k, exclude_doc_id=q.source_issue)
        retrieved_by_query[q.id] = [Retrieved(chunk=c, score=s, rank=i + 1) for i, (c, s) in enumerate(results)]

    validator = CitationValidator()

    all_raw: list[dict] = []
    summaries: list[dict] = []
    for name, provider in [("groq", GroqProvider()), ("gemini", GeminiProvider())]:
        raw, summary = run_provider(name, provider, queries, retrieved_by_query, validator)
        all_raw.extend(raw)
        summaries.append(summary)
        RESULTS_RAW.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_RAW.write_text(json.dumps({"summaries": summaries, "raw": all_raw}, indent=2))

    print(f"\nRaw results written to {RESULTS_RAW}")


if __name__ == "__main__":
    main()
