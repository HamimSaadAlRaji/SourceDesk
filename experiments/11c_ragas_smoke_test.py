"""Phase 11 Unit 1's actual goal (blocked since 2026-08-18, unblocked this
session by the Groq model swap in `config.py` -- see progress.md Open
Question #15): run RAGAS's `Faithfulness`/`ResponseRelevancy` (LLM-backed,
via `eval/ragas_adapters.py`) and `IDBasedContextPrecision`/
`IDBasedContextRecall` (pure ID-set arithmetic, no LLM) against one real
generated answer, to get a real per-query LLM-call cost before committing
to the N=30 batch (Unit 4).

One real dev-positive query, real dense retrieval (Phase 6 winning config),
real Groq generation (`qwen/qwen3.8-27b`, post-swap), real ground-truth
chunk IDs from the golden set for the ID-based metrics' `reference_context_ids`.
"""

from __future__ import annotations

import asyncio
import json
import random
import time

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, IDBasedContextPrecision, IDBasedContextRecall, ResponseRelevancy

from aegis.config import settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.eval.ragas_adapters import RagasLocalEmbeddings, build_groq_ragas_llm
from aegis.generation.llm.fallback import FallbackLLMClient
from aegis.generation.llm.groq_provider import GroqProvider
from aegis.generation.pipeline import GenerationPipeline
from aegis.guardrails.citations import CitationValidator
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, GoldenQuery, Query, Retrieved


def load_chunks() -> list[Chunk]:
    path = settings.paths.data_processed / "chunks.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def load_sample_query() -> GoldenQuery:
    path = settings.paths.data_golden / "golden_set_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        queries = [GoldenQuery(**json.loads(line)) for line in f]
    positives = [q for q in queries if not q.is_negative and q.ground_truth_chunk_ids]
    return random.Random(settings.golden.random_seed).sample(positives, 1)[0]


async def main() -> None:
    chunks = load_chunks()
    q = load_sample_query()
    print(f"query: {q.id!r} — {q.text[:100]!r}")
    print(f"ground_truth_chunk_ids: {len(q.ground_truth_chunk_ids)} chunks")

    cache = EmbeddingCache(settings.paths.embedding_cache)
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=cache, device="cpu")
    dense_index = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)
    results = dense_index.search(q.text, top_k=settings.generation.retrieval_top_k, exclude_doc_id=q.source_issue)
    retrieved = [Retrieved(chunk=c, score=s, rank=i + 1) for i, (c, s) in enumerate(results)]

    pipeline = GenerationPipeline(llm=FallbackLLMClient([GroqProvider()]), validator=CitationValidator())
    gen_start = time.perf_counter()
    result = pipeline.answer(Query(id=q.id, text=q.text), retrieved)
    gen_latency_ms = (time.perf_counter() - gen_start) * 1000
    print(f"generation: provider={result.provider} latency_ms={gen_latency_ms:.0f} valid_citation_rate={result.validation.valid_citation_rate}")
    print(f"answer (first 300 chars): {result.text[:300]!r}")

    sample = SingleTurnSample(
        user_input=q.text,
        response=result.text,
        retrieved_contexts=[pc.chunk.text for pc in result.context.chunks],
        retrieved_context_ids=[pc.chunk.id for pc in result.context.chunks],
        reference_context_ids=q.ground_truth_chunk_ids,
    )

    ragas_llm = build_groq_ragas_llm()
    ragas_embeddings = RagasLocalEmbeddings(encoder)

    metrics = {
        "faithfulness": Faithfulness(llm=ragas_llm),
        "response_relevancy": ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        "id_based_context_precision": IDBasedContextPrecision(),
        "id_based_context_recall": IDBasedContextRecall(),
    }

    scores: dict[str, float] = {}
    latencies_ms: dict[str, float] = {}
    for name, metric in metrics.items():
        start = time.perf_counter()
        try:
            score = await metric.single_turn_ascore(sample)
        except Exception as exc:  # noqa: BLE001 -- log and continue, don't lose the other metrics
            print(f"  [{name}] FAILED: {exc!r}")
            continue
        latencies_ms[name] = (time.perf_counter() - start) * 1000
        scores[name] = score
        print(f"  [{name}] score={score:.4f} latency_ms={latencies_ms[name]:.0f}")

    print(f"\nscores: {scores}")
    print(f"latencies_ms: {latencies_ms}")


if __name__ == "__main__":
    asyncio.run(main())
