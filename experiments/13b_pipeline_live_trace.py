"""Phase 13 Unit 2 -- run one real query through the real `AnswerPipeline`
(real Qdrant retrieval, real cross-encoder reranker, real router/generation
LLM calls) and confirm every stage's span actually landed in Langfuse
Cloud. This is the first real end-to-end request this codebase has ever
made -- every prior phase tested its own stage in isolation.

Deliberately uses throwaway cache db paths (`data/cache/phase13_*`), never
the real `settings.cache.*_db_path` files Phase 12's own measurements
depend on, and a placeholder `corpus_version` (design record 13b SS3 --
`IndexManifest.corpus_version()` has no real data yet, see progress.md's
disclosed Phase 12 open item).
"""

from __future__ import annotations

import json
import sys
import time

from aegis.config import REPO_ROOT, settings
from aegis.cache.exact_match import ExactMatchCache
from aegis.cache.semantic import SemanticCache
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.generation.llm.fallback import FallbackLLMClient
from aegis.generation.llm.gemini_provider import GeminiProvider
from aegis.generation.llm.groq_provider import GroqProvider
from aegis.generation.pipeline import GenerationPipeline
from aegis.guardrails.citations import CitationValidator
from aegis.guardrails.confidence import ConfidenceScorer
from aegis.observability.tracing import get_tracer
from aegis.pipeline import AnswerPipeline
from aegis.rerank.cross_encoder import CrossEncoderReranker
from aegis.retrieval.dense import DenseIndex
from aegis.router.classifier import RouterClassifier
from aegis.types import Chunk, GoldenQuery, Query

CORPUS_VERSION = "phase13-live-trace-v1"  # explicit placeholder, not IndexManifest -- see module docstring


def load_chunks() -> list[Chunk]:
    path = settings.paths.data_processed / "chunks.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def load_one_real_query() -> Query:
    path = settings.paths.data_golden / "golden_set_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        queries = [GoldenQuery(**json.loads(line)) for line in f]
    positive = next(q for q in queries if not q.is_negative and q.ground_truth_chunk_ids)
    return Query(id=positive.id, text=positive.text)


def main() -> int:
    chunks = load_chunks()
    query = load_one_real_query()
    print(f"{len(chunks)} chunks loaded; real query: {query.id!r} -> {query.text!r}")

    embedding_cache = EmbeddingCache(settings.paths.embedding_cache)
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=embedding_cache, device="cpu")
    dense_index = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)
    reranker = CrossEncoderReranker()

    pipeline = AnswerPipeline(
        exact_cache=ExactMatchCache(REPO_ROOT / "data" / "cache" / "phase13_exact.sqlite"),
        semantic_cache=SemanticCache(
            REPO_ROOT / "data" / "cache" / "phase13_semantic.sqlite",
            encoder=encoder,
            threshold=settings.cache.semantic_similarity_threshold,
        ),
        router=RouterClassifier(),
        retriever=dense_index,
        confidence_scorer=ConfidenceScorer(reranker=reranker, encode_fn=encoder.encode),
        generation=GenerationPipeline(
            llm=FallbackLLMClient([GroqProvider(), GeminiProvider()]),
            validator=CitationValidator(),
        ),
        tracer=get_tracer(),
        corpus_version=CORPUS_VERSION,
    )

    result = pipeline.answer(query)

    print(f"trace_id: {result.trace_id}")
    print(f"trace_url: {get_tracer().get_trace_url(trace_id=result.trace_id)}")
    print(f"cache_status: {result.cache_status}")
    print(f"route: {result.route}")
    print(f"escalate: {result.escalate}")
    if result.confidence is not None:
        print(f"confidence: {result.confidence.confidence:.4f}")
    print(f"provider: {result.provider}")
    print(f"citations: {result.citations}")
    print(f"text: {result.text[:300]}")

    get_tracer().flush()

    # Ingestion latency scales with real trace size -- Unit 1's tiny synthetic
    # 3-span trace was queryable at 8s, but a real 7-span trace still had 4
    # missing at 8s and was complete by ~14s. Poll instead of guessing a
    # fixed wait. Expected spans are branch-aware: a cache hit (this cache
    # is throwaway but persists across runs of this script, so a second run
    # against the same query legitimately short-circuits after 2 spans --
    # that's the pipeline working correctly, not a smaller trace to chase).
    if result.cache_status.value != "miss":
        expected = {"answer_query", "cache_lookup"}
    elif result.escalate:
        expected = {"answer_query", "cache_lookup", "routing"}
    else:
        expected = {"answer_query", "cache_lookup", "routing", "retrieval", "confidence", "generation", "validation"}
    names: list[str] = []
    for attempt in range(6):
        time.sleep(5)
        fetched = get_tracer().api.observations.get_many(trace_id=result.trace_id)
        names = sorted(o.name for o in fetched.data)
        print(f"poll {attempt + 1}: {names}")
        if expected.issubset(set(names)):
            break

    if not expected.issubset(set(names)):
        print("Real observations were fetched but some stages are still missing -- investigate before trusting tracing.")
        return 1

    print("Real end-to-end request traced and confirmed live in Langfuse Cloud.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
