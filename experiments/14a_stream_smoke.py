"""Phase 14 Unit 1 -- one real streamed request through the real pipeline: real
Groq router + generation, real Qdrant, real cross-encoder. Prints each event
with its arrival time so incremental delivery (and validation happening AFTER
the tokens, per the 2026-10-01 decision) is visible, then replays the same
query to show a cache hit emitting the same event shapes. Throwaway caches."""

from __future__ import annotations

import json
import sys
import time

from aegis.cache.exact_match import ExactMatchCache
from aegis.cache.semantic import SemanticCache
from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.events import CitationEvent, DoneEvent, ErrorEvent, TokenEvent
from aegis.generation.llm.fallback import FallbackLLMClient
from aegis.generation.llm.gemini_provider import GeminiProvider
from aegis.generation.llm.groq_provider import GroqProvider
from aegis.generation.pipeline import GenerationPipeline
from aegis.guardrails.citations import CitationValidator
from aegis.guardrails.confidence import ConfidenceScorer
from aegis.observability.logging import configure_logging
from aegis.observability.tracing import get_tracer
from aegis.pipeline import AnswerPipeline
from aegis.rerank.cross_encoder import CrossEncoderReranker
from aegis.retrieval.dense import DenseIndex
from aegis.router.classifier import RouterClassifier
from aegis.types import Chunk, GoldenQuery, Query

QID = "golden::issue::604"


def show(label: str, pipeline: AnswerPipeline, query: Query) -> list:
    print(f"\n=== {label} ===")
    t0 = time.perf_counter()
    events = []
    for e in pipeline.answer_stream(query):
        ms = (time.perf_counter() - t0) * 1000
        events.append((ms, e))
        if isinstance(e, TokenEvent):
            print(f"{ms:7.0f} ms  token     {e.text!r}")
        elif isinstance(e, CitationEvent):
            print(f"{ms:7.0f} ms  citation  [{e.citation_id}] {e.chunk_id}  ({len(e.text)} chars of source text)")
        elif isinstance(e, DoneEvent):
            print(f"{ms:7.0f} ms  done      cache={e.cache_status} route={e.route} escalate={e.escalate} provider={e.provider} citations={e.citations}")
        elif isinstance(e, ErrorEvent):
            print(f"{ms:7.0f} ms  ERROR     code={e.code} message={e.message!r}")
        else:
            print(f"{ms:7.0f} ms  {e.type}  {e}")
    return events


def main() -> int:
    configure_logging(level=40)  # errors only; this script prints its own timeline
    with open(settings.paths.data_processed / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f]
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        q = next(g for g in (GoldenQuery(**json.loads(l)) for l in f) if g.id == QID)

    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
    for n in ("phase14a_exact.sqlite", "phase14a_sem.sqlite"):
        (REPO_ROOT / "data" / "cache" / n).unlink(missing_ok=True)
    pipeline = AnswerPipeline(
        exact_cache=ExactMatchCache(REPO_ROOT / "data" / "cache" / "phase14a_exact.sqlite"),
        semantic_cache=SemanticCache(REPO_ROOT / "data" / "cache" / "phase14a_sem.sqlite", encoder=encoder, threshold=settings.cache.semantic_similarity_threshold),
        router=RouterClassifier(), retriever=DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder),
        confidence_scorer=ConfidenceScorer(reranker=CrossEncoderReranker(), encode_fn=encoder.encode),
        generation=GenerationPipeline(llm=FallbackLLMClient([GroqProvider(), GeminiProvider()]), validator=CitationValidator()),
        tracer=get_tracer(), corpus_version="phase14a",
    )
    print(f"warm_up: {pipeline.warm_up()}")

    query = Query(id=q.id, text=q.text)
    first = show("live request (cache miss)", pipeline, query)
    show("same query again (exact-cache hit)", pipeline, query)
    get_tracer().flush()

    toks = [(ms, e) for ms, e in first if isinstance(e, TokenEvent)]
    cits = [ms for ms, e in first if isinstance(e, CitationEvent)]
    if toks:
        print(f"\nfirst token at {toks[0][0]:.0f} ms, last token at {toks[-1][0]:.0f} ms ({len(toks)} token events)"
              + (f", first citation at {cits[0]:.0f} ms" if cits else ", no validated citations"))
    return 0 if any(isinstance(e, DoneEvent) for _, e in first) else 1


if __name__ == "__main__":
    sys.exit(main())
