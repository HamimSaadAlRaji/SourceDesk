"""One place that assembles the shipped AnswerPipeline (Phase 6's dense retrieval,
Phase 10's confidence scorer, Phase 12's caches, Groq -> Gemini fallback).

Before this, experiments 13b/13c/13j/14a each re-typed the same wiring; the API
server is the fourth consumer, which is the point where copying stops being
cheaper than a function. Experiments that need throwaway caches pass `cache_dir`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from aegis.cache.exact_match import ExactMatchCache
from aegis.cache.semantic import SemanticCache
from aegis.config import settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.embedding.vector_store import get_client
from aegis.generation.llm import default_client
from aegis.generation.pipeline import GenerationPipeline
from aegis.graph.pipeline import LangGraphPipeline
from aegis.guardrails.citations import CitationValidator
from aegis.guardrails.confidence import ConfidenceScorer
from aegis.observability.tracing import get_tracer
from aegis.pipeline import AnswerPipeline
from aegis.query.rewrite import QueryRewriter
from aegis.rerank.cross_encoder import CrossEncoderReranker
from aegis.retrieval.dense import DenseIndex
from aegis.router.classifier import RouterClassifier
from aegis.tenancy import PartitionedRetriever, resolve_partition
from aegis.types import Chunk

# Placeholder until an IndexManifest exists (design record 13b SS3): cache keys
# carry it, so changing the corpus must change this string.
DEFAULT_CORPUS_VERSION = "dev"


def load_chunks() -> list[Chunk]:
    with open(settings.paths.data_processed / "chunks.jsonl", encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def pipeline_class(backend: str | None = None) -> type[AnswerPipeline]:
    """Which orchestration runs the request flow (Phase 17). `None` reads the
    setting at call time, so an env override or a test patch takes effect."""
    backend = backend or settings.orchestration.backend
    if backend == "direct":
        return AnswerPipeline
    if backend == "langgraph":
        return LangGraphPipeline
    raise ValueError(f"unknown orchestration backend {backend!r}; expected 'direct' or 'langgraph'")


def build_pipeline(
    cache_dir: Path | None = None,
    corpus_version: str = DEFAULT_CORPUS_VERSION,
    partition: str = settings.tenancy.default_partition,
    backend: str | None = None,
) -> AnswerPipeline:
    """`cache_dir=None` uses the real cache paths from settings; pass a directory
    for isolated (throwaway) caches. `partition` binds retrieval and both caches
    to one partition for the pipeline's lifetime (design record 15b)."""
    return build_pipelines([partition], cache_dir, corpus_version, backend)[partition]


def build_pipelines(
    partitions: Iterable[str],
    cache_dir: Path | None = None,
    corpus_version: str = DEFAULT_CORPUS_VERSION,
    backend: str | None = None,
) -> dict[str, AnswerPipeline]:
    """One pipeline per partition (design record 15c: a principal selects a
    pipeline, never a filter). Models, LLM clients, caches and the corpus are
    shared -- a second partition costs one more retriever wrapper, not a second
    bge-large. Sharing the cache files is safe because every key carries the
    partition (15b)."""
    encoder = Encoder(
        model_name=settings.embedding.bge_large_model,
        cache=EmbeddingCache(settings.paths.embedding_cache),
        device="cpu",
    )
    if cache_dir is None:
        exact_path, semantic_path = settings.cache.exact_match_db_path, settings.cache.semantic_cache_db_path
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        exact_path, semantic_path = cache_dir / "exact.sqlite", cache_dir / "semantic.sqlite"
    exact_cache = ExactMatchCache(exact_path)
    semantic_cache = SemanticCache(
        semantic_path, encoder=encoder, threshold=settings.cache.semantic_similarity_threshold
    ) if settings.cache.semantic_cache_enabled else None
    # Read-only key: the served process can search but never write (15f).
    dense = DenseIndex(load_chunks(), settings.qdrant.collection_bge_large, encoder, client=get_client(read_only=True))
    router = RouterClassifier()
    confidence = ConfidenceScorer(reranker=CrossEncoderReranker(), encode_fn=encoder.encode)
    generation = GenerationPipeline(llm=default_client(), validator=CitationValidator())  # openai -> groq -> gemini (15g)
    rewriter = QueryRewriter() if settings.query.rewrite_enabled else None
    tracer = get_tracer()
    cls = pipeline_class(backend)
    return {
        name: cls(
            exact_cache=exact_cache,
            semantic_cache=semantic_cache,
            router=router,
            retriever=PartitionedRetriever(dense, resolve_partition(name)),
            confidence_scorer=confidence,
            generation=generation,
            tracer=tracer,
            corpus_version=corpus_version,
            rewriter=rewriter,
            partition=name,
        )
        for name in dict.fromkeys(partitions)
    }
