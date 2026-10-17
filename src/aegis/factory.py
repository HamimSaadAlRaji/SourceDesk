"""One place that assembles the shipped AnswerPipeline (Phase 6's dense retrieval,
Phase 10's confidence scorer, Phase 12's caches, Groq -> Gemini fallback).

Before this, experiments 13b/13c/13j/14a each re-typed the same wiring; the API
server is the fourth consumer, which is the point where copying stops being
cheaper than a function. Experiments that need throwaway caches pass `cache_dir`.
"""

from __future__ import annotations

import json
from pathlib import Path

from aegis.cache.exact_match import ExactMatchCache
from aegis.cache.semantic import SemanticCache
from aegis.config import settings
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
from aegis.query.rewrite import QueryRewriter
from aegis.rerank.cross_encoder import CrossEncoderReranker
from aegis.retrieval.dense import DenseIndex
from aegis.router.classifier import RouterClassifier
from aegis.types import Chunk

# Placeholder until an IndexManifest exists (design record 13b SS3): cache keys
# carry it, so changing the corpus must change this string.
DEFAULT_CORPUS_VERSION = "dev"


def load_chunks() -> list[Chunk]:
    with open(settings.paths.data_processed / "chunks.jsonl", encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def build_pipeline(cache_dir: Path | None = None, corpus_version: str = DEFAULT_CORPUS_VERSION) -> AnswerPipeline:
    """`cache_dir=None` uses the real cache paths from settings; pass a directory
    for isolated (throwaway) caches."""
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
    return AnswerPipeline(
        exact_cache=ExactMatchCache(exact_path),
        semantic_cache=SemanticCache(
            semantic_path, encoder=encoder, threshold=settings.cache.semantic_similarity_threshold
        ) if settings.cache.semantic_cache_enabled else None,
        router=RouterClassifier(),
        retriever=DenseIndex(load_chunks(), settings.qdrant.collection_bge_large, encoder),
        confidence_scorer=ConfidenceScorer(reranker=CrossEncoderReranker(), encode_fn=encoder.encode),
        generation=GenerationPipeline(
            llm=FallbackLLMClient([GroqProvider(), GeminiProvider()]), validator=CitationValidator()
        ),
        tracer=get_tracer(),
        corpus_version=corpus_version,
        rewriter=QueryRewriter() if settings.query.rewrite_enabled else None,
    )
