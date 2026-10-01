"""Phase 13 Unit 4 -- does running the real Langfuse tracer inside the
pipeline slow the `confidence` stage?

Unit 3/4 found the confidence stage costs ~2.4 s inside `AnswerPipeline` but
~0.56 s (back-to-back) / ~1.0 s (70 s idle) when `ConfidenceScorer` is timed
alone (13e/13f). The pipeline differs by: real router LLM call before it, and
the real Langfuse tracer around every stage. This isolates the tracer: same
real retriever/reranker/scorer/caches, but router and LLM are fakes (zero
quota spend) and the tracer is either the real one or a no-op.
"""

from __future__ import annotations

import contextlib
import json
import random
import sys
import time

import numpy as np

from aegis.cache.exact_match import ExactMatchCache
from aegis.cache.semantic import SemanticCache
from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.generation.pipeline import GenerationPipeline
from aegis.guardrails.citations import CitationValidator
from aegis.guardrails.confidence import ConfidenceScorer
from aegis.observability.tracing import get_tracer
from aegis.pipeline import AnswerPipeline
from aegis.rerank.cross_encoder import CrossEncoderReranker
from aegis.retrieval.dense import DenseIndex
from aegis.router.classifier import Route
from aegis.types import Chunk, GoldenQuery, Query

N = 12


class NoopTracer:
    def create_trace_id(self) -> str:
        return "noop"

    @contextlib.contextmanager
    def start_as_current_observation(self, **kwargs):
        class _S:
            def update(self, **f): ...

        yield _S()


class FixedRouter:
    def classify(self, query: str) -> Route:
        return Route.DOCS_HOWTO


class FakeLLM:
    def generate_with_provider(self, messages, temperature, top_p, max_tokens):
        return "a fake answer [1]", "fake"


def main() -> int:
    with open(settings.paths.data_processed / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f]
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        dev = [q for q in (GoldenQuery(**json.loads(l)) for l in f) if q.split == "dev" and not q.is_negative]
    queries = random.Random(settings.golden.random_seed).sample(dev, 2 * N + 1)

    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
    dense = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)
    scorer = ConfidenceScorer(reranker=CrossEncoderReranker(), encode_fn=encoder.encode)
    validator = CitationValidator(entailment_fn=lambda claim, chunk_text: True)

    def build(tracer, tag: str) -> AnswerPipeline:
        for n in (f"phase13g_{tag}_exact.sqlite", f"phase13g_{tag}_sem.sqlite"):
            (REPO_ROOT / "data" / "cache" / n).unlink(missing_ok=True)
        return AnswerPipeline(
            exact_cache=ExactMatchCache(REPO_ROOT / "data" / "cache" / f"phase13g_{tag}_exact.sqlite"),
            semantic_cache=SemanticCache(REPO_ROOT / "data" / "cache" / f"phase13g_{tag}_sem.sqlite", encoder=encoder, threshold=settings.cache.semantic_similarity_threshold),
            router=FixedRouter(), retriever=dense, confidence_scorer=scorer,
            generation=GenerationPipeline(llm=FakeLLM(), validator=validator),
            tracer=tracer, corpus_version="phase13g",
        )

    noop, real = build(NoopTracer(), "noop"), build(get_tracer(), "real")
    print(f"warm_up: {noop.warm_up()}")

    def run(pipe, qs) -> list[float]:
        out = []
        for q in qs:
            r = pipe.answer(Query(id=q.id, text=q.text))
            out.append(r.stage_latencies_ms["confidence"])
        return out

    a = run(noop, queries[1 : 1 + N])
    b = run(real, queries[1 + N : 1 + 2 * N])
    print(f"noop tracer n={len(a)}: confidence p50={np.percentile(a,50):.0f} p95={np.percentile(a,95):.0f}  {[round(x) for x in a]}")
    print(f"real tracer n={len(b)}: confidence p50={np.percentile(b,50):.0f} p95={np.percentile(b,95):.0f}  {[round(x) for x in b]}")
    get_tracer().flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
