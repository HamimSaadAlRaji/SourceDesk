"""Phase 13 Unit 4 (continued) -- localize the confidence-stage gap.

Prior experiments (13e-13g) used different queries per arm and only timed the
stage total. Here: the SAME queries in every arm, production-like setup (real
Groq router before the stage, real Langfuse tracer around it, fake generation
so no generation quota is spent), varying only the idle gap between queries,
and sub-timing the two things confidence scoring does -- the cross-encoder
call and the chunk-embedding call -- so the extra time can be attributed.

Arm order is fixed (10 s, 0 s, 70 s) and reported; a residual order effect is
possible and is disclosed with the results, not assumed away.
"""

from __future__ import annotations

import json
import os
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
from aegis.router.classifier import RouterClassifier
from aegis.types import Chunk, GoldenQuery, Query

N = 8
GAPS = [10, 0, 70]
OUT = REPO_ROOT / "docs" / "results" / os.environ.get("AEGIS_H_OUT", "confidence-localize-raw.json")


class FakeLLM:
    def generate_with_provider(self, messages, temperature, top_p, max_tokens):
        return "a fake answer [1]", "fake"


class TimedReranker:
    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def rerank(self, query, candidates, top_k):
        t = time.perf_counter()
        out = self.inner.rerank(query, candidates, top_k)
        self.calls.append((time.perf_counter() - t) * 1000.0)
        return out


def main() -> int:
    with open(settings.paths.data_processed / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f]
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        dev = [q for q in (GoldenQuery(**json.loads(l)) for l in f) if q.split == "dev" and not q.is_negative]
    queries = random.Random(settings.golden.random_seed).sample(dev, N)

    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
    encode_calls: list[float] = []

    def timed_encode(texts):
        t = time.perf_counter()
        out = encoder.encode(texts)
        encode_calls.append((time.perf_counter() - t) * 1000.0)
        return out

    reranker = TimedReranker(CrossEncoderReranker())
    scorer = ConfidenceScorer(reranker=reranker, encode_fn=timed_encode)
    dense = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)
    validator = CitationValidator(entailment_fn=lambda claim, chunk_text: True)

    def build(tag: str) -> AnswerPipeline:
        for n in (f"phase13h_{tag}_exact.sqlite", f"phase13h_{tag}_sem.sqlite"):
            (REPO_ROOT / "data" / "cache" / n).unlink(missing_ok=True)
        return AnswerPipeline(
            exact_cache=ExactMatchCache(REPO_ROOT / "data" / "cache" / f"phase13h_{tag}_exact.sqlite"),
            semantic_cache=SemanticCache(REPO_ROOT / "data" / "cache" / f"phase13h_{tag}_sem.sqlite", encoder=encoder, threshold=settings.cache.semantic_similarity_threshold),
            router=RouterClassifier(), retriever=dense, confidence_scorer=scorer,
            generation=GenerationPipeline(llm=FakeLLM(), validator=validator),
            tracer=get_tracer(), corpus_version="phase13h",
        )

    first = build("warm")
    print(f"warm_up: {first.warm_up()}", flush=True)

    rows = []
    for gap in GAPS:
        pipe = build(f"gap{gap}")
        for i, q in enumerate(queries):
            if gap and i > 0:
                time.sleep(gap)
            reranker.calls.clear()
            encode_calls.clear()
            r = pipe.answer(Query(id=q.id, text=q.text))
            row = {
                "gap_s": gap, "i": i, "query_id": q.id,
                "confidence_stage_ms": r.stage_latencies_ms["confidence"],
                "rerank_ms": sum(reranker.calls), "encode_ms": sum(encode_calls),
                "routing_ms": r.stage_latencies_ms["routing"],
            }
            row["other_ms"] = row["confidence_stage_ms"] - row["rerank_ms"] - row["encode_ms"]
            rows.append(row)
            print(f"gap={gap:2d}s q{i} stage={row['confidence_stage_ms']:6.0f} rerank={row['rerank_ms']:6.0f} encode={row['encode_ms']:5.0f} other={row['other_ms']:5.0f} routing={row['routing_ms']:5.0f}", flush=True)
    OUT.write_text(json.dumps(rows, indent=2))

    print("\nsummary (queries 1..N-1 per arm; query 0 has no preceding gap):")
    for gap in GAPS:
        g = [r for r in rows if r["gap_s"] == gap and r["i"] > 0]
        f = lambda k: f"p50={np.percentile([r[k] for r in g],50):5.0f} max={max(r[k] for r in g):5.0f}"
        print(f"gap={gap:2d}s n={len(g)}: stage {f('confidence_stage_ms')} | rerank {f('rerank_ms')} | encode {f('encode_ms')} | other {f('other_ms')}")
    get_tracer().flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
