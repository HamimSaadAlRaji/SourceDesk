"""Phase 13 Unit 3 -- per-stage latency attribution: p50/p95/p99 per stage of
the real `AnswerPipeline` over a real batch of dev queries, so "it's slow"
becomes "confidence scoring is X ms of a Y ms p95" (plan.md Phase 13 build
item 3).

Real everything: shipped retrieval (dense, Phase 6's winning config), real
cross-encoder confidence scoring, real router + generation LLM calls. Caches
are freshly emptied so every query takes the miss path (cache-hit latency is
a separate, already-measured number: Phase 12).

Sample: N_POSITIVES dev positives + N_NEGATIVES dev negatives, seeded. The
mix is deliberate -- negatives mostly escalate (no generation), so stage
sample counts differ per stage and are reported, not hidden.

The FIRST query is the cold start (lazy model loads, connection setup) and is
reported separately, excluded from percentiles -- it's the number the
"warm models at startup" optimization (build item 4) is about.

Checkpointed per query and resumable, like every other LLM-quota-touching
script in this project. Per-query LLM failures are recorded, not fatal.
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

N_POSITIVES = int(os.environ.get("AEGIS_LAT_N_POS", "20"))
N_NEGATIVES = int(os.environ.get("AEGIS_LAT_N_NEG", "5"))
# Seconds to idle between queries. 0 = back-to-back (a burst, which trips the
# free-tier per-minute token caps -- see design record 13c). ~70 lets the
# per-minute window fully reset so each query sees an unthrottled provider.
PACE_SECONDS = float(os.environ.get("AEGIS_LAT_PACE_SECONDS", "0"))
# Call AnswerPipeline.warm_up() before the first query (Phase 13 Unit 4).
WARM_UP = os.environ.get("AEGIS_LAT_WARMUP", "0") == "1"
CORPUS_VERSION = "phase13-latency-v1"  # explicit placeholder, see design record 13b SS3
RAW_PATH = REPO_ROOT / "docs" / "results" / os.environ.get("AEGIS_LAT_OUT", "latency-attribution-raw.json")
CACHE_DIR = REPO_ROOT / "data" / "cache"
STAGES = ("cache_lookup", "routing", "retrieval", "confidence", "generation", "validation", "escalation")


def load_chunks() -> list[Chunk]:
    with open(settings.paths.data_processed / "chunks.jsonl", encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def load_sample() -> list[GoldenQuery]:
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        dev = [q for q in (GoldenQuery(**json.loads(line)) for line in f) if q.split == "dev"]
    rng = random.Random(settings.golden.random_seed)
    positives = [q for q in dev if not q.is_negative and q.ground_truth_chunk_ids]
    negatives = [q for q in dev if q.is_negative]
    sample = rng.sample(positives, N_POSITIVES) + rng.sample(negatives, N_NEGATIVES)
    rng.shuffle(sample)
    return sample


def load_checkpoint() -> dict[str, dict]:
    if not RAW_PATH.exists():
        return {}
    return {row["query_id"]: row for row in json.loads(RAW_PATH.read_text())}


def save_checkpoint(rows: dict[str, dict]) -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(json.dumps(list(rows.values()), indent=2))


def main() -> int:
    sample = load_sample()
    rows = load_checkpoint()
    todo = [q for q in sample if q.id not in rows or rows[q.id].get("error")]
    print(f"{len(sample)} sampled ({N_POSITIVES} pos + {N_NEGATIVES} neg), {len(rows) - sum(1 for r in rows.values() if r.get('error'))} already done, {len(todo)} to run")
    if not todo:
        print("nothing to run")
        return 0

    for name in ("phase13c_exact.sqlite", "phase13c_semantic.sqlite"):
        (CACHE_DIR / name).unlink(missing_ok=True)

    t0 = time.perf_counter()
    chunks = load_chunks()
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
    dense_index = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)
    reranker = CrossEncoderReranker()
    print(f"startup (chunk load + model construction): {time.perf_counter() - t0:.1f}s")

    pipeline = AnswerPipeline(
        exact_cache=ExactMatchCache(CACHE_DIR / "phase13c_exact.sqlite"),
        semantic_cache=SemanticCache(CACHE_DIR / "phase13c_semantic.sqlite", encoder=encoder, threshold=settings.cache.semantic_similarity_threshold),
        router=RouterClassifier(),
        retriever=dense_index,
        confidence_scorer=ConfidenceScorer(reranker=reranker, encode_fn=encoder.encode),
        generation=GenerationPipeline(llm=FallbackLLMClient([GroqProvider(), GeminiProvider()]), validator=CitationValidator()),
        tracer=get_tracer(),
        corpus_version=CORPUS_VERSION,
    )

    warm_up_ms = None
    if WARM_UP:
        t = time.perf_counter()
        warm_up_ms = pipeline.warm_up()
        print(f"warm_up: {warm_up_ms} total={(time.perf_counter() - t) * 1000:.0f} ms")

    for i, q in enumerate(todo, start=1):
        if i > 1 and PACE_SECONDS:
            time.sleep(PACE_SECONDS)
        wall_start = time.perf_counter()
        try:
            result = pipeline.answer(Query(id=q.id, text=q.text))
        except Exception as exc:  # per-query failure recorded, not fatal
            rows[q.id] = {"query_id": q.id, "is_negative": q.is_negative, "error": f"{type(exc).__name__}: {exc}"[:300]}
            print(f"[{i}/{len(todo)}] {q.id} ERROR {rows[q.id]['error']}")
            save_checkpoint(rows)
            continue
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        rows[q.id] = {
            "query_id": q.id,
            "is_negative": q.is_negative,
            "cold_start": i == 1,
            "pace_seconds": PACE_SECONDS,
            "warm_up_ms": warm_up_ms if i == 1 else None,
            "cache_status": result.cache_status.value,
            "escalate": result.escalate,
            "route": result.route.value if result.route else None,
            "provider": result.provider,
            "confidence": result.confidence.confidence if result.confidence else None,
            "stage_latencies_ms": result.stage_latencies_ms,
            "wall_ms": wall_ms,
            "trace_id": result.trace_id,
            "error": None,
        }
        save_checkpoint(rows)
        stages = ", ".join(f"{k}={v:.0f}" for k, v in result.stage_latencies_ms.items())
        print(f"[{i}/{len(todo)}] {q.id} escalate={result.escalate} provider={result.provider} wall={wall_ms:.0f}ms | {stages}")

    get_tracer().flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
