"""Phase 13 Unit 4 (continued) -- is the idle-gap slowness a GPU wake-up cost?

13h localized the confidence-stage gap entirely to the cross-encoder call and
showed it grows with idle time before the request. Hypothesis: after a long
idle gap the first GPU work is slow (power state), and a tiny throwaway rerank
issued at request arrival pays that cost instead of the real one.

Paired + interleaved design (fixes 13g's different-queries and order confounds):
every query runs twice, once WITH a 1-pair throwaway rerank just before
`answer()` and once WITHOUT, alternating A/B order, 70 s idle before every
run. Fixed router (see FixedRouter) + real Langfuse tracer, fake generation: zero Groq quota.
The throwaway's own cost is timed and reported, not hidden.
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
from aegis.router.classifier import Route, RouterClassifier
from aegis.types import Chunk, GoldenQuery, Query

# The one difference left between these fast runs (~0.5-0.7 s confidence) and the slow paced
# batches (~2.4 s for the SAME queries): the batches run the real router, real generation LLM
# and real citation validator in-process. Set AEGIS_I_REAL_STACK=1 (needs Groq/Gemini quota)
# to run that exact configuration through this same paired design.
REAL_STACK = os.environ.get("AEGIS_I_REAL_STACK", "0") == "1"
N_QUERIES = 6
GAP_S = 70
OUT = REPO_ROOT / "docs" / "results" / ("gpu-wake-paired-realstack-raw.json" if os.environ.get("AEGIS_I_REAL_STACK") == "1" else "gpu-wake-paired-raw.json")


class FixedRouter:
    """The router isn't the variable under test, and its real Groq model's daily
    token cap was exhausted by earlier Phase 13 runs (every real router call was
    failing into the billing fail-safe -- see design record 13d/13e). 13h already
    showed a real router present does not by itself reproduce the gap."""

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
    queries = random.Random(settings.golden.random_seed).sample(dev, N_QUERIES)

    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
    reranker = CrossEncoderReranker()
    scorer = ConfidenceScorer(reranker=reranker, encode_fn=encoder.encode)
    dense = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)
    if REAL_STACK:
        from aegis.generation.llm.fallback import FallbackLLMClient
        from aegis.generation.llm.gemini_provider import GeminiProvider
        from aegis.generation.llm.groq_provider import GroqProvider

        llm, validator, router = FallbackLLMClient([GroqProvider(), GeminiProvider()]), CitationValidator(), RouterClassifier()
    else:
        llm, validator, router = FakeLLM(), CitationValidator(entailment_fn=lambda claim, chunk_text: True), FixedRouter()
    dummy_pair = [(chunks[0], 1.0)]

    def build(tag: str) -> AnswerPipeline:
        for n in (f"phase13i_{tag}_exact.sqlite", f"phase13i_{tag}_sem.sqlite"):
            (REPO_ROOT / "data" / "cache" / n).unlink(missing_ok=True)
        return AnswerPipeline(
            exact_cache=ExactMatchCache(REPO_ROOT / "data" / "cache" / f"phase13i_{tag}_exact.sqlite"),
            semantic_cache=SemanticCache(REPO_ROOT / "data" / "cache" / f"phase13i_{tag}_sem.sqlite", encoder=encoder, threshold=settings.cache.semantic_similarity_threshold),
            router=router, retriever=dense, confidence_scorer=scorer,
            generation=GenerationPipeline(llm=llm, validator=validator),
            tracer=get_tracer(), corpus_version="phase13i",
        )

    print(f"warm_up: {build('warm').warm_up()}", flush=True)

    # Each query twice, arms alternating who goes first: ABBA ABBA ...
    schedule = []
    for i, q in enumerate(queries):
        order = ("wake", "plain") if i % 2 == 0 else ("plain", "wake")
        schedule += [(q, arm) for arm in order]

    rows = []
    for k, (q, arm) in enumerate(schedule):
        if k:
            time.sleep(GAP_S)
        pipe = build(f"{arm}{k}")
        wake_ms = None
        if arm == "wake":
            t = time.perf_counter()
            reranker.rerank("wake", dummy_pair, top_k=1)
            wake_ms = (time.perf_counter() - t) * 1000.0
        r = pipe.answer(Query(id=q.id, text=q.text))
        row = {"k": k, "arm": arm, "query_id": q.id, "route": r.route.value if r.route else None, "provider": r.provider,
               "confidence_stage_ms": r.stage_latencies_ms.get("confidence"), "wake_ms": wake_ms,
               "routing_ms": r.stage_latencies_ms.get("routing")}
        rows.append(row)
        if REAL_STACK:
            # Contaminated-run guard. These are docs questions: a billing route means the router's API call failed
            # into its fail-safe (Groq quota); a non-groq provider means generation fell back to Gemini (quota again,
            # and it would burn Gemini's daily cap). Either way the run no longer measures what it claims to.
            if row["route"] == "billing_account_legal" or (r.provider not in (None, "groq") and not r.escalate):
                OUT.write_text(json.dumps(rows, indent=2))
                print(f"ABORT at run {k}: route={row['route']} provider={r.provider} -- quota/provider failure, data not trustworthy", flush=True)
                return 2
        print(f"{k:2d} {arm:5s} {q.id[-12:]:>12s} route={row['route']} confidence={row['confidence_stage_ms']} wake={wake_ms}", flush=True)
    OUT.write_text(json.dumps(rows, indent=2))

    def col(arm):
        return [r["confidence_stage_ms"] for r in rows if r["arm"] == arm and r["confidence_stage_ms"] is not None]
    for arm in ("plain", "wake"):
        v = col(arm)
        print(f"{arm:5s} n={len(v)} confidence p50={np.percentile(v,50):.0f} mean={np.mean(v):.0f} max={max(v):.0f} all={[round(x) for x in v]}")
    w = [r["wake_ms"] for r in rows if r["wake_ms"] is not None]
    print(f"wake throwaway cost: p50={np.percentile(w,50):.0f} max={max(w):.0f} ms")
    get_tracer().flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
