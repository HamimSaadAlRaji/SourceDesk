"""Phase 13 Unit 4 (continued) -- is the confidence stage's variance a
first-time-per-shape cost (MPS kernel compile for a new padded sequence
length)?

Evidence so far: the same query costs ~0.6 s in one script/process and ~2.5 s in
another (e.g. issue 753), the extra time is entirely inside the cross-encoder
call, and nothing about load, tracer, router, idle gaps, real generation, or
power state explains it. If it IS first-time-per-shape, then in ONE process the
first scoring of a query's candidates should be slow and an immediate repeat
(identical tensors) fast. Zero LLM calls.
"""

from __future__ import annotations

import json
import sys
import time

from aegis.config import settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.guardrails.confidence import ConfidenceScorer
from aegis.rerank.cross_encoder import CrossEncoderReranker
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, GoldenQuery

IDS = ["golden::issue::2689", "golden::issue::2884", "golden::issue::3212", "golden::issue::753", "golden::issue::219", "golden::issue::11104"]
REPEATS = 3


def main() -> int:
    with open(settings.paths.data_processed / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f]
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        golden = {q.id: q for q in (GoldenQuery(**json.loads(l)) for l in f)}

    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
    dense = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)
    reranker = CrossEncoderReranker()
    scorer = ConfidenceScorer(reranker=reranker, encode_fn=encoder.encode)

    # same warm-up the pipeline does
    t = time.perf_counter()
    scorer.score("how do I declare a query parameter", dense.search("how do I declare a query parameter", top_k=10))
    print(f"warm_up: {(time.perf_counter() - t) * 1000:.0f} ms\n", flush=True)

    tok = None
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
        from aegis.rerank.cross_encoder import _MODEL_CACHE
        tok = next(iter(_MODEL_CACHE.values())).tokenizer
    except Exception:
        pass

    for qid in IDS:
        q = golden[qid]
        cands = dense.search(q.text, top_k=10)
        pool = cands[: settings.confidence.rerank_pool_size]
        padded = None
        if tok is not None:
            enc = tok([q.text] * len(pool), [c.text for c, _ in pool], truncation=True, max_length=512, padding=True, return_tensors="pt")
            padded = tuple(enc["input_ids"].shape)
        times = []
        for _ in range(REPEATS):
            t = time.perf_counter()
            scorer.score(q.text, cands)
            times.append((time.perf_counter() - t) * 1000.0)
        print(f"{qid[-12:]:>12} padded_shape={padded}  call1={times[0]:6.0f}  call2={times[1]:6.0f}  call3={times[2]:6.0f} ms", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
