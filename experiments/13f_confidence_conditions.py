"""Phase 13 Unit 4 -- why does the real `ConfidenceScorer.score` cost ~2,500 ms
in the paced pipeline baseline but ~570 ms when its parts are timed
back-to-back (13e)? Two candidate explanations, one experiment each:

  A. back-to-back: real scorer, queries one after another (hot GPU)
  B. idle-gapped:  same scorer, 70 s idle before each call (mirrors the paced
                   pipeline baseline, where the reranker sat idle between queries)

If B >> A, the baseline number was inflated by idle-induced slowness, not by
the confidence stage's own work. Zero LLM calls.
"""

from __future__ import annotations

import json
import random
import sys
import time

import numpy as np

from aegis.config import settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.guardrails.confidence import ConfidenceScorer
from aegis.rerank.cross_encoder import CrossEncoderReranker
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, GoldenQuery

N_BACK_TO_BACK = 15
N_IDLE = 5
IDLE_SECONDS = 70


def main() -> int:
    with open(settings.paths.data_processed / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f]
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        dev = [q for q in (GoldenQuery(**json.loads(l)) for l in f) if q.split == "dev" and not q.is_negative]
    queries = random.Random(settings.golden.random_seed).sample(dev, N_BACK_TO_BACK + N_IDLE + 1)

    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
    dense = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)
    scorer = ConfidenceScorer(reranker=CrossEncoderReranker(), encode_fn=encoder.encode)

    def timed(q) -> float:
        cands = dense.search(q.text, top_k=10)
        t = time.perf_counter()
        scorer.score(q.text, cands)
        return (time.perf_counter() - t) * 1000.0

    print(f"cold (excluded): {timed(queries[0]):.0f} ms")
    a = [timed(q) for q in queries[1 : 1 + N_BACK_TO_BACK]]
    print(f"A back-to-back n={len(a)}: p50={np.percentile(a,50):.0f} p95={np.percentile(a,95):.0f} ms  all={[round(x) for x in a]}")

    b = []
    for q in queries[1 + N_BACK_TO_BACK :]:
        time.sleep(IDLE_SECONDS)
        b.append(timed(q))
    print(f"B idle-gapped ({IDLE_SECONDS}s) n={len(b)}: p50={np.percentile(b,50):.0f} p95={np.percentile(b,95):.0f} ms  all={[round(x) for x in b]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
