"""Phase 13 Unit 4 -- where do the ~2.5 s inside `ConfidenceScorer.score` go?

Unit 3 attributed 67% of the generate-path p50 to the `confidence` stage.
That stage is (a) one cross-encoder call over `rerank_pool_size` pairs plus
(b) an encode of every top-k chunk text for the semantic-agreement signal.
This times them separately, per call, on real queries. Zero LLM calls.
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
from aegis.rerank.cross_encoder import CrossEncoderReranker
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, GoldenQuery

N_QUERIES = 15


def ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def main() -> int:
    with open(settings.paths.data_processed / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f]
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        dev = [q for q in (GoldenQuery(**json.loads(l)) for l in f) if q.split == "dev" and not q.is_negative]
    queries = random.Random(settings.golden.random_seed).sample(dev, N_QUERIES)

    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
    dense = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)
    pool = settings.confidence.rerank_pool_size
    print(f"rerank device={settings.rerank.device}, pool={pool}, pairs per rerank call={pool}")

    reranker = CrossEncoderReranker()
    rows = []
    for i, q in enumerate(queries):
        cands = dense.search(q.text, top_k=10)

        t = time.perf_counter()
        reranker.rerank(q.text, cands[:pool], top_k=pool)
        rerank_ms = ms(t)

        texts = [c.text for c, _ in cands]
        t = time.perf_counter()
        encoder.encode(texts)
        encode_ms = ms(t)

        rows.append((rerank_ms, encode_ms, sum(len(c.text) for c, _ in cands[:pool])))
        print(f"q{i:02d} rerank={rerank_ms:8.0f} ms  encode_top10={encode_ms:7.0f} ms  chars_in_reranked_pool={rows[-1][2]}")

    warm = np.array(rows[1:])
    print(f"\nfirst call (cold): rerank={rows[0][0]:.0f} ms, encode={rows[0][1]:.0f} ms")
    print(f"warm (n={len(warm)}): rerank p50={np.percentile(warm[:,0],50):.0f} p95={np.percentile(warm[:,0],95):.0f} | "
          f"encode p50={np.percentile(warm[:,1],50):.0f} p95={np.percentile(warm[:,1],95):.0f}")
    print(f"corr(rerank_ms, pool_chars) = {np.corrcoef(warm[:,0], warm[:,2])[0,1]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
