"""Exports a self-contained candidate-pool file for the Phase 7 pool-size
sweep, so the GPU-bound reranking step (`experiments/07b_kaggle_rerank_sweep.py`)
can run on a free Kaggle GPU instead of this machine's memory-constrained M1
Pro (see docs/explainers/07-reranking-and-encoder-architectures.md §6 --
MPS hit macOS's "stuck" scheduler state twice under this session's system-
wide memory pressure, and real corpus-length CPU inference measured at
~780ms/pair, projecting a ~5.7 hour sweep -- neither usable in-session).

Dense retrieval itself is cheap and already fully local (~15ms/query,
Qdrant) -- only the cross-encoder step needs to move. This script does the
dense retrieval here (same Phase 6 winning config: heading-aware chunking,
bge-large-en-v1.5, no contextual enrichment) and writes every golden
query's top-100 candidate pool, with real chunk text, ground truth, and the
bi-encoder score, so nothing on Kaggle needs Qdrant or the local corpus.
"""

from __future__ import annotations

import json

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, GoldenQuery

EXPORT_PATH = REPO_ROOT / "data" / "processed" / "rerank_candidates_export.json"


def load_chunks() -> list[Chunk]:
    path = settings.paths.data_processed / "chunks.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def load_positive_queries() -> list[GoldenQuery]:
    path = settings.paths.data_golden / "golden_set_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        queries = [GoldenQuery(**json.loads(line)) for line in f]
    return [q for q in queries if not q.is_negative]


def main() -> None:
    chunks = load_chunks()
    queries = load_positive_queries()
    print(f"{len(chunks)} chunks, {len(queries)} positive golden queries")

    cache = EmbeddingCache(settings.paths.embedding_cache)
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=cache)
    dense_index = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)

    max_pool = max(settings.rerank.pool_sizes)
    print(f"Retrieving top-{max_pool} dense candidates per query...")

    export_queries = []
    for q in queries:
        hits = dense_index.search(q.text, top_k=max_pool, exclude_doc_id=q.source_issue)
        export_queries.append(
            {
                "id": q.id,
                "text": q.text,
                "ground_truth_chunk_ids": q.ground_truth_chunk_ids,
                "candidates": [{"chunk_id": c.id, "text": c.text, "bi_encoder_score": score} for c, score in hits],
            }
        )

    payload = {
        "pool_sizes": sorted(settings.rerank.pool_sizes),
        "final_top_k": settings.rerank.final_top_k,
        "reranker_model": settings.rerank.model_name,
        "queries": export_queries,
    }
    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPORT_PATH.write_text(json.dumps(payload))
    size_mb = EXPORT_PATH.stat().st_size / 1e6
    print(f"Wrote {EXPORT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
