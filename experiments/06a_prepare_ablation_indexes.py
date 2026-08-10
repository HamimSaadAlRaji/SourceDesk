"""Phase 6 support script: build the chunk sets and Qdrant collections the
ablation grid needs beyond what Phase 3 already indexed (heading-aware,
enrichment off, all three encoders).

Confirmed center after the retrieval-strategy and encoder axes (measured
in-conversation, not yet in docs/results/ until the full grid is written
up): `dense` retrieval, `bge_large` encoder. This script re-chunks and
re-embeds with *that* encoder only for each new combination — not the
cartesian product (plan.md Phase 6 step 3) — using the same length-sorted-
batching + `torch.mps.empty_cache()` pattern `embedding/run.py` already
validated against real M1 memory pressure (Phase 3).

Usage: `uv run python experiments/06a_prepare_ablation_indexes.py <key>`
where `<key>` is one of `enriched`, `fixed`, `recursive`, `semantic`.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from aegis.chunking.factory import build_chunker
from aegis.config import settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.embedding.vector_store import create_collection, get_client, upsert_chunks
from aegis.types import Chunk, Document

_ENCODE_BATCH_SIZE = 8  # matches embedding/run.py's bge_large setting

_COMBOS = {
    "enriched": {"strategy": "heading_aware", "enrichment": True, "collection": "aegis_ablation_heading_aware_enriched"},
    "fixed": {"strategy": "fixed", "enrichment": False, "collection": "aegis_ablation_fixed"},
    "recursive": {"strategy": "recursive", "enrichment": False, "collection": "aegis_ablation_recursive"},
    "semantic": {"strategy": "semantic", "enrichment": False, "collection": "aegis_ablation_semantic"},
}


def load_documents(path) -> list[Document]:
    with open(path, encoding="utf-8") as f:
        return [Document(**json.loads(line)) for line in f]


def chunk_corpus(documents: list[Document], strategy: str, enrichment: bool) -> list[Chunk]:
    chunker = build_chunker(strategy, contextual_enrichment=enrichment)
    chunks: list[Chunk] = []
    t0 = time.time()
    for i, doc in enumerate(documents):
        chunks.extend(chunker.chunk(doc))
        if (i + 1) % 500 == 0:
            print(f"  chunked {i + 1}/{len(documents)} docs, {len(chunks)} chunks so far ({time.time() - t0:.1f}s)")
    print(f"Chunked {len(documents)} docs -> {len(chunks)} chunks in {time.time() - t0:.1f}s ({strategy}, enrichment={enrichment})")
    return chunks


def embed_and_index(chunks: list[Chunk], collection_name: str) -> None:
    cache = EmbeddingCache(settings.paths.embedding_cache)
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=cache, batch_size=_ENCODE_BATCH_SIZE)

    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i].embedding_text))
    texts_sorted = [chunks[i].embedding_text for i in order]

    try:
        import torch
    except ImportError:
        torch = None

    all_vectors = []
    t0 = time.time()
    outer_batch_size = 128
    for start in range(0, len(texts_sorted), outer_batch_size):
        batch = texts_sorted[start : start + outer_batch_size]
        vecs = encoder.encode(batch)
        all_vectors.append(vecs)
        if torch is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        done = start + len(batch)
        elapsed = time.time() - t0
        print(f"  embedded {done}/{len(texts_sorted)} ({done / elapsed:.1f} chunks/sec)")
    sorted_vectors = np.concatenate(all_vectors, axis=0)

    vectors = np.zeros_like(sorted_vectors)
    vectors[order] = sorted_vectors
    print(f"Embedding done in {time.time() - t0:.1f}s, shape={vectors.shape}")

    client = get_client()
    create_collection(client, collection_name, dim=vectors.shape[1])
    upsert_chunks(client, collection_name, chunks, vectors)
    print(f"Indexed {len(chunks)} points into Qdrant collection {collection_name!r}")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in _COMBOS:
        print(f"Usage: python experiments/06a_prepare_ablation_indexes.py <{'|'.join(_COMBOS)}>")
        sys.exit(1)

    key = sys.argv[1]
    combo = _COMBOS[key]

    documents_path = settings.paths.data_processed / "documents.jsonl"
    documents = load_documents(documents_path)
    print(f"Loaded {len(documents)} documents from {documents_path}")

    chunks = chunk_corpus(documents, combo["strategy"], combo["enrichment"])

    chunks_path = settings.paths.data_processed / f"chunks_ablation_{key}.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")
    print(f"Wrote {chunks_path}")

    embed_and_index(chunks, combo["collection"])


if __name__ == "__main__":
    main()
