"""Embed the persisted chunk corpus with a given model and index it into
Qdrant. Usage: `uv run python -m aegis.embedding.run <model-key>` where
`<model-key>` is one of `bge_small`, `bge_large`, `bge_m3`.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from aegis.config import settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.embedding.vector_store import create_collection, get_client, upsert_chunks
from aegis.types import Chunk

# (model name, collection name, sentence-transformers internal batch_size).
# bge-small was run first at batch_size=32 and hit real, measured M1 16GB
# memory pressure (`vm.swapusage` showed ~3.5GB swapped mid-run) — a smaller
# batch size for the two much larger candidates is the mitigation plan.md's
# own risk table already calls for ("smaller batches ... bge-large only for
# final runs"), applied here after being forced to actually hit the problem.
_MODEL_KEYS = {
    "bge_small": (settings.embedding.dev_model, settings.qdrant.collection_bge_small, 32),
    "bge_large": (settings.embedding.bge_large_model, settings.qdrant.collection_bge_large, 8),
    "bge_m3": (settings.embedding.bge_m3_model, settings.qdrant.collection_bge_m3, 8),
}


def load_chunks(path) -> list[Chunk]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk(**json.loads(line)))
    return chunks


def embed_and_index(model_key: str, outer_batch_size: int = 128) -> None:
    if model_key not in _MODEL_KEYS:
        raise ValueError(f"Unknown model key {model_key!r}; expected one of {sorted(_MODEL_KEYS)}")
    model_name, collection_name, encode_batch_size = _MODEL_KEYS[model_key]

    chunks_path = settings.paths.data_processed / "chunks.jsonl"
    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    cache = EmbeddingCache(settings.paths.embedding_cache)
    encoder = Encoder(model_name=model_name, cache=cache, batch_size=encode_batch_size)

    # Sort by text length before batching so each batch pads to a similar
    # length instead of the longest item in an arbitrary (document-order)
    # window — found necessary after the bge-small run: a batch spanning
    # issue::2812-2936 included a 2,830-token atomic-code-block chunk next to
    # ~135-token ones, forcing the whole batch to pad to the outlier's length
    # and dropping that batch's throughput roughly 25x below the run's
    # average. Original order is restored before saving/indexing.
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i].embedding_text))
    texts_sorted = [chunks[i].embedding_text for i in order]

    all_vectors = []
    t0 = time.time()
    try:
        import torch
    except ImportError:
        torch = None

    for start in range(0, len(texts_sorted), outer_batch_size):
        batch = texts_sorted[start : start + outer_batch_size]
        vecs = encoder.encode(batch)
        all_vectors.append(vecs)
        if torch is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elapsed = time.time() - t0
        done = start + len(batch)
        print(f"  embedded {done}/{len(texts_sorted)} ({done / elapsed:.1f} chunks/sec)")
    sorted_vectors = np.concatenate(all_vectors, axis=0)

    # Undo the length sort to restore document order.
    vectors = np.zeros_like(sorted_vectors)
    vectors[order] = sorted_vectors
    print(f"Embedding done in {time.time() - t0:.1f}s, shape={vectors.shape}")

    client = get_client()
    create_collection(client, collection_name, dim=vectors.shape[1])
    upsert_chunks(client, collection_name, chunks, vectors)
    print(f"Indexed {len(chunks)} points into Qdrant collection {collection_name!r}")

    np.save(settings.paths.data_processed / f"vectors_{model_key}.npy", vectors)
    print(f"Saved raw vectors to vectors_{model_key}.npy (for the brute-force KNN ground truth)")


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python -m aegis.embedding.run <{'|'.join(_MODEL_KEYS)}>")
        sys.exit(1)
    embed_and_index(sys.argv[1])


if __name__ == "__main__":
    main()
