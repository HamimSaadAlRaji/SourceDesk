"""Rebuild the served Qdrant collection from its two source files, so CI can
run the retrieval eval without re-embedding the corpus (design record 16b).

`chunks.jsonl` (ingest) and `vectors_bge_large.npy` (the embedding run) are
produced separately and nothing in either records that row i of one is chunk i
of the other. `check_alignment` refuses the mismatches that would otherwise
attach vectors to the wrong text and corrupt every downstream metric without
raising. The upsert itself is the real `upsert_chunks`, not a CI copy of it, so
the payload and point IDs are exactly what the served index has.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import CollectionStatus

from aegis.embedding.vector_store import create_collection, upsert_chunks
from aegis.types import Chunk

# bge-large-en-v1.5 emits unit vectors (the Encoder passes normalize_embeddings=True);
# 1e-3 absorbs float32 rounding without admitting a 2x-scaled or unnormalized matrix.
_UNIT_NORM_TOLERANCE = 1e-3


class IndexAlignmentError(ValueError):
    """The chunk file and the vector file do not describe the same corpus."""


def check_alignment(chunks: Sequence[Chunk], vectors: np.ndarray, expected_dim: int) -> None:
    if vectors.ndim != 2 or vectors.shape[1] != expected_dim:
        raise IndexAlignmentError(f"vector dim is {vectors.shape[1:]} but expected dim {expected_dim}")
    if len(chunks) != vectors.shape[0]:
        raise IndexAlignmentError(f"{len(chunks)} chunks but {vectors.shape[0]} vectors")
    ids = [c.id for c in chunks]
    if len(set(ids)) != len(ids):
        raise IndexAlignmentError("duplicate chunk ids (point IDs derive from the id, so one would overwrite another)")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=_UNIT_NORM_TOLERANCE):
        raise IndexAlignmentError(f"vectors are not unit length (norm range {norms.min():.4f}-{norms.max():.4f})")


def corpus_fingerprint(chunks: Sequence[Chunk], vectors: np.ndarray) -> str:
    """Order-sensitive hash of (chunk id, embedded text) pairs plus the raw vector
    bytes. Hashes the text itself rather than the stored `content_hash` field,
    which a hand edit of chunks.jsonl would not update. Printed with every CI
    eval result, so a metric movement can be told apart from "someone swapped the
    restored files"."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(f"{c.id}\x00{c.embedding_text}\x00".encode())
    h.update(np.ascontiguousarray(vectors, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def restore_index(
    client: QdrantClient,
    collection_name: str,
    chunks: Sequence[Chunk],
    vectors: np.ndarray,
    *,
    green_timeout_s: float = 600.0,
) -> dict[str, float]:
    """Create `collection_name` (wiping any previous one -- it is a throwaway in
    CI) and fill it. Returns seconds spent upserting and waiting for the HNSW
    build: Qdrant answers queries before indexing finishes, but from an
    unindexed segment (exact scan), which would make the first metrics measure a
    different retriever than the served one."""
    check_alignment(chunks, vectors, expected_dim=vectors.shape[1])
    t0 = time.perf_counter()
    create_collection(client, collection_name, dim=vectors.shape[1])
    upsert_chunks(client, collection_name, list(chunks), vectors)
    upsert_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    while time.perf_counter() - t1 < green_timeout_s:
        info = client.get_collection(collection_name)
        if info.status == CollectionStatus.GREEN and (info.points_count or 0) == len(chunks):
            return {"upsert_s": upsert_s, "index_wait_s": time.perf_counter() - t1}
        time.sleep(1.0)
    raise TimeoutError(f"{collection_name} not green after {green_timeout_s:.0f}s")
