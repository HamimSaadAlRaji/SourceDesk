"""Qdrant collection management: one collection per encoder candidate, a
single named `dense` vector per point (a `sparse` named vector joins it in
Phase 5 — plan.md's hybrid-search schema), and payload indexes on the three
fields the router/retrieval-filtering layers need (`product_area`,
`doc_type`, `doc_version`).

I/O-bound wrapper over the Qdrant client — integration-tested against the
real local Qdrant instance (plan.md Phase 0 already established
`QdrantClient(...).get_collections()` as reachable), not unit-tested with a
mock client per CLAUDE.md's I/O-component guidance.
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from aegis.config import settings
from aegis.types import Chunk

DENSE_VECTOR_NAME = "dense"
_PAYLOAD_INDEX_FIELDS = {
    "product_area": PayloadSchemaType.KEYWORD,
    "doc_type": PayloadSchemaType.KEYWORD,
    "doc_version": PayloadSchemaType.KEYWORD,
}

# Qdrant point IDs must be an unsigned int or a UUID -- verified directly
# (an arbitrary string like a real chunk_id, e.g.
# "doc::release-notes.md::heading_aware::961", is rejected server-side).
# uuid5 gives a deterministic point ID from that same real chunk_id, stable
# across runs -- required for delta indexing's upsert/delete-by-id to ever
# target the right point twice (plan.md Phase 12 step 4). Previously
# `upsert_chunks` used the chunk's position in the passed-in list as the
# point ID, which only ever worked because `create_collection` fully wipes
# and rebuilds every time it's called -- silently unsafe for any future
# incremental upsert, since one deletion anywhere in the list shifts every
# later chunk's position, and therefore its point ID.
_POINT_ID_NAMESPACE = uuid.UUID("d9f4a9d0-6e0c-4d3b-8e29-9d4a6e7b1f00")


def chunk_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


def get_client(read_only: bool = False) -> QdrantClient:
    """`read_only=True` for the served pipeline (design record 15f): Qdrant's
    read-only key can search but not write or delete. If an admin key is set
    but no read-only key, this refuses rather than serving with write rights.
    No keys configured at all = a local Qdrant without auth (no key sent)."""
    q = settings.qdrant
    if read_only:
        if q.api_key and not q.read_only_api_key:
            raise RuntimeError("QDRANT_API_KEY is set but no read-only key (QDRANT_READ_ONLY_API_KEY); refusing to serve "
                               "with the admin key")
        key = q.read_only_api_key or None
    else:
        key = q.api_key or None
    # Plain HTTP is acceptable only because the port is bound to 127.0.0.1
    # (docker-compose.yml); a remote Qdrant needs https=True.
    return QdrantClient(host=q.host, port=q.port, api_key=key, https=False)


def create_collection(
    client: QdrantClient,
    collection_name: str,
    dim: int,
    hnsw_m: int | None = None,
    hnsw_ef_construct: int | None = None,
) -> None:
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=dim,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(
                    m=hnsw_m if hnsw_m is not None else settings.qdrant.hnsw_m,
                    ef_construct=hnsw_ef_construct if hnsw_ef_construct is not None else settings.qdrant.hnsw_ef_construct,
                ),
            )
        },
    )
    for field_name, schema in _PAYLOAD_INDEX_FIELDS.items():
        client.create_payload_index(collection_name=collection_name, field_name=field_name, field_schema=schema)


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    dim: int,
    hnsw_m: int | None = None,
    hnsw_ef_construct: int | None = None,
) -> None:
    """Like `create_collection`, but a no-op if the collection already
    exists -- non-destructive, for the delta path (Phase 12 Unit 4), which
    must never wipe the points it isn't touching. `create_collection`
    itself stays `recreate_collection`-based, unchanged, for its one real
    caller (`embedding/run.py`'s full-corpus rebuild)."""
    if client.collection_exists(collection_name):
        return
    create_collection(client, collection_name, dim, hnsw_m, hnsw_ef_construct)


def upsert_chunks(client: QdrantClient, collection_name: str, chunks: list[Chunk], vectors, batch_size: int = 256) -> None:
    points = [
        PointStruct(
            id=chunk_point_id(chunk.id),
            vector={DENSE_VECTOR_NAME: vectors[i].tolist()},
            payload={
                "chunk_id": chunk.id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "heading_path": chunk.heading_path,
                "doc_version": chunk.doc_version,
                "doc_type": chunk.doc_type.value,
                "product_area": chunk.product_area,
                "token_count": chunk.token_count,
            },
        )
        for i, chunk in enumerate(chunks)
    ]
    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[start : start + batch_size])


def delete_chunks(client: QdrantClient, collection_name: str, chunk_ids: list[str]) -> None:
    if not chunk_ids:
        return
    client.delete(
        collection_name=collection_name,
        points_selector=[chunk_point_id(cid) for cid in chunk_ids],
    )
