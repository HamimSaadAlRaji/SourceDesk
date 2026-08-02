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


def get_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant.host, port=settings.qdrant.port)


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


def upsert_chunks(client: QdrantClient, collection_name: str, chunks: list[Chunk], vectors, batch_size: int = 256) -> None:
    points = [
        PointStruct(
            id=i,
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
