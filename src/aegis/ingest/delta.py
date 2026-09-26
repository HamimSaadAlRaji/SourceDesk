"""Delta indexing (plan.md Phase 12, step 4): diff two chunk-corpus
snapshots by content hash so a reindex only re-embeds what changed and
purges chunks whose document was deleted or superseded -- the Air Canada
failure mode (a retired policy page still being retrievable) fixed as a
mechanism, with `tests/test_delta_indexing.py` proving it as a regression
test, not just described.

Also defines `corpus_version` for real: a hash over the exact set of
`(chunk_id, content_hash)` pairs currently indexed. `aegis.cache.
exact_match`/`aegis.cache.semantic` (Units 1-2) treated `corpus_version` as
an opaque caller-supplied string; `IndexManifest.corpus_version()` is where
it actually comes from -- change, add, or remove even one chunk and the
whole corpus gets a new version, orphaning every cache entry keyed to the
old one with no explicit purge step required.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel
from qdrant_client import QdrantClient

from aegis.embedding.encoder import Encoder
from aegis.embedding.vector_store import delete_chunks, ensure_collection, upsert_chunks
from aegis.types import Chunk


class IndexManifest(BaseModel):
    """chunk_id -> content_hash for every chunk currently indexed in one
    Qdrant collection. The source of truth for "what's actually in the
    index right now" -- not re-derived from Qdrant itself, since Qdrant's
    payload doesn't carry `content_hash` (same reasoning as `DenseIndex`
    keeping real `Chunk` objects in memory rather than round-tripping
    them through Qdrant's payload, `retrieval/dense.py`)."""

    entries: dict[str, str] = {}

    @classmethod
    def from_chunks(cls, chunks: list[Chunk]) -> IndexManifest:
        return cls(entries={chunk.id: chunk.content_hash for chunk in chunks})

    @classmethod
    def load(cls, path: Path) -> IndexManifest:
        if not path.exists():
            return cls()
        return cls.model_validate_json(path.read_text())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json())

    def corpus_version(self) -> str:
        canonical = json.dumps(sorted(self.entries.items()), separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DeltaPlan(BaseModel):
    added: list[str]
    changed: list[str]
    deleted: list[str]
    unchanged: list[str]

    @property
    def to_embed(self) -> list[str]:
        return self.added + self.changed


def diff(old: IndexManifest, new: IndexManifest) -> DeltaPlan:
    old_ids = set(old.entries)
    new_ids = set(new.entries)
    common = old_ids & new_ids

    return DeltaPlan(
        added=sorted(new_ids - old_ids),
        deleted=sorted(old_ids - new_ids),
        changed=sorted(cid for cid in common if old.entries[cid] != new.entries[cid]),
        unchanged=sorted(cid for cid in common if old.entries[cid] == new.entries[cid]),
    )


def apply_delta(
    client: QdrantClient,
    collection_name: str,
    plan: DeltaPlan,
    new_chunks_by_id: dict[str, Chunk],
    encoder: Encoder,
    dim: int,
) -> None:
    """Embeds only `plan.to_embed`, upserts those points by stable ID
    (`vector_store.chunk_point_id`), and deletes `plan.deleted`'s points --
    every other point in the collection is untouched. Never calls
    `create_collection`'s `recreate_collection` path; `ensure_collection`
    only creates the collection the first time it doesn't exist yet."""
    ensure_collection(client, collection_name, dim=dim)

    to_embed_ids = plan.to_embed
    if to_embed_ids:
        chunks_to_embed = [new_chunks_by_id[cid] for cid in to_embed_ids]
        vectors = encoder.encode([c.embedding_text for c in chunks_to_embed])
        upsert_chunks(client, collection_name, chunks_to_embed, vectors)

    delete_chunks(client, collection_name, plan.deleted)
