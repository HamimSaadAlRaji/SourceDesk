"""Delta indexing (plan.md Phase 12, step 4).

`IndexManifest`/`diff`/`corpus_version` are pure, deterministic logic --
a silent bug here means a deleted or superseded document's chunks quietly
stay indexed (or a live document's chunks get purged), the exact failure
mode this unit exists to prevent. Covered here with fakes, no I/O.

`apply_delta` is I/O-bound (a real Qdrant collection), covered with one
real integration test against the local Qdrant instance, proving the
literal plan.md exit criterion: a deleted document's chunks become
unretrievable. Uses a throwaway collection (`_test_delta_indexing`), never
one of the real `aegis_chunks_*` collections other phases' results depend
on, and drops it in a fixture teardown regardless of outcome.
"""

from __future__ import annotations

import numpy as np
import pytest
from qdrant_client import QdrantClient

from aegis.embedding.vector_store import get_client
from aegis.ingest.delta import DeltaPlan, IndexManifest, apply_delta, diff
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, DocType

# --- IndexManifest ---


def _chunk(chunk_id: str, doc_id: str, content_hash: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        text=f"text for {chunk_id}",
        doc_type=DocType.DOCUMENTATION,
        content_hash=content_hash,
        token_count=3,
        embedding_text=f"text for {chunk_id}",
    )


def test_from_chunks_maps_id_to_content_hash():
    chunks = [_chunk("a::1", "a", "hash1"), _chunk("a::2", "a", "hash2")]
    manifest = IndexManifest.from_chunks(chunks)
    assert manifest.entries == {"a::1": "hash1", "a::2": "hash2"}


def test_load_missing_file_returns_empty_manifest(tmp_path):
    manifest = IndexManifest.load(tmp_path / "does_not_exist.json")
    assert manifest.entries == {}


def test_save_then_load_roundtrips(tmp_path):
    path = tmp_path / "manifest.json"
    original = IndexManifest(entries={"a::1": "hash1"})
    original.save(path)
    assert IndexManifest.load(path) == original


# --- corpus_version ---


def test_corpus_version_is_deterministic_regardless_of_dict_order():
    m1 = IndexManifest(entries={"a": "1", "b": "2"})
    m2 = IndexManifest(entries={"b": "2", "a": "1"})
    assert m1.corpus_version() == m2.corpus_version()


def test_corpus_version_changes_when_a_hash_changes():
    m1 = IndexManifest(entries={"a": "1"})
    m2 = IndexManifest(entries={"a": "2"})
    assert m1.corpus_version() != m2.corpus_version()


def test_corpus_version_changes_when_a_chunk_is_added():
    m1 = IndexManifest(entries={"a": "1"})
    m2 = IndexManifest(entries={"a": "1", "b": "2"})
    assert m1.corpus_version() != m2.corpus_version()


def test_corpus_version_changes_when_a_chunk_is_deleted():
    m1 = IndexManifest(entries={"a": "1", "b": "2"})
    m2 = IndexManifest(entries={"a": "1"})
    assert m1.corpus_version() != m2.corpus_version()


def test_corpus_version_same_content_same_version():
    m1 = IndexManifest(entries={"a": "1", "b": "2"})
    m2 = IndexManifest(entries={"a": "1", "b": "2"})
    assert m1.corpus_version() == m2.corpus_version()


# --- diff ---


def test_diff_detects_added():
    old = IndexManifest(entries={"a": "1"})
    new = IndexManifest(entries={"a": "1", "b": "2"})
    plan = diff(old, new)
    assert plan == DeltaPlan(added=["b"], changed=[], deleted=[], unchanged=["a"])


def test_diff_detects_deleted():
    old = IndexManifest(entries={"a": "1", "b": "2"})
    new = IndexManifest(entries={"a": "1"})
    plan = diff(old, new)
    assert plan == DeltaPlan(added=[], changed=[], deleted=["b"], unchanged=["a"])


def test_diff_detects_changed():
    old = IndexManifest(entries={"a": "1"})
    new = IndexManifest(entries={"a": "1-edited"})
    plan = diff(old, new)
    assert plan == DeltaPlan(added=[], changed=["a"], deleted=[], unchanged=[])


def test_diff_detects_unchanged():
    old = IndexManifest(entries={"a": "1"})
    new = IndexManifest(entries={"a": "1"})
    plan = diff(old, new)
    assert plan == DeltaPlan(added=[], changed=[], deleted=[], unchanged=["a"])


def test_diff_combines_all_four_categories():
    old = IndexManifest(entries={"kept": "1", "edited": "old", "removed": "1"})
    new = IndexManifest(entries={"kept": "1", "edited": "new", "new_one": "1"})
    plan = diff(old, new)
    assert plan.added == ["new_one"]
    assert plan.changed == ["edited"]
    assert plan.deleted == ["removed"]
    assert plan.unchanged == ["kept"]


def test_to_embed_is_added_plus_changed():
    plan = DeltaPlan(added=["a"], changed=["b"], deleted=["c"], unchanged=["d"])
    assert plan.to_embed == ["a", "b"]


# --- apply_delta: the real regression test (Air Canada failure mode) ---

_TEST_COLLECTION = "_test_delta_indexing"
_DIM = 4


class _FakeEncoder:
    """Fixed, known vectors -- doc A's chunks and doc B's chunks live in
    clearly separated regions of a 4D space, so a real Qdrant cosine search
    deterministically returns the intended cluster."""

    _VECTORS = {
        "text for doc-a::1": [1.0, 0.0, 0.0, 0.0],
        "text for doc-a::2": [0.9, 0.1, 0.0, 0.0],
        "text for doc-b::1": [0.0, 1.0, 0.0, 0.0],
        "text for doc-b::2": [0.0, 0.9, 0.1, 0.0],
        "query near doc a": [1.0, 0.0, 0.0, 0.0],
        "query near doc b": [0.0, 1.0, 0.0, 0.0],
    }

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([self._VECTORS[t] for t in texts], dtype=np.float32)


@pytest.fixture
def qdrant_client() -> QdrantClient:
    client = get_client()
    client.delete_collection(_TEST_COLLECTION)  # in case a previous run crashed before teardown
    yield client
    client.delete_collection(_TEST_COLLECTION)


def test_deleted_document_becomes_unretrievable(qdrant_client):
    doc_a_chunks = [_chunk("doc-a::1", "doc-a", "hash-a1"), _chunk("doc-a::2", "doc-a", "hash-a2")]
    doc_b_chunks = [_chunk("doc-b::1", "doc-b", "hash-b1"), _chunk("doc-b::2", "doc-b", "hash-b2")]
    all_chunks = doc_a_chunks + doc_b_chunks
    encoder = _FakeEncoder()

    # Initial index: everything is "added" against an empty old manifest.
    old_manifest = IndexManifest()
    full_manifest = IndexManifest.from_chunks(all_chunks)
    initial_plan = diff(old_manifest, full_manifest)
    apply_delta(
        qdrant_client, _TEST_COLLECTION, initial_plan,
        {c.id: c for c in all_chunks}, encoder, dim=_DIM,
    )

    assert qdrant_client.count(_TEST_COLLECTION).count == 4

    dense_index_before = DenseIndex(all_chunks, _TEST_COLLECTION, encoder, client=qdrant_client)
    results_before = dense_index_before.search("query near doc a", top_k=4)
    assert {c.id for c, _ in results_before if c.doc_id == "doc-a"} == {"doc-a::1", "doc-a::2"}

    # doc-a is deleted/superseded: the new real corpus no longer contains it.
    new_manifest = IndexManifest.from_chunks(doc_b_chunks)
    deletion_plan = diff(full_manifest, new_manifest)
    assert deletion_plan.deleted == ["doc-a::1", "doc-a::2"]

    apply_delta(
        qdrant_client, _TEST_COLLECTION, deletion_plan,
        {c.id: c for c in doc_b_chunks}, encoder, dim=_DIM,
    )

    # Real check at the Qdrant layer itself, not just the wrapper: exactly
    # doc-b's 2 points remain.
    assert qdrant_client.count(_TEST_COLLECTION).count == 2

    # Real check through the real serving path: a query that used to surface
    # doc-a's chunks now can't, because a real caller rebuilding DenseIndex
    # only ever has the real, current corpus (doc-b only) in memory.
    dense_index_after = DenseIndex(doc_b_chunks, _TEST_COLLECTION, encoder, client=qdrant_client)
    results_after = dense_index_after.search("query near doc a", top_k=4)
    assert all(c.doc_id != "doc-a" for c, _ in results_after)

    # doc-b itself is untouched by the delta.
    results_b = dense_index_after.search("query near doc b", top_k=4)
    assert {c.id for c, _ in results_b} == {"doc-b::1", "doc-b::2"}
