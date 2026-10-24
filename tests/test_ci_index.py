"""The CI index restore (design record 16b). The restore rebuilds Qdrant from two
files that were produced separately (chunks.jsonl by ingest, the .npy by the
embedding run). Row i of one must be chunk i of the other; nothing in either file
says so, so a mismatch would silently attach every vector to the wrong text and
corrupt every retrieval metric downstream. These pin the checks that refuse it."""

from __future__ import annotations

import numpy as np
import pytest

from aegis.eval.ci_index import IndexAlignmentError, check_alignment, corpus_fingerprint
from aegis.types import Chunk, DocType


def _chunk(i: int, text: str = "t") -> Chunk:
    return Chunk(
        id=f"doc::{i}", doc_id=f"doc::{i // 2}", text=text, heading_path=[], doc_version="1",
        doc_type=DocType.DOCUMENTATION, product_area="x", content_hash=f"h{i}", token_count=1,
        embedding_text=text,
    )


def _unit_vectors(n: int, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_matching_chunks_and_vectors_pass():
    check_alignment([_chunk(i) for i in range(5)], _unit_vectors(5), expected_dim=4)


def test_fewer_vectors_than_chunks_is_refused():
    with pytest.raises(IndexAlignmentError, match="5 chunks.*4 vectors"):
        check_alignment([_chunk(i) for i in range(5)], _unit_vectors(4), expected_dim=4)


def test_wrong_dimension_is_refused():
    with pytest.raises(IndexAlignmentError, match="dim"):
        check_alignment([_chunk(i) for i in range(3)], _unit_vectors(3, dim=8), expected_dim=4)


def test_non_unit_vectors_are_refused():
    # Cosine over non-normalized vectors ranks differently from the served index.
    vectors = _unit_vectors(3) * 2.0
    with pytest.raises(IndexAlignmentError, match="unit"):
        check_alignment([_chunk(i) for i in range(3)], vectors, expected_dim=4)


def test_duplicate_chunk_ids_are_refused():
    # chunk_point_id is a pure function of the id, so a duplicate silently overwrites a point.
    chunks = [_chunk(0), _chunk(0), _chunk(1)]
    with pytest.raises(IndexAlignmentError, match="duplicate"):
        check_alignment(chunks, _unit_vectors(3), expected_dim=4)


def test_fingerprint_changes_when_a_chunk_is_edited_or_reordered():
    chunks = [_chunk(i) for i in range(4)]
    vectors = _unit_vectors(4)
    base = corpus_fingerprint(chunks, vectors)
    assert base == corpus_fingerprint(list(chunks), vectors.copy())

    edited = list(chunks)
    edited[2] = _chunk(2, text="different")
    assert corpus_fingerprint(edited, vectors) != base

    assert corpus_fingerprint(chunks[::-1], vectors) != base
    assert corpus_fingerprint(chunks, vectors[::-1].copy()) != base
