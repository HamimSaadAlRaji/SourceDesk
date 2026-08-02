import pytest

from aegis.chunking.fixed import FixedChunker
from aegis.chunking.tokens import count_tokens
from aegis.types import DocType, Document


def _doc(text: str, **overrides) -> Document:
    base = dict(
        id="doc::tutorial/example.md",
        doc_type=DocType.DOCUMENTATION,
        text=text,
        content_hash="irrelevant-doc-level-hash",
        title="Example",
        doc_version="0.115.0",
        product_area="request-data",
    )
    base.update(overrides)
    return Document(**base)


def test_short_document_is_a_single_chunk():
    chunker = FixedChunker(max_tokens=100, overlap_tokens=10)
    chunks = chunker.chunk(_doc("Just a short paragraph about FastAPI."))

    assert len(chunks) == 1
    assert chunks[0].text == "Just a short paragraph about FastAPI."


def test_long_document_splits_and_respects_max_tokens():
    text = "FastAPI dependency injection lets you share logic across routes. " * 50
    chunker = FixedChunker(max_tokens=30, overlap_tokens=5)
    chunks = chunker.chunk(_doc(text))

    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.text) <= 30


def test_overlap_present_between_adjacent_chunks():
    text = "word{} ".format(0) * 1
    text = " ".join(f"token{i}" for i in range(200))
    chunker = FixedChunker(max_tokens=30, overlap_tokens=10)
    chunks = chunker.chunk(_doc(text))

    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        assert set(prev.text.split()) & set(nxt.text.split())


def test_heading_path_is_always_empty_naive_strategy():
    chunker = FixedChunker(max_tokens=10, overlap_tokens=2)
    text = " ".join(f"token{i}" for i in range(50))
    chunks = chunker.chunk(_doc(text))

    assert all(c.heading_path == [] for c in chunks)


def test_document_metadata_propagated_to_every_chunk():
    chunker = FixedChunker(max_tokens=10, overlap_tokens=2)
    text = " ".join(f"token{i}" for i in range(50))
    doc = _doc(text)
    chunks = chunker.chunk(doc)

    for c in chunks:
        assert c.doc_id == doc.id
        assert c.doc_type == doc.doc_type
        assert c.doc_version == doc.doc_version
        assert c.product_area == doc.product_area


def test_chunk_ids_are_unique():
    chunker = FixedChunker(max_tokens=10, overlap_tokens=2)
    text = " ".join(f"token{i}" for i in range(50))
    chunks = chunker.chunk(_doc(text))

    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_contextual_enrichment_toggle_changes_embedding_text_only():
    text = "Some prose about path operations."
    plain_chunks = FixedChunker(max_tokens=100, overlap_tokens=5, contextual_enrichment=False).chunk(_doc(text))
    enriched_chunks = FixedChunker(max_tokens=100, overlap_tokens=5, contextual_enrichment=True).chunk(_doc(text))

    assert plain_chunks[0].embedding_text == plain_chunks[0].text
    assert enriched_chunks[0].text == plain_chunks[0].text
    assert enriched_chunks[0].embedding_text != enriched_chunks[0].text
    assert enriched_chunks[0].embedding_text.endswith(text)


def test_never_corrupts_a_multibyte_character_at_a_forced_split_boundary():
    # Emoji sits right where a tight max_tokens budget would otherwise force
    # a cut through it (see aegis.chunking.tokens.nearest_safe_boundary).
    text = "Congratulations! You created your first API. 🎉 Keep going with FastAPI and Pydantic."
    chunker = FixedChunker(max_tokens=12, overlap_tokens=2)
    chunks = chunker.chunk(_doc(text))

    assert all("�" not in c.text for c in chunks)
    assert "🎉" in "".join(c.text for c in chunks)


def test_overlap_must_be_smaller_than_max_tokens():
    with pytest.raises(ValueError):
        FixedChunker(max_tokens=10, overlap_tokens=10)
