import pytest

from aegis.chunking.recursive import RecursiveChunker, recursive_split_pieces
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


# --- recursive_split_pieces: the separator-descent logic in isolation ---


def test_pieces_never_exceed_max_tokens():
    text = (
        "Paragraph one about FastAPI dependency injection and how it works in practice.\n\n"
        "Paragraph two about Pydantic models and validation. It has multiple sentences. "
        "Here is a third sentence to pad it out further.\n\n"
        "Paragraph three is short."
    )
    pieces = recursive_split_pieces(text, max_tokens=15)
    assert all(count_tokens(p) <= 15 for p in pieces)


def test_prefers_paragraph_boundary_when_it_fits():
    text = "Short first paragraph.\n\nShort second paragraph."
    pieces = recursive_split_pieces(text, max_tokens=100)
    assert pieces == [text]  # whole thing fits, no split needed


def test_oversized_paragraph_falls_back_to_sentence_split():
    para = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
    pieces = recursive_split_pieces(para, max_tokens=10)
    assert len(pieces) > 1
    assert all(count_tokens(p) <= 10 for p in pieces)


def test_unbroken_text_with_no_separators_uses_hard_token_fallback():
    text = "x" * 2000  # single unbroken token-ish blob, no spaces at all
    pieces = recursive_split_pieces(text, max_tokens=20)
    assert len(pieces) > 1
    assert all(count_tokens(p) <= 20 for p in pieces)
    assert "".join(pieces) == text


def test_hard_fallback_never_corrupts_a_multibyte_character():
    text = "a" * 50 + "🎉" + "b" * 50  # no separators anywhere near the emoji
    pieces = recursive_split_pieces(text, max_tokens=10)
    assert all("�" not in p for p in pieces)
    assert "🎉" in "".join(pieces)


# --- RecursiveChunker: full pipeline, TDD against the Chunker protocol ---


def test_short_document_is_a_single_chunk():
    chunker = RecursiveChunker(max_tokens=100, overlap_tokens=10)
    chunks = chunker.chunk(_doc("Just a short paragraph about FastAPI."))
    assert len(chunks) == 1
    assert chunks[0].text == "Just a short paragraph about FastAPI."


def test_long_document_splits_and_respects_max_tokens():
    text = "\n\n".join(f"Paragraph {i} talks about routing, dependencies, and validation." for i in range(30))
    chunker = RecursiveChunker(max_tokens=25, overlap_tokens=5)
    chunks = chunker.chunk(_doc(text))

    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.text) <= 25


def test_document_metadata_propagated_and_ids_unique():
    text = "\n\n".join(f"Paragraph {i} about FastAPI." for i in range(20))
    doc = _doc(text)
    chunker = RecursiveChunker(max_tokens=15, overlap_tokens=3)
    chunks = chunker.chunk(doc)

    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.doc_id == doc.id and c.doc_type == doc.doc_type for c in chunks)


def test_contextual_enrichment_toggle():
    text = "Some prose about path operations and query parameters here."
    plain = RecursiveChunker(max_tokens=100, overlap_tokens=5, contextual_enrichment=False).chunk(_doc(text))
    enriched = RecursiveChunker(max_tokens=100, overlap_tokens=5, contextual_enrichment=True).chunk(_doc(text))

    assert plain[0].embedding_text == plain[0].text
    assert enriched[0].embedding_text != enriched[0].text
    assert enriched[0].embedding_text.endswith(text)


def test_overlap_tokens_must_be_smaller_than_max_tokens():
    with pytest.raises(ValueError):
        RecursiveChunker(max_tokens=10, overlap_tokens=10)
