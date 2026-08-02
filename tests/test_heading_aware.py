import pytest

from aegis.chunking.heading_aware import HeadingAwareChunker
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


def test_no_headings_falls_back_to_single_section_empty_heading_path():
    chunker = HeadingAwareChunker(max_tokens=200, overlap_tokens=10)
    chunks = chunker.chunk(_doc("Just prose with no markdown headings at all."))

    assert len(chunks) == 1
    assert chunks[0].heading_path == []


def test_single_heading_produces_heading_path():
    text = "# Query Parameters\n\nSome content about query parameters."
    chunks = HeadingAwareChunker(max_tokens=200, overlap_tokens=10).chunk(_doc(text))

    assert len(chunks) == 1
    assert chunks[0].heading_path == ["Query Parameters"]
    assert "Some content" in chunks[0].text


def test_nested_headings_produce_full_breadcrumb():
    text = (
        "# Tutorial\n\nIntro text.\n\n"
        "## Query Parameters\n\nSection text.\n\n"
        "### Predefined Values\n\nDeep section text."
    )
    chunks = HeadingAwareChunker(max_tokens=200, overlap_tokens=10).chunk(_doc(text))

    paths = {tuple(c.heading_path) for c in chunks}
    assert ("Tutorial",) in paths
    assert ("Tutorial", "Query Parameters") in paths
    assert ("Tutorial", "Query Parameters", "Predefined Values") in paths


def test_sibling_headings_do_not_inherit_each_others_children():
    text = (
        "# Tutorial\n\n"
        "## Section A\n\n### Deep A\n\ncontent A\n\n"
        "## Section B\n\ncontent B"
    )
    chunks = HeadingAwareChunker(max_tokens=200, overlap_tokens=10).chunk(_doc(text))

    section_b_chunks = [c for c in chunks if "content B" in c.text]
    assert len(section_b_chunks) == 1
    assert section_b_chunks[0].heading_path == ["Tutorial", "Section B"]


def test_code_block_never_split_even_when_it_alone_exceeds_max_tokens():
    huge_code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(300)) + "\n```"
    text = f"# Example\n\nHere is a large example:\n\n{huge_code}\n\nExplanation after."
    chunker = HeadingAwareChunker(max_tokens=20, overlap_tokens=5)
    chunks = chunker.chunk(_doc(text))

    code_chunks = [c for c in chunks if c.text.strip().startswith("```python")]
    assert len(code_chunks) == 1
    assert code_chunks[0].text.strip() == huge_code
    # Documented exception: an atomic code block bigger than max_tokens still
    # produces one oversized chunk rather than being split.
    assert count_tokens(code_chunks[0].text) > 20


def test_non_code_chunks_never_exceed_max_tokens():
    huge_code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(300)) + "\n```"
    text = f"# Example\n\nHere is a large example:\n\n{huge_code}\n\nExplanation after."
    chunker = HeadingAwareChunker(max_tokens=20, overlap_tokens=5)
    chunks = chunker.chunk(_doc(text))

    for c in chunks:
        if not c.text.strip().startswith("```"):
            assert count_tokens(c.text) <= 20


def test_hash_style_comment_inside_code_block_is_not_treated_as_heading():
    text = "# Real Heading\n\n```python\n# this is a python comment, not a heading\nx = 1\n```"
    chunks = HeadingAwareChunker(max_tokens=200, overlap_tokens=10).chunk(_doc(text))

    assert len(chunks) == 1
    assert chunks[0].heading_path == ["Real Heading"]
    assert "# this is a python comment" in chunks[0].text


def test_overlap_attaches_tail_of_prose_to_following_code_chunk():
    # Code fits comfortably under max_tokens on its own (19 tokens) but
    # prose (21 tokens) + code together (40) don't fit in a 35-token budget
    # — the "normal" packing branch (not the oversized-atomic exception), so
    # overlap should carry the prose's tail into the code chunk.
    prose = "This example demonstrates dependency injection in FastAPI routes. " * 2
    code = "```python\n" + "\n".join(f"x{i}={i}" for i in range(3)) + "\n```"
    text = f"# Example\n\n{prose}\n\n{code}"
    chunker = HeadingAwareChunker(max_tokens=35, overlap_tokens=8)
    chunks = chunker.chunk(_doc(text))

    code_chunk = next(c for c in chunks if "```python" in c.text)
    assert "dependency injection" in code_chunk.text or "FastAPI routes" in code_chunk.text


def test_document_metadata_propagated_and_ids_unique():
    text = "# A\n\ncontent a\n\n# B\n\ncontent b\n\n# C\n\ncontent c"
    doc = _doc(text)
    chunks = HeadingAwareChunker(max_tokens=200, overlap_tokens=10).chunk(doc)

    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.doc_id == doc.id and c.doc_type == doc.doc_type and c.product_area == doc.product_area for c in chunks)


def test_contextual_enrichment_uses_the_chunks_own_heading_path():
    text = "# Tutorial\n\n## Query Parameters\n\nSection body text here."
    chunks = HeadingAwareChunker(max_tokens=200, overlap_tokens=10, contextual_enrichment=True).chunk(_doc(text))

    target = next(c for c in chunks if "Section body" in c.text)
    assert "Tutorial > Query Parameters" in target.embedding_text
    assert target.embedding_text.endswith(target.text)


def test_overlap_must_be_smaller_than_max_tokens():
    with pytest.raises(ValueError):
        HeadingAwareChunker(max_tokens=10, overlap_tokens=10)
