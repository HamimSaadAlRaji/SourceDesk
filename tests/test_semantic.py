import pytest

from aegis.chunking.semantic import SemanticChunker, split_sentences
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


def _topic_embed(sentences: list[str]) -> list[list[float]]:
    """Deterministic fake embedder: sentences about 'cats' and 'dogs' point
    in orthogonal directions, anything else is neutral. Lets the breakpoint
    logic be tested without loading a real model."""
    vectors = []
    for s in sentences:
        if "cat" in s.lower():
            vectors.append([1.0, 0.0])
        elif "dog" in s.lower():
            vectors.append([0.0, 1.0])
        else:
            vectors.append([0.7, 0.7])
    return vectors


def _failing_embed(sentences: list[str]) -> list[list[float]]:
    raise AssertionError("embed_fn should not be called for a single-sentence segment")


# --- split_sentences ---


def test_split_sentences_basic():
    text = "First sentence. Second sentence. Third sentence."
    assert split_sentences(text) == ["First sentence.", "Second sentence.", "Third sentence."]


def test_split_sentences_single_sentence_no_split():
    assert split_sentences("Just one sentence with no terminator") == ["Just one sentence with no terminator"]


def test_split_sentences_handles_question_and_exclamation():
    text = "Is this a question? Yes it is! Great."
    assert split_sentences(text) == ["Is this a question?", "Yes it is!", "Great."]


# --- SemanticChunker: grouping behavior via a fake, deterministic embedder ---


def test_groups_topically_similar_sentences_and_cuts_at_topic_shift():
    # Each topic group is ~11-13 tokens; a 15-token budget lets a group
    # stand alone but not merge with its neighbor, isolating the semantic
    # grouping decision from the separate greedy-merge packing step.
    text = "Cats are great pets. Cats sleep most of the day. Dogs are loyal friends. Dogs enjoy long walks."
    chunker = SemanticChunker(
        max_tokens=15, overlap_tokens=0, breakpoint_percentile=0.5, embed_fn=_topic_embed
    )
    chunks = chunker.chunk(_doc(text))

    assert len(chunks) == 2
    assert "Cats are great" in chunks[0].text and "Cats sleep" in chunks[0].text
    assert "Dogs are loyal" in chunks[1].text and "Dogs enjoy" in chunks[1].text


def test_single_sentence_segment_never_calls_embed_fn():
    chunker = SemanticChunker(max_tokens=200, overlap_tokens=0, embed_fn=_failing_embed)
    chunks = chunker.chunk(_doc("Only one sentence here with no period-separated neighbors"))
    assert len(chunks) == 1


def test_oversized_group_falls_back_to_recursive_split():
    # All sentences look identical to the fake embedder -> no breakpoints ->
    # one big group, which must still respect max_tokens via the recursive
    # fallback rather than being emitted as one oversized chunk.
    text = " ".join(f"Cats sentence number {i} about pets." for i in range(30))
    chunker = SemanticChunker(max_tokens=25, overlap_tokens=5, embed_fn=_topic_embed)
    chunks = chunker.chunk(_doc(text))

    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.text) <= 25


def test_code_block_never_split_even_when_oversized():
    huge_code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(300)) + "\n```"
    text = f"Some intro prose about cats.\n\n{huge_code}\n\nSome outro prose about dogs."
    chunker = SemanticChunker(max_tokens=20, overlap_tokens=5, embed_fn=_topic_embed)
    chunks = chunker.chunk(_doc(text))

    code_chunks = [c for c in chunks if c.text.strip().startswith("```python")]
    assert len(code_chunks) == 1
    assert code_chunks[0].text.strip() == huge_code
    assert count_tokens(code_chunks[0].text) > 20


def test_heading_path_always_empty_no_structural_attribution():
    text = "# A Heading\n\nCats are great pets. Dogs are loyal friends."
    chunker = SemanticChunker(max_tokens=200, overlap_tokens=0, breakpoint_percentile=0.5, embed_fn=_topic_embed)
    chunks = chunker.chunk(_doc(text))
    assert all(c.heading_path == [] for c in chunks)


def test_document_metadata_propagated_and_ids_unique():
    text = "Cats are great pets. Cats nap often. Dogs are loyal. Dogs bark a lot."
    doc = _doc(text)
    chunker = SemanticChunker(max_tokens=200, overlap_tokens=0, breakpoint_percentile=0.5, embed_fn=_topic_embed)
    chunks = chunker.chunk(doc)

    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.doc_id == doc.id and c.doc_type == doc.doc_type for c in chunks)


def test_contextual_enrichment_toggle():
    text = "Cats are great pets. Dogs are loyal friends."
    plain = SemanticChunker(
        max_tokens=200, overlap_tokens=0, contextual_enrichment=False, breakpoint_percentile=0.5, embed_fn=_topic_embed
    ).chunk(_doc(text))
    enriched = SemanticChunker(
        max_tokens=200, overlap_tokens=0, contextual_enrichment=True, breakpoint_percentile=0.5, embed_fn=_topic_embed
    ).chunk(_doc(text))

    assert plain[0].embedding_text == plain[0].text
    assert enriched[0].embedding_text != enriched[0].text


def test_overlap_must_be_smaller_than_max_tokens():
    with pytest.raises(ValueError):
        SemanticChunker(max_tokens=10, overlap_tokens=10, embed_fn=_topic_embed)
