"""TDD for document-level ground-truth transfer (Phase 6, chunking axis).

The golden set's `ground_truth_chunk_ids` are heading-aware chunk IDs
(`doc::foo.md::heading_aware::10`) -- they don't exist at all in
fixed/recursive/semantic chunking's ID space, since every chunker embeds
its own strategy name in the chunk ID (`Chunk.doc_id` is the one field
that stays stable across all four -- confirmed by grepping every chunker's
`Chunk(...)` call). Comparing chunking strategies fairly requires scoring
at the document level for this one axis: a chunk counts as relevant if it
belongs to one of the ground-truth chunks' source documents.
"""

from aegis.eval.doc_level import chunk_ids_to_doc_ids, doc_level_search_fn


def test_chunk_ids_to_doc_ids_maps_and_dedupes():
    chunk_id_to_doc_id = {"c1": "d1", "c2": "d1", "c3": "d2"}

    result = chunk_ids_to_doc_ids(["c1", "c2", "c3"], chunk_id_to_doc_id)

    assert result == ["d1", "d2"]


def test_chunk_ids_to_doc_ids_skips_unknown_chunk_ids():
    chunk_id_to_doc_id = {"c1": "d1"}

    result = chunk_ids_to_doc_ids(["c1", "unknown_chunk"], chunk_id_to_doc_id)

    assert result == ["d1"]


def test_doc_level_search_fn_dedupes_keeping_first_rank_occurrence():
    chunk_id_to_doc_id = {"c1": "d1", "c2": "d1", "c3": "d2"}
    inner_search_fn = lambda q: [("c1", 0.9), ("c2", 0.8), ("c3", 0.7)]  # noqa: E731

    wrapped = doc_level_search_fn(inner_search_fn, chunk_id_to_doc_id)

    assert wrapped("some query") == [("d1", 0.9), ("d2", 0.7)]


def test_doc_level_search_fn_skips_chunks_with_unknown_doc_mapping():
    chunk_id_to_doc_id = {"c1": "d1"}
    inner_search_fn = lambda q: [("c1", 0.9), ("unmapped_chunk", 0.5)]  # noqa: E731

    wrapped = doc_level_search_fn(inner_search_fn, chunk_id_to_doc_id)

    assert wrapped("some query") == [("d1", 0.9)]


def test_doc_level_search_fn_passes_the_query_through_unchanged():
    seen = []

    def inner_search_fn(q):
        seen.append(q)
        return []

    wrapped = doc_level_search_fn(inner_search_fn, {})
    wrapped("the exact query object")

    assert seen == ["the exact query object"]
