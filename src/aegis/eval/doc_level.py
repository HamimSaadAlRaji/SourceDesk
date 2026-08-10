"""Document-level ground-truth transfer (Phase 6, chunking axis).

The golden set's `ground_truth_chunk_ids` are heading-aware chunk IDs
(`doc::foo.md::heading_aware::10`). Every chunker embeds its own strategy
name in the chunk ID, so those IDs don't exist at all in fixed/recursive/
semantic chunking's ID space -- `Chunk.doc_id` is the one field confirmed
stable across all four (`doc_id=document.id` in every chunker). Comparing
chunking strategies fairly means scoring this one axis at the document
level: a retrieved chunk counts as relevant if it belongs to one of the
ground-truth chunks' source documents, not by exact chunk-ID match.
"""

from __future__ import annotations

from typing import Callable

SearchFn = Callable[[object], list[tuple[str, float]]]


def chunk_ids_to_doc_ids(chunk_ids: list[str], chunk_id_to_doc_id: dict[str, str]) -> list[str]:
    """Ground truth transfer: chunk-level ground truth -> the sorted,
    deduplicated set of doc_ids those chunks belong to. A chunk_id absent
    from the mapping (shouldn't happen for a frozen golden set against its
    own indexed corpus) is skipped rather than raising."""
    return sorted({chunk_id_to_doc_id[cid] for cid in chunk_ids if cid in chunk_id_to_doc_id})


def doc_level_search_fn(search_fn: SearchFn, chunk_id_to_doc_id: dict[str, str]) -> SearchFn:
    """Wrap a chunk-returning search_fn so it returns rank-ordered,
    deduplicated doc_ids instead. The first (best-ranked) chunk from a
    given document wins that document's score; a chunk_id with no known
    doc mapping is skipped."""

    def wrapped(query):
        doc_hits: list[tuple[str, float]] = []
        seen_docs: set[str] = set()
        for chunk_id, score in search_fn(query):
            doc_id = chunk_id_to_doc_id.get(chunk_id)
            if doc_id is not None and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                doc_hits.append((doc_id, score))
        return doc_hits

    return wrapped
