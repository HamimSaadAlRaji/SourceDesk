"""BM25 sparse retrieval (plan.md Phase 5, step 2).

`bm25s` is imported -- BM25 itself is a solved problem (CLAUDE.md §2).
What's actually built here is the wiring: the index is tokenized through
`aegis.analysis.analyzer.analyze`, not bm25s's own default tokenizer, so
`snake_case`, `dotted.path`, `SCREAMING_CASE`, and `--flag` identifiers
survive exactly as the analyzer's tests guarantee.
"""

from __future__ import annotations

import bm25s

from aegis.analysis.analyzer import analyze
from aegis.config import settings
from aegis.types import Chunk


class SparseIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        k1: float = settings.retrieval.bm25_k1,
        b: float = settings.retrieval.bm25_b,
    ) -> None:
        self.chunks = chunks
        self._bm25 = bm25s.BM25(k1=k1, b=b)
        tokenized_corpus = [analyze(chunk.embedding_text) for chunk in chunks]
        self._bm25.index(tokenized_corpus, show_progress=False)

    def search(self, query: str, top_k: int = 10, exclude_doc_id: str | None = None) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        query_tokens = [analyze(query)]
        # bm25s has no native per-query filter, unlike Qdrant's must_not
        # (DenseIndex.search). Over-fetch and filter instead -- exact, and
        # cheap since bm25s search over this corpus is fast regardless of
        # k. Needed for eval: a golden query's own source issue is
        # otherwise a trivial near-verbatim lexical match that wastes a
        # top-k slot on a chunk ground truth deliberately excludes
        # (mirrors aegis.eval.retrieve_candidates's must_not filter).
        fetch_k = top_k if exclude_doc_id is None else max(top_k * 5, 50)
        fetch_k = min(fetch_k, len(self.chunks))
        indices, scores = self._bm25.retrieve(query_tokens, k=fetch_k, show_progress=False)
        results = [(self.chunks[i], float(s)) for i, s in zip(indices[0], scores[0])]
        if exclude_doc_id is not None:
            results = [(chunk, score) for chunk, score in results if chunk.doc_id != exclude_doc_id]
        return results[:top_k]
