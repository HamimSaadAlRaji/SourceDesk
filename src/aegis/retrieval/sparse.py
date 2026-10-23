"""BM25 sparse retrieval (plan.md Phase 5, step 2).

`bm25s` is imported -- BM25 itself is a solved problem (CLAUDE.md §2).
What's actually built here is the wiring: the index is tokenized through
`aegis.analysis.analyzer.analyze`, not bm25s's own default tokenizer, so
`snake_case`, `dotted.path`, `SCREAMING_CASE`, and `--flag` identifiers
survive exactly as the analyzer's tests guarantee.
"""

from __future__ import annotations

from collections.abc import Callable, Collection

import bm25s
import numpy as np

from aegis.analysis.analyzer import analyze
from aegis.config import settings
from aegis.types import Chunk, DocType


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
        self._masks: dict[tuple, np.ndarray] = {}

    def search(
        self,
        query: str,
        top_k: int = 10,
        exclude_doc_id: str | None = None,
        product_area: str | None = None,
        doc_type: DocType | None = None,
        doc_types: Collection[DocType] | None = None,
    ) -> list[tuple[Chunk, float]]:
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

        # Metadata filters (product_area / doc_type / a partition's doc_types).
        # These used not to exist here at all, so HybridRetriever's filters
        # silently applied to the dense leg only: 33.8% of a "documentation
        # only" top-10 was issue chunks on the real corpus (design record 15,
        # A1). A bm25s weight mask zeroes out-of-filter documents *before*
        # top-k, so in-filter recall isn't lost to over-fetching. Masked
        # documents can still come back at score 0 when fewer than k match,
        # so the post-filter below is what actually guarantees exclusion.
        allowed = self._filter_fn(product_area, doc_type, doc_types)
        weight_mask = self._mask(product_area, doc_type, doc_types) if allowed is not None else None

        indices, scores = self._bm25.retrieve(query_tokens, k=fetch_k, show_progress=False, weight_mask=weight_mask)
        results = [(self.chunks[i], float(s)) for i, s in zip(indices[0], scores[0])]
        if allowed is not None:
            results = [(chunk, score) for chunk, score in results if allowed(chunk)]
        if exclude_doc_id is not None:
            results = [(chunk, score) for chunk, score in results if chunk.doc_id != exclude_doc_id]
        return results[:top_k]

    @staticmethod
    def _filter_fn(product_area, doc_type, doc_types) -> Callable[[Chunk], bool] | None:
        if product_area is None and doc_type is None and doc_types is None:
            return None
        allowed_types = frozenset(doc_types) if doc_types is not None else None
        return lambda c: ((product_area is None or c.product_area == product_area)
                          and (doc_type is None or c.doc_type == doc_type)
                          and (allowed_types is None or c.doc_type in allowed_types))

    def _mask(self, product_area, doc_type, doc_types) -> np.ndarray:
        key = (product_area, doc_type, frozenset(doc_types) if doc_types is not None else None)
        if key not in self._masks:
            allowed = self._filter_fn(product_area, doc_type, doc_types)
            self._masks[key] = np.array([allowed(c) for c in self.chunks], dtype=np.float32)
        return self._masks[key]
