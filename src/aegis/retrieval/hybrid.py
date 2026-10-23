"""The composed retrieval pipeline (plan.md Phase 5, step 6): one interface
over dense, sparse, and both fusion strategies, switched by config.

This is the object Phase 6's ablation grid drives directly -- each cell of
the retrieval axis (`dense · bm25 · hybrid-rrf · hybrid-linear`, plan.md
Phase 6 step 3) is one `RetrievalStrategy` value passed to `.search()`.
"""

from __future__ import annotations

from collections.abc import Collection
from enum import StrEnum
from typing import Protocol

from aegis.config import settings
from aegis.retrieval.fusion import linear_fusion, reciprocal_rank_fusion
from aegis.types import Chunk, DocType


class RetrievalStrategy(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_LINEAR = "hybrid_linear"


class _SearchableIndex(Protocol):
    def search(self, query: str, top_k: int, **kwargs) -> list[tuple[Chunk, float]]: ...


class HybridRetriever:
    def __init__(
        self,
        sparse_index: _SearchableIndex,
        dense_index: _SearchableIndex,
        candidate_pool_size: int = settings.retrieval.hybrid_candidate_pool_size,
    ) -> None:
        self.sparse_index = sparse_index
        self.dense_index = dense_index
        self.candidate_pool_size = candidate_pool_size

    def search(
        self,
        query: str,
        top_k: int = 10,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RRF,
        product_area: str | None = None,
        doc_type: DocType | None = None,
        exclude_doc_id: str | None = None,
        doc_types: Collection[DocType] | None = None,
    ) -> list[tuple[Chunk, float]]:
        # Every filter goes to every leg. The sparse leg used to get only
        # exclude_doc_id, so filters silently applied to dense alone (design
        # record 15, A1: 33.8% of a docs-only top-10 was issues on the real corpus).
        filters = dict(product_area=product_area, doc_type=doc_type, exclude_doc_id=exclude_doc_id, doc_types=doc_types)
        if strategy == RetrievalStrategy.DENSE:
            return self.dense_index.search(query, top_k=top_k, **filters)

        if strategy == RetrievalStrategy.SPARSE:
            return self.sparse_index.search(query, top_k=top_k, **filters)

        if strategy not in (RetrievalStrategy.HYBRID_RRF, RetrievalStrategy.HYBRID_LINEAR):
            raise ValueError(f"Unknown retrieval strategy: {strategy!r}")

        pool = max(top_k, self.candidate_pool_size)
        dense_results = self.dense_index.search(query, top_k=pool, **filters)
        sparse_results = self.sparse_index.search(query, top_k=pool, **filters)

        if strategy == RetrievalStrategy.HYBRID_RRF:
            fused = reciprocal_rank_fusion(
                [dense_results, sparse_results],
                weights=[settings.retrieval.rrf_dense_weight, settings.retrieval.rrf_sparse_weight],
            )
        else:
            fused = linear_fusion([dense_results, sparse_results])

        return fused[:top_k]
