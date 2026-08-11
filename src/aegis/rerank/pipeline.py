"""Composes a base retriever with the cross-encoder reranker into one
`.search()` interface (plan.md Phase 7 exit criterion: reranking
toggleable). `enabled=False` skips the reranker entirely and asks the base
retriever for `top_k` directly, so the harness can A/B the exact same
config with reranking on or off by flipping one flag -- the same shape as
`HybridRetriever`'s strategy switch (Phase 5).

Pure composition, tested with fake retriever/reranker (spies) in
`tests/test_rerank_pipeline.py` -- the real path (`HybridRetriever` +
`CrossEncoderReranker`) is exercised by `experiments/07_rerank_sweep.py`.
"""

from __future__ import annotations

from typing import Protocol

from aegis.types import Chunk


class _SearchableRetriever(Protocol):
    def search(self, query: str, top_k: int, **kwargs) -> list[tuple[Chunk, float]]: ...


class _Reranker(Protocol):
    def rerank(self, query: str, candidates: list[tuple[Chunk, float]], top_k: int) -> list[tuple[Chunk, float]]: ...


class RerankedRetriever:
    def __init__(
        self,
        base_retriever: _SearchableRetriever,
        reranker: _Reranker,
        pool_size: int,
        enabled: bool = True,
    ) -> None:
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.pool_size = pool_size
        self.enabled = enabled

    def search(self, query: str, top_k: int = 10, **kwargs) -> list[tuple[Chunk, float]]:
        if not self.enabled:
            return self.base_retriever.search(query, top_k=top_k, **kwargs)

        pool = self.base_retriever.search(query, top_k=max(top_k, self.pool_size), **kwargs)
        return self.reranker.rerank(query, pool, top_k)
