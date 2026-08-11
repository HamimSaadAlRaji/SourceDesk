"""TDD for the reranking pipeline wrapper (plan.md Phase 7 exit criterion:
"reranking toggleable").

`RerankedRetriever` is pure composition -- pull a wider candidate pool from
a base retriever, then rerank down to `top_k`, or skip straight to the base
retriever when disabled. Tested with fake base retriever/reranker (spies),
the same treatment `tests/test_hybrid.py` gives `HybridRetriever`.
"""

from aegis.rerank.pipeline import RerankedRetriever
from aegis.types import Chunk, DocType


def _chunk(id: str) -> Chunk:
    return Chunk(
        id=id,
        doc_id=id,
        text=f"text for {id}",
        doc_type=DocType.DOCUMENTATION,
        content_hash="hash",
        token_count=3,
        embedding_text=f"text for {id}",
    )


class _FakeRetriever:
    def __init__(self, results: list[tuple[Chunk, float]]):
        self._results = results
        self.calls: list[dict] = []

    def search(self, query, top_k=10, **kwargs):
        self.calls.append({"query": query, "top_k": top_k, **kwargs})
        return self._results[:top_k]


class _FakeReranker:
    def __init__(self, reordered: list[tuple[Chunk, float]]):
        self._reordered = reordered
        self.calls: list[dict] = []

    def rerank(self, query, candidates, top_k):
        self.calls.append({"query": query, "candidates": candidates, "top_k": top_k})
        return self._reordered[:top_k]


A, B, C = (_chunk(x) for x in ["a", "b", "c"])


def test_enabled_pulls_pool_size_candidates_then_reranks_to_top_k():
    base = _FakeRetriever([(A, 0.9), (B, 0.8), (C, 0.7)])
    reranker = _FakeReranker([(C, 5.0), (A, 4.0)])
    pipeline = RerankedRetriever(base, reranker, pool_size=3, enabled=True)

    results = pipeline.search("query", top_k=2)

    assert base.calls[0]["top_k"] == 3
    assert len(reranker.calls) == 1
    assert reranker.calls[0]["top_k"] == 2
    assert [c.id for c, _ in results] == ["c", "a"]


def test_disabled_skips_the_reranker_entirely():
    base = _FakeRetriever([(A, 0.9), (B, 0.8)])
    reranker = _FakeReranker([(B, 5.0), (A, 4.0)])
    pipeline = RerankedRetriever(base, reranker, pool_size=3, enabled=False)

    results = pipeline.search("query", top_k=2)

    assert base.calls[0]["top_k"] == 2
    assert reranker.calls == []
    assert [c.id for c, _ in results] == ["a", "b"]


def test_pool_size_never_shrinks_below_requested_top_k():
    base = _FakeRetriever([(A, 0.9)])
    reranker = _FakeReranker([(A, 5.0)])
    pipeline = RerankedRetriever(base, reranker, pool_size=3, enabled=True)

    pipeline.search("query", top_k=10)

    assert base.calls[0]["top_k"] == 10


def test_extra_kwargs_are_forwarded_to_the_base_retriever():
    base = _FakeRetriever([(A, 0.9)])
    reranker = _FakeReranker([(A, 5.0)])
    pipeline = RerankedRetriever(base, reranker, pool_size=3, enabled=True)

    pipeline.search("query", top_k=1, exclude_doc_id="issue::100", product_area="security")

    assert base.calls[0]["exclude_doc_id"] == "issue::100"
    assert base.calls[0]["product_area"] == "security"


def test_reranker_receives_the_query_and_the_base_pool():
    pool = [(A, 0.9), (B, 0.8)]
    base = _FakeRetriever(pool)
    reranker = _FakeReranker([(B, 5.0), (A, 4.0)])
    pipeline = RerankedRetriever(base, reranker, pool_size=2, enabled=True)

    pipeline.search("what is dependency injection", top_k=2)

    assert reranker.calls[0]["query"] == "what is dependency injection"
    assert reranker.calls[0]["candidates"] == pool
