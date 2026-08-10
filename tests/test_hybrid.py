"""TDD for the composed retrieval pipeline (plan.md Phase 5, step 6).

`HybridRetriever` is pure strategy-switch/composition logic: which
index(es) to call, how large a candidate pool to pull before fusing, which
fusion function to apply, and truncating to `top_k`. It's tested here with
fake sparse/dense indexes (recording spies) rather than the real ones --
the real `SparseIndex` is exercised by `tests/test_sparse.py`, and the real
`DenseIndex` needs live Qdrant and is integration-verified by actually
running it (CLAUDE.md §6), the same treatment as the rest of Phase 3/4's
I/O-bound components. This file is only responsible for the composition,
which is exactly the kind of logic a fake can catch a silent bug in
without needing real infrastructure.
"""

import pytest

from aegis.retrieval.hybrid import HybridRetriever, RetrievalStrategy
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


class _FakeIndex:
    """Records every call it receives and returns canned results."""

    def __init__(self, results: list[tuple[Chunk, float]]):
        self._results = results
        self.calls: list[dict] = []

    def search(self, query, top_k=10, **kwargs):
        self.calls.append({"query": query, "top_k": top_k, **kwargs})
        return self._results[:top_k]


A, B, C = (_chunk(x) for x in ["a", "b", "c"])


def test_dense_strategy_calls_only_dense_index():
    dense = _FakeIndex([(A, 0.9), (B, 0.8)])
    sparse = _FakeIndex([(C, 5.0)])
    retriever = HybridRetriever(sparse_index=sparse, dense_index=dense)

    results = retriever.search("query", top_k=2, strategy=RetrievalStrategy.DENSE)

    assert [c.id for c, _ in results] == ["a", "b"]
    assert len(dense.calls) == 1
    assert len(sparse.calls) == 0


def test_sparse_strategy_calls_only_sparse_index():
    dense = _FakeIndex([(A, 0.9)])
    sparse = _FakeIndex([(C, 5.0), (B, 4.0)])
    retriever = HybridRetriever(sparse_index=sparse, dense_index=dense)

    results = retriever.search("query", top_k=2, strategy=RetrievalStrategy.SPARSE)

    assert [c.id for c, _ in results] == ["c", "b"]
    assert len(dense.calls) == 0
    assert len(sparse.calls) == 1


def test_hybrid_rrf_strategy_calls_both_indexes_and_fuses():
    dense = _FakeIndex([(A, 0.9), (B, 0.8)])
    sparse = _FakeIndex([(B, 5.0), (A, 4.0)])
    retriever = HybridRetriever(sparse_index=sparse, dense_index=dense)

    results = retriever.search("query", top_k=2, strategy=RetrievalStrategy.HYBRID_RRF)

    assert len(dense.calls) == 1
    assert len(sparse.calls) == 1
    # A is rank 1 in dense + rank 2 in sparse, B is rank 2 in dense + rank
    # 1 in sparse -- symmetric, so they tie under equal weights.
    assert {c.id for c, _ in results} == {"a", "b"}


def test_hybrid_linear_strategy_calls_both_indexes_and_fuses():
    dense = _FakeIndex([(A, 10.0), (B, 0.0)])
    sparse = _FakeIndex([(A, 10.0), (B, 0.0)])
    retriever = HybridRetriever(sparse_index=sparse, dense_index=dense)

    results = retriever.search("query", top_k=2, strategy=RetrievalStrategy.HYBRID_LINEAR)

    assert [c.id for c, _ in results] == ["a", "b"]


def test_hybrid_strategies_request_a_wider_pool_than_top_k():
    # Fusion needs breadth to find cross-method overlap; requesting only
    # top_k from each side before fusing would silently starve it.
    dense = _FakeIndex([(A, 0.9), (B, 0.8), (C, 0.7)])
    sparse = _FakeIndex([(C, 5.0), (B, 4.0), (A, 3.0)])
    retriever = HybridRetriever(sparse_index=sparse, dense_index=dense, candidate_pool_size=3)

    retriever.search("query", top_k=1, strategy=RetrievalStrategy.HYBRID_RRF)

    assert dense.calls[0]["top_k"] == 3
    assert sparse.calls[0]["top_k"] == 3


def test_results_are_truncated_to_top_k_after_fusion():
    dense = _FakeIndex([(A, 0.9), (B, 0.8), (C, 0.7)])
    sparse = _FakeIndex([(A, 5.0), (B, 4.0), (C, 3.0)])
    retriever = HybridRetriever(sparse_index=sparse, dense_index=dense, candidate_pool_size=3)

    results = retriever.search("query", top_k=1, strategy=RetrievalStrategy.HYBRID_RRF)

    assert len(results) == 1


def test_dense_strategy_forwards_metadata_filters():
    dense = _FakeIndex([(A, 0.9)])
    sparse = _FakeIndex([])
    retriever = HybridRetriever(sparse_index=sparse, dense_index=dense)

    retriever.search("query", strategy=RetrievalStrategy.DENSE, product_area="security", doc_type=DocType.DOCUMENTATION)

    assert dense.calls[0]["product_area"] == "security"
    assert dense.calls[0]["doc_type"] == DocType.DOCUMENTATION


def test_exclude_doc_id_is_forwarded_to_both_indexes_for_hybrid_strategies():
    dense = _FakeIndex([(A, 0.9)])
    sparse = _FakeIndex([(B, 5.0)])
    retriever = HybridRetriever(sparse_index=sparse, dense_index=dense)

    retriever.search("query", strategy=RetrievalStrategy.HYBRID_RRF, exclude_doc_id="issue::100")

    assert dense.calls[0]["exclude_doc_id"] == "issue::100"
    assert sparse.calls[0]["exclude_doc_id"] == "issue::100"


def test_exclude_doc_id_is_forwarded_for_sparse_strategy():
    sparse = _FakeIndex([(B, 5.0)])
    retriever = HybridRetriever(sparse_index=sparse, dense_index=_FakeIndex([]))

    retriever.search("query", strategy=RetrievalStrategy.SPARSE, exclude_doc_id="issue::100")

    assert sparse.calls[0]["exclude_doc_id"] == "issue::100"


def test_unknown_strategy_raises_value_error():
    retriever = HybridRetriever(sparse_index=_FakeIndex([]), dense_index=_FakeIndex([]))

    with pytest.raises(ValueError):
        retriever.search("query", strategy="not_a_real_strategy")
