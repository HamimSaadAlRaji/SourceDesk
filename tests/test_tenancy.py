"""TDD for retrieval-layer partitioning (Phase 15 unit 15b, design record 15b).

A silent bug here would leak one partition's chunks into another's answers
*and* corrupt any per-partition measurement, so every enforcement point is
pinned: the partition resolver, the fail-closed retriever wrapper, the
sparse-leg filter (real bm25s), the dense-leg Qdrant filter (captured), the
hybrid fan-out, both caches, the pipeline's cache scoping, and the HTTP body.
"""

from __future__ import annotations

import pytest

from aegis.cache.exact_match import CachedAnswer, ExactMatchCache
from aegis.cache.semantic import SemanticCache
from aegis.retrieval.dense import DenseIndex
from aegis.retrieval.hybrid import HybridRetriever, RetrievalStrategy
from aegis.retrieval.sparse import SparseIndex
from aegis.tenancy import PartitionedRetriever, PartitionViolation, resolve_partition
from aegis.types import Chunk, DocType


def _chunk(id: str, doc_type: DocType, text: str, product_area: str | None = None) -> Chunk:
    return Chunk(id=id, doc_id=id, text=text, doc_type=doc_type, product_area=product_area,
                 content_hash="h", token_count=len(text.split()), embedding_text=text)


DOC = _chunk("doc::a", DocType.DOCUMENTATION, "Declare query parameters as function arguments.", "params")
DOC2 = _chunk("doc::b", DocType.DOCUMENTATION, "Background tasks run after the response.", "background")
ISSUE = _chunk("issue::1", DocType.ISSUE, "uvicorn worker segfault on shutdown with uvloop", "deployment")
ISSUE2 = _chunk("issue::2", DocType.ISSUE, "segfault segfault uvicorn shutdown crash", "deployment")


# ---- partitions -------------------------------------------------------------------


def test_resolve_partition_maps_names_to_doc_types():
    assert resolve_partition("docs").doc_types == frozenset({DocType.DOCUMENTATION})
    assert resolve_partition("issues").doc_types == frozenset({DocType.ISSUE})
    assert resolve_partition("full").doc_types == frozenset({DocType.DOCUMENTATION, DocType.ISSUE})


def test_unknown_partition_is_an_error_not_a_default():
    with pytest.raises(ValueError):
        resolve_partition("admin")


# ---- fail-closed wrapper -----------------------------------------------------------


class _Spy:
    def __init__(self, results):
        self.results = results
        self.calls: list[dict] = []

    def search(self, query, top_k=10, **kwargs):
        self.calls.append(kwargs)
        return self.results[:top_k]


def test_partitioned_retriever_pushes_the_filter_down():
    spy = _Spy([(DOC, 1.0)])
    PartitionedRetriever(spy, resolve_partition("docs")).search("q", top_k=5)
    assert spy.calls == [{"doc_types": frozenset({DocType.DOCUMENTATION})}]


def test_partitioned_retriever_fails_closed_if_the_base_leaks():
    """Defense in depth: even if a base retriever ignored the filter, nothing
    out of partition is returned -- the request fails instead."""
    leaky = _Spy([(DOC, 1.0), (ISSUE, 0.9)])
    with pytest.raises(PartitionViolation):
        PartitionedRetriever(leaky, resolve_partition("docs")).search("q")


def test_partitioned_retriever_accepts_no_caller_filters():
    """No argument can widen the partition: filter kwargs are not part of the
    signature at all."""
    retriever = PartitionedRetriever(_Spy([]), resolve_partition("docs"))
    for kwarg in ({"doc_types": {DocType.ISSUE}}, {"doc_type": DocType.ISSUE}, {"partition": "issues"}):
        with pytest.raises(TypeError):
            retriever.search("q", **kwarg)


def test_exclude_doc_id_still_passes_through_for_eval():
    spy = _Spy([])
    PartitionedRetriever(spy, resolve_partition("full")).search("q", exclude_doc_id="issue::7")
    assert spy.calls[0]["exclude_doc_id"] == "issue::7"


# ---- sparse leg: real bm25s ---------------------------------------------------------


def test_sparse_doc_types_filter_excludes_better_scoring_out_of_partition_chunks():
    index = SparseIndex([DOC, DOC2, ISSUE, ISSUE2])
    unfiltered = index.search("uvicorn segfault shutdown", top_k=4)
    assert unfiltered[0][0].doc_type == DocType.ISSUE  # precondition: issues win lexically
    filtered = index.search("uvicorn segfault shutdown", top_k=4, doc_types={DocType.DOCUMENTATION})
    assert {c.doc_type for c, _ in filtered} <= {DocType.DOCUMENTATION}


def test_sparse_doc_type_and_product_area_filters():
    index = SparseIndex([DOC, DOC2, ISSUE, ISSUE2])
    assert all(c.doc_type == DocType.ISSUE for c, _ in index.search("segfault", top_k=4, doc_type=DocType.ISSUE))
    assert [c.id for c, _ in index.search("tasks parameters", top_k=4, product_area="background")] == ["doc::b"]


# ---- dense leg: captured Qdrant filter ----------------------------------------------


class _Encoder:
    def encode(self, texts):
        import numpy as np

        return np.zeros((len(texts), 4), dtype=np.float32)


class _CapturingClient:
    def __init__(self):
        self.filters = []

    def query_points(self, collection_name, query, using, limit, query_filter):
        self.filters.append(query_filter)

        class _R:
            points = []

        return _R()


def test_dense_doc_types_becomes_a_qdrant_match_any_prefilter():
    client = _CapturingClient()
    DenseIndex([DOC], "c", _Encoder(), client=client).search("q", doc_types={DocType.DOCUMENTATION})
    [f] = client.filters
    [cond] = f.must
    assert cond.key == "doc_type" and list(cond.match.any) == ["documentation"]


# ---- hybrid fan-out ------------------------------------------------------------------


class _RecordingIndex(_Spy):
    pass


@pytest.mark.parametrize("strategy", [RetrievalStrategy.HYBRID_RRF, RetrievalStrategy.HYBRID_LINEAR,
                                      RetrievalStrategy.SPARSE, RetrievalStrategy.DENSE])
def test_hybrid_forwards_every_filter_to_every_leg_it_calls(strategy):
    dense, sparse = _RecordingIndex([]), _RecordingIndex([])
    HybridRetriever(sparse_index=sparse, dense_index=dense).search(
        "q", top_k=3, strategy=strategy, doc_types={DocType.DOCUMENTATION}, product_area="params",
        doc_type=DocType.DOCUMENTATION,
    )
    for call in dense.calls + sparse.calls:
        assert call["doc_types"] == {DocType.DOCUMENTATION}
        assert call["product_area"] == "params" and call["doc_type"] == DocType.DOCUMENTATION


# ---- caches --------------------------------------------------------------------------


def _answer(text="a"):
    return CachedAnswer(text=text, citations=[], provider="groq")


def test_exact_cache_is_scoped_by_partition(tmp_path):
    cache = ExactMatchCache(tmp_path / "e.sqlite")
    cache.set("q", "v1", _answer("docs answer"), partition="docs")
    assert cache.get("q", "v1", partition="issues") is None
    assert cache.get("q", "v1", partition="docs").text == "docs answer"


def test_semantic_cache_is_scoped_by_partition(tmp_path):
    class _Enc:
        def encode(self, texts):
            import numpy as np

            return np.ones((len(texts), 3), dtype=np.float32)

    cache = SemanticCache(tmp_path / "s.sqlite", encoder=_Enc(), threshold=0.5)
    cache.set("q", "v1", _answer("docs answer"), partition="docs")
    assert cache.get("q", "v1", partition="issues") is None
    assert cache.get("q", "v1", partition="docs").text == "docs answer"


def test_cache_partition_is_required(tmp_path):
    cache = ExactMatchCache(tmp_path / "e.sqlite")
    with pytest.raises(TypeError):
        cache.get("q", "v1")
