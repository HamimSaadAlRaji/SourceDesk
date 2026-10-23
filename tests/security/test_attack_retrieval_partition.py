"""Attacks on the retrieval-layer filter and the cache (Phase 15, tenant
separation precondition).

Before (design record 15, A1/A2): the hybrid retriever applied filters to the
dense leg only, and the cache key had no partition. Both attacks succeeded
(xfail). After (design record 15b): every leg takes every filter and the cache
scope includes the partition; these now pass as regression tests.
"""

from __future__ import annotations

from aegis.cache.exact_match import CachedAnswer, ExactMatchCache
from aegis.retrieval.hybrid import HybridRetriever, RetrievalStrategy
from aegis.retrieval.sparse import SparseIndex
from aegis.types import Chunk, DocType


def _chunk(id: str, doc_type: DocType, text: str) -> Chunk:
    return Chunk(id=id, doc_id=id, text=text, doc_type=doc_type, content_hash="h",
                 token_count=len(text.split()), embedding_text=text)


DOC = _chunk("doc::a", DocType.DOCUMENTATION, "Declare query parameters as function arguments.")
ISSUE = _chunk("issue::1", DocType.ISSUE, "uvicorn worker segfault on shutdown with uvloop")
QUERY = "uvicorn worker segfault on shutdown"  # matches the issue chunk lexically, not the doc


class _FilteringDense:
    """Honors doc_type/doc_types, as the real DenseIndex does via a Qdrant pre-filter."""

    def search(self, query, top_k=10, doc_type=None, doc_types=None, **kwargs):
        pool = [(DOC, 0.9), (ISSUE, 0.8)]
        keep = [(c, s) for c, s in pool if (doc_type is None or c.doc_type == doc_type)
                and (doc_types is None or c.doc_type in doc_types)]
        return keep[:top_k]


def _retriever() -> HybridRetriever:
    return HybridRetriever(sparse_index=SparseIndex([DOC, ISSUE]), dense_index=_FilteringDense())


def test_hybrid_doc_type_filter_holds_on_every_leg():
    results = _retriever().search(QUERY, top_k=10, strategy=RetrievalStrategy.HYBRID_RRF,
                                  doc_type=DocType.DOCUMENTATION)
    assert results, "expected results"
    assert all(c.doc_type == DocType.DOCUMENTATION for c, _ in results)


def test_sparse_strategy_honors_doc_type_filter():
    results = _retriever().search(QUERY, top_k=10, strategy=RetrievalStrategy.SPARSE,
                                  doc_type=DocType.DOCUMENTATION)
    assert all(c.doc_type == DocType.DOCUMENTATION for c, _ in results)


def test_dense_strategy_honors_doc_type_filter():
    """Control: the dense path is not vulnerable (pins it so it stays that way)."""
    results = _retriever().search(QUERY, top_k=10, strategy=RetrievalStrategy.DENSE,
                                  doc_type=DocType.DOCUMENTATION)
    assert [c.id for c, _ in results] == ["doc::a"]


def test_cache_replays_an_answer_to_any_caller_with_the_same_text(tmp_path):
    """The mechanism under attack, shown working: the key is (corpus_version,
    normalized text) and nothing else, so whoever asks second gets the first
    caller's answer -- across case and whitespace variants."""
    cache = ExactMatchCache(tmp_path / "c.sqlite")
    cache.set("How do I reset my API key?", "dev", CachedAnswer(text="answer for caller A", citations=[], provider="groq"),
              partition="full")
    hit = cache.get("  how do i RESET my api key?  ", "dev", partition="full")
    assert hit is not None and hit.text == "answer for caller A"


def test_cache_hit_does_not_cross_partitions(tmp_path):
    cache = ExactMatchCache(tmp_path / "c.sqlite")
    cache.set("How do I reset my API key?", "dev", CachedAnswer(text="docs-tenant answer", citations=[], provider="groq"),
              partition="docs")
    assert cache.get("How do I reset my API key?", "dev", partition="issues") is None
