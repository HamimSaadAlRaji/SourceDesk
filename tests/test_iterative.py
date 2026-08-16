from aegis.query.iterative import IterativeRetriever, QueryReformulator
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


A, B = (_chunk(x) for x in ["a", "b"])


class _FakeRetriever:
    def __init__(self, by_query):
        self._by_query = by_query
        self.calls: list[str] = []

    def search(self, query, top_k=10, **kwargs):
        self.calls.append(query)
        return self._by_query.get(query, [])[:top_k]


def test_high_confidence_first_try_never_reformulates():
    base = _FakeRetriever({"q": [(A, 0.9)]})
    reformulator = QueryReformulator(reformulate_fn=lambda q: "should not be called")
    retriever = IterativeRetriever(base, reformulator, score_threshold=0.5, max_retries=1, enabled=True)

    results = retriever.search("q", top_k=1)

    assert base.calls == ["q"]
    assert [c.id for c, _ in results] == ["a"]


def test_low_confidence_triggers_exactly_one_retry():
    base = _FakeRetriever({"q": [(A, 0.1)], "reformulated q": [(B, 0.9)]})
    reformulator = QueryReformulator(reformulate_fn=lambda q: f"reformulated {q}")
    retriever = IterativeRetriever(base, reformulator, score_threshold=0.5, max_retries=1, enabled=True)

    results = retriever.search("q", top_k=1)

    assert base.calls == ["q", "reformulated q"]
    assert [c.id for c, _ in results] == ["b"]


def test_retry_is_capped_even_if_still_low_confidence():
    base = _FakeRetriever({"q": [(A, 0.1)], "reformulated q": [(A, 0.1)]})
    reformulator = QueryReformulator(reformulate_fn=lambda q: f"reformulated {q}")
    retriever = IterativeRetriever(base, reformulator, score_threshold=0.5, max_retries=1, enabled=True)

    retriever.search("q", top_k=1)

    assert base.calls == ["q", "reformulated q"]


def test_empty_results_count_as_low_confidence():
    base = _FakeRetriever({"q": [], "reformulated q": [(A, 0.9)]})
    reformulator = QueryReformulator(reformulate_fn=lambda q: f"reformulated {q}")
    retriever = IterativeRetriever(base, reformulator, score_threshold=0.5, max_retries=1, enabled=True)

    results = retriever.search("q", top_k=1)

    assert [c.id for c, _ in results] == ["a"]


def test_disabled_never_retries_even_on_low_confidence():
    base = _FakeRetriever({"q": [(A, 0.1)]})
    reformulator = QueryReformulator(reformulate_fn=lambda q: "should not be called")
    retriever = IterativeRetriever(base, reformulator, score_threshold=0.5, max_retries=1, enabled=False)

    results = retriever.search("q", top_k=1)

    assert base.calls == ["q"]
    assert [c.id for c, _ in results] == ["a"]


def test_reformulator_fallback_on_empty_output():
    reformulator = QueryReformulator(reformulate_fn=lambda q: "   ")

    assert reformulator.reformulate("original") == "original"
