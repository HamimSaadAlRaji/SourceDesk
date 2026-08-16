from aegis.query.hyde import HydeGenerator, HydeRetriever
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


A = _chunk("a")


def test_generate_returns_the_hypothetical_answer():
    generator = HydeGenerator(hyde_fn=lambda q: "a plausible hypothetical answer")

    assert generator.generate("how do I use dependency injection?") == "a plausible hypothetical answer"


def test_generate_falls_back_to_the_raw_query_on_empty_output():
    generator = HydeGenerator(hyde_fn=lambda q: "   ")

    assert generator.generate("how do I use dependency injection?") == "how do I use dependency injection?"


class _FakeRetriever:
    def __init__(self, by_query):
        self._by_query = by_query
        self.calls: list[str] = []

    def search(self, query, top_k=10, **kwargs):
        self.calls.append(query)
        return self._by_query.get(query, [])[:top_k]


def test_disabled_searches_the_raw_query_directly():
    base = _FakeRetriever({"raw query": [(A, 0.9)]})
    generator = HydeGenerator(hyde_fn=lambda q: "hypothetical answer")
    retriever = HydeRetriever(base, generator, enabled=False)

    results = retriever.search("raw query", top_k=1)

    assert base.calls == ["raw query"]
    assert [c.id for c, _ in results] == ["a"]


def test_enabled_searches_the_hypothetical_answer_not_the_raw_query():
    base = _FakeRetriever({"hypothetical answer": [(A, 0.9)]})
    generator = HydeGenerator(hyde_fn=lambda q: "hypothetical answer")
    retriever = HydeRetriever(base, generator, enabled=True)

    results = retriever.search("raw query", top_k=1)

    assert base.calls == ["hypothetical answer"]
    assert [c.id for c, _ in results] == ["a"]
