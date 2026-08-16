from aegis.query.expand import ExpandedRetriever, QueryExpander
from aegis.retrieval.fusion import reciprocal_rank_fusion
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


A, B, C = (_chunk(x) for x in ["a", "b", "c"])


def test_expand_always_includes_the_original_query_first():
    expander = QueryExpander(expand_fn=lambda q, n: ["paraphrase 1", "paraphrase 2"], n=2)

    result = expander.expand("original query")

    assert result[0] == "original query"
    assert result == ["original query", "paraphrase 1", "paraphrase 2"]


def test_expand_deduplicates_paraphrases_identical_to_the_original():
    expander = QueryExpander(expand_fn=lambda q, n: ["original query", "paraphrase 2"], n=2)

    result = expander.expand("original query")

    assert result == ["original query", "paraphrase 2"]


def test_expand_drops_blank_paraphrases():
    expander = QueryExpander(expand_fn=lambda q, n: ["", "  ", "paraphrase 2"], n=2)

    result = expander.expand("original query")

    assert result == ["original query", "paraphrase 2"]


def test_failed_expand_fn_degrades_to_just_the_original_query():
    expander = QueryExpander(expand_fn=lambda q, n: [], n=2)

    result = expander.expand("original query")

    assert result == ["original query"]


class _FakeRetriever:
    def __init__(self, by_query: dict[str, list[tuple[Chunk, float]]]):
        self._by_query = by_query
        self.calls: list[str] = []

    def search(self, query, top_k=10, **kwargs):
        self.calls.append(query)
        return self._by_query.get(query, [])[:top_k]


def test_disabled_skips_expansion_entirely():
    base = _FakeRetriever({"q": [(A, 0.9)]})
    expander = QueryExpander(expand_fn=lambda q, n: ["p1", "p2"], n=2)
    retriever = ExpandedRetriever(base, expander, enabled=False)

    results = retriever.search("q", top_k=1)

    assert base.calls == ["q"]
    assert [c.id for c, _ in results] == ["a"]


def test_enabled_searches_every_expanded_query_and_fuses_with_rrf():
    base = _FakeRetriever(
        {
            "q": [(A, 0.9), (B, 0.5)],
            "p1": [(B, 0.9), (C, 0.5)],
        }
    )
    expander = QueryExpander(expand_fn=lambda q, n: ["p1"], n=1)
    retriever = ExpandedRetriever(base, expander, enabled=True)

    results = retriever.search("q", top_k=3)

    assert set(base.calls) == {"q", "p1"}
    expected = reciprocal_rank_fusion([[(A, 0.9), (B, 0.5)], [(B, 0.9), (C, 0.5)]])
    assert [c.id for c, _ in results] == [c.id for c, _ in expected][:3]
