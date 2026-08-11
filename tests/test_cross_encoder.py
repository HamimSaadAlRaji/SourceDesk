"""TDD for the cross-encoder reranker (plan.md Phase 7, step 1).

`CrossEncoderReranker` is pure reorder/truncate logic over an injected
`score_fn` -- tested here with a fake, the same treatment
`aegis.embedding.encoder.Encoder` gets in `tests/test_encoder.py`. The real
`bge-reranker-v2-m3` path is exercised by `experiments/07_rerank_sweep.py`,
where it's genuinely GPU-bound.
"""

from aegis.rerank.cross_encoder import CrossEncoderReranker
from aegis.types import Chunk, DocType


def _chunk(id: str, text: str = "") -> Chunk:
    return Chunk(
        id=id,
        doc_id=id,
        text=text or f"text for {id}",
        doc_type=DocType.DOCUMENTATION,
        content_hash="hash",
        token_count=3,
        embedding_text=text or f"text for {id}",
    )


class _FakeScorer:
    """Scores by a fixed lookup table keyed on chunk text, and records
    every batch of pairs it was called with."""

    def __init__(self, scores_by_text: dict[str, float]):
        self.scores_by_text = scores_by_text
        self.calls: list[list[tuple[str, str]]] = []

    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        return [self.scores_by_text[text] for _query, text in pairs]


A, B, C = (_chunk(x, text=f"passage {x}") for x in ["a", "b", "c"])


def test_rerank_reorders_by_descending_score():
    scorer = _FakeScorer({"passage a": 0.1, "passage b": 0.9, "passage c": 0.5})
    reranker = CrossEncoderReranker(score_fn=scorer)
    candidates = [(A, 0.9), (B, 0.5), (C, 0.7)]  # bi-encoder order: a, c, b

    results = reranker.rerank("query", candidates, top_k=3)

    assert [c.id for c, _ in results] == ["b", "c", "a"]


def test_rerank_truncates_to_top_k():
    scorer = _FakeScorer({"passage a": 0.1, "passage b": 0.9, "passage c": 0.5})
    reranker = CrossEncoderReranker(score_fn=scorer)
    candidates = [(A, 0.0), (B, 0.0), (C, 0.0)]

    results = reranker.rerank("query", candidates, top_k=2)

    assert [c.id for c, _ in results] == ["b", "c"]


def test_rerank_scores_query_and_chunk_text_pairs():
    scorer = _FakeScorer({"passage a": 1.0, "passage b": 1.0})
    reranker = CrossEncoderReranker(score_fn=scorer)
    candidates = [(A, 0.0), (B, 0.0)]

    reranker.rerank("what is dependency injection", candidates, top_k=2)

    assert scorer.calls == [[("what is dependency injection", "passage a"), ("what is dependency injection", "passage b")]]


def test_rerank_on_empty_candidates_returns_empty_and_makes_no_calls():
    scorer = _FakeScorer({})
    reranker = CrossEncoderReranker(score_fn=scorer)

    results = reranker.rerank("query", [], top_k=5)

    assert results == []
    assert scorer.calls == []


def test_rerank_with_pool_smaller_than_top_k_returns_all_sorted():
    scorer = _FakeScorer({"passage a": 0.2, "passage b": 0.8})
    reranker = CrossEncoderReranker(score_fn=scorer)
    candidates = [(A, 0.0), (B, 0.0)]

    results = reranker.rerank("query", candidates, top_k=10)

    assert [c.id for c, _ in results] == ["b", "a"]


def test_rerank_scores_are_returned_alongside_chunks():
    scorer = _FakeScorer({"passage a": 0.1, "passage b": 0.9})
    reranker = CrossEncoderReranker(score_fn=scorer)
    candidates = [(A, 0.0), (B, 0.0)]

    results = reranker.rerank("query", candidates, top_k=2)

    assert results[0] == (B, 0.9)
    assert results[1] == (A, 0.1)
