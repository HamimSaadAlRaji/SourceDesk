"""TDD for rank fusion (plan.md Phase 5, steps 4-5).

RRF (`score(d) = sum(weight_r / (k + rank_r(d)))`) and score-normalized
linear fusion are both ~20 lines, but they're the project's own -- no
library call to lean on, and a silent bug here corrupts the fused ranking
every downstream measurement (Phase 6's ablation grid) depends on. CLAUDE.md
§6 names RRF explicitly as a TDD component.

Both functions take and return `list[tuple[Chunk, float]]` -- the same
shape `SparseIndex.search` and `DenseIndex.search` already use (Phase 5
exit criterion: "All retrieval modes behind one interface") -- and fuse by
`chunk.id`, since dense and sparse construct independent `Chunk` instances
for what is semantically the same corpus entry.
"""

from aegis.retrieval.fusion import linear_fusion, reciprocal_rank_fusion
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


A, B, C, D = (_chunk(x) for x in ["a", "b", "c", "d"])


def _scores_by_id(fused: list[tuple[Chunk, float]]) -> dict[str, float]:
    return {chunk.id: score for chunk, score in fused}


def test_rrf_matches_the_formula_for_a_single_ranked_list():
    # score(d) = weight / (k + rank), rank 1-indexed, weight=1, k=60 default
    dense = [(A, 0.9), (B, 0.8)]

    scores = _scores_by_id(reciprocal_rank_fusion([dense]))

    assert scores["a"] == 1 / (60 + 1)
    assert scores["b"] == 1 / (60 + 2)


def test_rrf_sums_contributions_when_a_chunk_appears_in_multiple_lists():
    dense = [(A, 0.9), (B, 0.8), (C, 0.7)]
    sparse = [(C, 5.0), (A, 4.0), (B, 3.0)]

    scores = _scores_by_id(reciprocal_rank_fusion([dense, sparse]))

    assert scores["a"] == 1 / (60 + 1) + 1 / (60 + 2)
    assert scores["b"] == 1 / (60 + 2) + 1 / (60 + 3)
    assert scores["c"] == 1 / (60 + 3) + 1 / (60 + 1)


def test_rrf_rewards_cross_method_agreement_over_a_single_top_rank():
    # The classic RRF behavior the design record cites: at the default
    # damping constant (k=60), the gap between nearby top ranks is small,
    # so a document ranked decently by *both* methods outranks one ranked
    # #1 in only one -- once that method's rank is low enough.
    fillers = [_chunk(f"f{i}") for i in range(6)]
    dense = [(A, 0.9), (D, 0.5), (B, 0.4), (C, 0.3)]
    sparse = [(D, 9.0), (C, 8.0), (B, 7.0), *[(f, 1.0) for f in fillers], (A, 0.1)]

    scores = _scores_by_id(reciprocal_rank_fusion([dense, sparse]))

    # B: rank 3 in both lists. A: rank 1 in dense, rank 10 in sparse.
    assert scores["b"] > scores["a"]


def test_rrf_respects_custom_k():
    dense = [(A, 0.9)]

    default_scores = _scores_by_id(reciprocal_rank_fusion([dense]))
    custom_scores = _scores_by_id(reciprocal_rank_fusion([dense], k=10))

    assert default_scores["a"] == 1 / 61
    assert custom_scores["a"] == 1 / 11


def test_rrf_applies_per_list_weights():
    dense = [(A, 0.9)]
    sparse = [(A, 5.0)]

    scores = _scores_by_id(reciprocal_rank_fusion([dense, sparse], weights=[2.0, 0.5]))

    assert scores["a"] == 2.0 / 61 + 0.5 / 61


def test_rrf_sorts_results_descending_by_fused_score():
    dense = [(A, 0.9), (B, 0.8), (C, 0.7)]
    sparse = [(C, 5.0), (A, 4.0), (B, 3.0)]

    fused = reciprocal_rank_fusion([dense, sparse])

    ids = [chunk.id for chunk, _ in fused]
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)
    assert set(ids) == {"a", "b", "c"}


def test_rrf_with_no_ranked_lists_returns_empty():
    assert reciprocal_rank_fusion([]) == []


def test_linear_fusion_min_max_normalizes_each_list_independently():
    dense = [(A, 10.0), (B, 0.0)]  # normalizes to 1.0, 0.0
    sparse = [(A, 100.0), (B, 50.0)]  # normalizes to 1.0, 0.0

    scores = _scores_by_id(linear_fusion([dense, sparse]))

    assert scores["a"] == 2.0
    assert scores["b"] == 0.0


def test_linear_fusion_applies_weights_after_normalization():
    dense = [(A, 10.0), (B, 0.0)]

    scores = _scores_by_id(linear_fusion([dense], weights=[0.5]))

    assert scores["a"] == 0.5
    assert scores["b"] == 0.0


def test_linear_fusion_handles_equal_scores_without_dividing_by_zero():
    # max == min in this list -- must not raise ZeroDivisionError.
    dense = [(A, 3.0), (B, 3.0)]

    scores = _scores_by_id(linear_fusion([dense]))

    assert scores["a"] == 1.0
    assert scores["b"] == 1.0


def test_linear_fusion_sums_normalized_scores_across_lists():
    dense = [(A, 10.0), (B, 0.0)]
    sparse = [(A, 0.0), (B, 10.0)]

    scores = _scores_by_id(linear_fusion([dense, sparse]))

    assert scores["a"] == 1.0
    assert scores["b"] == 1.0


def test_linear_fusion_sorts_results_descending():
    dense = [(A, 1.0), (B, 5.0), (C, 3.0)]

    fused = linear_fusion([dense])

    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)
