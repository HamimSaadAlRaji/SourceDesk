from aegis.eval.negatives import build_negative_queries
from aegis.types import NegativeCategory


def test_exactly_forty_negatives_ten_per_category():
    queries = build_negative_queries()
    assert len(queries) == 40
    for category in NegativeCategory:
        assert sum(1 for q in queries if q.negative_category == category) == 10


def test_all_negatives_are_flagged_and_have_expected_behavior():
    queries = build_negative_queries()
    for q in queries:
        assert q.is_negative is True
        assert q.expected_behavior
        assert q.ground_truth_chunk_ids == []
        assert q.stratum is None


def test_negative_ids_are_unique():
    queries = build_negative_queries()
    ids = [q.id for q in queries]
    assert len(ids) == len(set(ids))
