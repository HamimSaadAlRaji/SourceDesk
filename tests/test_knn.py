import numpy as np

from aegis.embedding.knn import brute_force_knn, brute_force_knn_batch, recall_at_k


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def test_brute_force_knn_returns_closest_by_cosine_similarity():
    # Four points at 0°, 30°, 90°, 180° on the unit circle; query at 0°.
    corpus = np.array(
        [
            _unit(np.array([1.0, 0.0])),  # 0°  — identical to query
            _unit(np.array([np.sqrt(3), 1.0])),  # 30°
            _unit(np.array([0.0, 1.0])),  # 90°
            _unit(np.array([-1.0, 0.0])),  # 180° — most distant
        ],
        dtype=np.float32,
    )
    query = _unit(np.array([1.0, 0.0])).astype(np.float32)

    result = brute_force_knn(query, corpus, k=2)

    assert list(result) == [0, 1]


def test_brute_force_knn_k_larger_than_corpus_returns_all_sorted():
    corpus = np.array([_unit(np.array([1.0, 0.0])), _unit(np.array([0.0, 1.0]))], dtype=np.float32)
    query = _unit(np.array([1.0, 0.0])).astype(np.float32)

    result = brute_force_knn(query, corpus, k=10)

    assert len(result) == 2
    assert result[0] == 0


def test_brute_force_knn_batch_matches_single_query_version():
    rng = np.random.default_rng(42)
    corpus = rng.normal(size=(50, 8)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.normal(size=(5, 8)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    batch_result = brute_force_knn_batch(queries, corpus, k=5)

    for i, query in enumerate(queries):
        single_result = brute_force_knn(query, corpus, k=5)
        assert list(batch_result[i]) == list(single_result)


def test_recall_at_k_perfect_match_is_one():
    exact = [[0, 1, 2], [3, 4, 5]]
    approx = [[0, 1, 2], [3, 4, 5]]
    assert recall_at_k(approx, exact) == 1.0


def test_recall_at_k_partial_overlap():
    exact = [[0, 1, 2, 3]]
    approx = [[0, 1, 9, 9]]  # 2 of 4 true neighbors found
    assert recall_at_k(approx, exact) == 0.5


def test_recall_at_k_order_independent():
    exact = [[0, 1, 2]]
    approx = [[2, 0, 1]]  # same set, different order
    assert recall_at_k(approx, exact) == 1.0
