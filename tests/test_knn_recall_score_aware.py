from aegis.embedding.knn import recall_at_k_score_aware


def test_exact_id_match_scores_perfect_recall():
    approx_scores = [[0.9, 0.8, 0.7]]
    exact_kth_scores = [0.7]
    assert recall_at_k_score_aware(approx_scores, exact_kth_scores, k=3) == 1.0


def test_tied_neighbor_with_different_id_still_counts_as_correct():
    # A different point tied at exactly the k-th-best score is an equally
    # valid answer — this is the whole point of the score-aware metric.
    approx_scores = [[0.95, 0.9, 0.7]]  # third slot ties the true k-th score
    exact_kth_scores = [0.7]
    assert recall_at_k_score_aware(approx_scores, exact_kth_scores, k=3) == 1.0


def test_worse_than_kth_score_is_not_correct():
    approx_scores = [[0.95, 0.9, 0.5]]  # third slot is worse than true k-th
    exact_kth_scores = [0.7]
    assert recall_at_k_score_aware(approx_scores, exact_kth_scores, k=3) == 2 / 3


def test_epsilon_tolerance_absorbs_float_noise():
    approx_scores = [[0.9, 0.7 - 1e-7]]
    exact_kth_scores = [0.7]
    assert recall_at_k_score_aware(approx_scores, exact_kth_scores, k=2) == 1.0


def test_averages_across_queries():
    approx_scores = [[0.9, 0.8], [0.9, 0.1]]
    exact_kth_scores = [0.8, 0.8]
    assert recall_at_k_score_aware(approx_scores, exact_kth_scores, k=2) == 0.75


def test_fewer_results_than_k_penalizes_recall_even_if_all_correct():
    # The bug this test guards against: a method returning only 2 of the
    # requested 10 results, both correct, must NOT score 1.0 — it found 2 of
    # the true top 10, i.e. recall 0.2.
    approx_scores = [[0.95, 0.9]]  # only 2 results returned, both above kth
    exact_kth_scores = [0.5]
    assert recall_at_k_score_aware(approx_scores, exact_kth_scores, k=10) == 0.2


def test_empty_result_list_scores_zero_not_skipped():
    approx_scores = [[], [0.9, 0.8]]
    exact_kth_scores = [0.5, 0.8]
    # query 1: 0/10 = 0.0; query 2: 2/10 = 0.2 -> average 0.1
    assert recall_at_k_score_aware(approx_scores, exact_kth_scores, k=10) == 0.1
