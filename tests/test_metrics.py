"""Retrieval metrics via `ranx` (plan.md Phase 6, step 1).

`ranx` itself is imported -- Precision/Recall/MRR/MAP/NDCG are solved
problems, and a subtly wrong hand-rolled NDCG would mislead for weeks
before anyone noticed (CLAUDE.md §2). It's verified exactly once, here,
against a hand-computed 5-document example, then trusted everywhere else
this project uses it (the eval harness, Phase 6's ablation grid).
"""

import math

import pytest

from aegis.eval.metrics import build_qrels, build_run, compute_metrics


def test_ranx_matches_a_hand_computed_5_document_example():
    # Query q1, 5 documents. Relevant: d1, d3, d5 (ranks 1, 3, 5 in the
    # run below). Run, descending score: d1, d2, d3, d4, d5.
    qrels = build_qrels({"q1": ["d1", "d3", "d5"]})
    run = build_run({"q1": [("d1", 5.0), ("d2", 4.0), ("d3", 3.0), ("d4", 2.0), ("d5", 1.0)]})

    results = compute_metrics(qrels, run, ["precision@5", "recall@5", "mrr", "map@5", "ndcg@5"])

    # precision@5 = (# relevant in top 5) / 5 = 3/5
    assert results["precision@5"] == pytest.approx(3 / 5)
    # recall@5 = (# relevant in top 5) / (# relevant total) = 3/3
    assert results["recall@5"] == pytest.approx(1.0)
    # MRR = 1 / rank of first relevant hit = 1/1
    assert results["mrr"] == pytest.approx(1.0)

    # AP = mean of precision@k at each rank a relevant doc appears
    # (rank 1: 1/1, rank 3: 2/3, rank 5: 3/5), divided by # relevant.
    hand_ap = ((1 / 1) + (2 / 3) + (3 / 5)) / 3
    assert results["map@5"] == pytest.approx(hand_ap)

    # DCG with binary relevance [1,0,1,0,1] at ranks 1-5; IDCG is the
    # same 3 relevant docs pulled to ranks 1-3: [1,1,1,0,0].
    dcg = 1 / math.log2(2) + 0 / math.log2(3) + 1 / math.log2(4) + 0 / math.log2(5) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert results["ndcg@5"] == pytest.approx(dcg / idcg)


def test_build_qrels_marks_ground_truth_chunks_as_relevant():
    qrels = build_qrels({"q1": ["c1", "c2"], "q2": ["c3"]})

    assert qrels.qrels == {"q1": {"c1": 1, "c2": 1}, "q2": {"c3": 1}}


def test_build_run_carries_retrieval_scores():
    run = build_run({"q1": [("c1", 0.9), ("c2", 0.5)]})

    assert run.run == {"q1": {"c1": 0.9, "c2": 0.5}}


def test_compute_metrics_returns_plain_floats_not_numpy_scalars():
    qrels = build_qrels({"q1": ["c1"]})
    run = build_run({"q1": [("c1", 1.0)]})

    results = compute_metrics(qrels, run, ["recall@10"])

    assert type(results["recall@10"]) is float
