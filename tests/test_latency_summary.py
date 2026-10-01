"""TDD for per-stage latency aggregation (Phase 13 Unit 3). Every Phase 13
latency claim -- and the Phase 16 CI p95 gate after it -- flows through
this arithmetic, so it's checked against hand-computed values."""

import pytest

from aegis.observability.latency import StageStats, stage_latency_summary


def test_percentiles_match_hand_computed_linear_interpolation():
    # 1..100 ms: numpy-style linear interpolation -> p50=50.5, p95=95.05, p99=99.01
    rows = [{"retrieval": float(ms)} for ms in range(1, 101)]

    stats = stage_latency_summary(rows)["retrieval"]

    assert stats.n == 100
    assert stats.p50 == pytest.approx(50.5)
    assert stats.p95 == pytest.approx(95.05)
    assert stats.p99 == pytest.approx(99.01)
    assert stats.mean == pytest.approx(50.5)
    assert stats.max == pytest.approx(100.0)


def test_stages_absent_from_some_requests_count_only_requests_that_ran_them():
    rows = [
        {"routing": 10.0, "generation": 1000.0},
        {"routing": 20.0},  # escalated: never generated
        {"routing": 30.0},
    ]

    summary = stage_latency_summary(rows)

    assert summary["routing"].n == 3
    assert summary["generation"].n == 1
    assert summary["generation"].p95 == pytest.approx(1000.0)


def test_single_sample_has_all_percentiles_equal_to_it():
    stats = stage_latency_summary([{"confidence": 42.0}])["confidence"]

    assert stats == StageStats(n=1, mean=42.0, p50=42.0, p95=42.0, p99=42.0, max=42.0)


def test_empty_input_yields_empty_summary():
    assert stage_latency_summary([]) == {}
