"""The CI gate comparison (design record 16c). Every failure mode here is a way a
gate can pass when it should not: a metric that is missing, NaN, or measured on a
different corpus or query set is not "no regression", it is "no information", and
must fail. The tolerance boundary and the direction of each metric are pinned too."""

from __future__ import annotations

import math

from aegis.eval.gate import GateReport, MetricSpec, compare_files, evaluate, render_markdown

SPECS = [
    MetricSpec(name="recall@10", tolerance=0.02),
    MetricSpec(name="mrr", tolerance=0.02),
    MetricSpec(name="p95_ms", tolerance=300.0, direction="lower"),
]


def _report(fingerprint: str = "fp1", n: int = 150, **metrics: float) -> GateReport:
    base = {"recall@10": 0.44, "mrr": 0.86, "p95_ms": 1000.0}
    base.update(metrics)
    return GateReport(suite="retrieval", fingerprint=fingerprint, n_queries=n, metrics=base)


def _status(result, name: str) -> str:
    return next(c.status for c in result.checks if c.name == name)


def test_identical_reports_pass():
    result = evaluate(_report(), _report(), SPECS)
    assert result.passed
    assert not result.problems
    assert {c.status for c in result.checks} == {"pass"}


def test_drop_inside_the_band_passes_and_beyond_it_fails():
    assert evaluate(_report(**{"recall@10": 0.425}), _report(), SPECS).passed
    result = evaluate(_report(**{"recall@10": 0.41}), _report(), SPECS)
    assert not result.passed
    assert _status(result, "recall@10") == "fail"


def test_a_drop_of_exactly_the_tolerance_passes():
    # Boundary is inclusive: tolerance is the largest drop we call noise.
    # Values chosen to be exact in binary floating point.
    specs = [MetricSpec(name="recall@10", tolerance=0.25)]
    baseline = GateReport(suite="s", fingerprint="f", n_queries=1, metrics={"recall@10": 0.75})
    current = GateReport(suite="s", fingerprint="f", n_queries=1, metrics={"recall@10": 0.5})
    assert evaluate(current, baseline, specs).passed


def test_improvement_beyond_the_band_passes_but_is_flagged():
    result = evaluate(_report(**{"recall@10": 0.50}), _report(), SPECS)
    assert result.passed
    assert _status(result, "recall@10") == "improved"


def test_lower_is_better_metrics_fail_when_they_rise():
    result = evaluate(_report(p95_ms=1400.0), _report(), SPECS)
    assert not result.passed
    assert _status(result, "p95_ms") == "fail"
    # ...and dropping latency is an improvement, not a failure.
    faster = evaluate(_report(p95_ms=500.0), _report(), SPECS)
    assert faster.passed
    assert _status(faster, "p95_ms") == "improved"


def test_a_metric_missing_from_the_current_run_fails():
    current = _report()
    del current.metrics["mrr"]
    result = evaluate(current, _report(), SPECS)
    assert not result.passed
    assert any("mrr" in p for p in result.problems)


def test_a_metric_missing_from_the_baseline_fails():
    baseline = _report()
    del baseline.metrics["mrr"]
    result = evaluate(_report(), baseline, SPECS)
    assert not result.passed
    assert any("mrr" in p for p in result.problems)


def test_nan_and_infinite_current_values_fail():
    for bad in (math.nan, math.inf, -math.inf):
        result = evaluate(_report(**{"recall@10": bad}), _report(), SPECS)
        assert not result.passed, bad


def test_a_different_corpus_fingerprint_fails():
    result = evaluate(_report(fingerprint="other"), _report(), SPECS)
    assert not result.passed
    assert any("fingerprint" in p for p in result.problems)


def test_a_different_query_count_fails():
    result = evaluate(_report(n=149), _report(), SPECS)
    assert not result.passed
    assert any("n_queries" in p for p in result.problems)


def test_metrics_without_a_spec_are_ignored_not_gated():
    current = _report()
    current.metrics["map@10"] = 0.0
    assert evaluate(current, _report(), SPECS).passed


def test_a_gate_with_no_specs_refuses_to_pass():
    # Zero checks would "pass" vacuously.
    assert not evaluate(_report(), _report(), []).passed


def test_markdown_shows_each_metric_its_delta_and_the_problems():
    result = evaluate(_report(fingerprint="other", **{"recall@10": 0.30}), _report(), SPECS)
    md = render_markdown("Retrieval", result)
    assert "recall@10" in md and "0.4400" in md and "0.3000" in md
    assert "-0.1400" in md
    assert "FAIL" in md
    assert "fingerprint" in md


# --- run provenance (2026-10-24): the hosted runner returned two different numeric
# outcomes on identical inputs (0.4435 vs the M1's bit-identical 0.4317 recall@10) and
# nothing recorded which hardware produced which. `info` carries that provenance. It is
# metadata: it must round-trip, be shown, and never influence the verdict. ---


def test_info_round_trips_and_never_changes_the_verdict():
    a = _report()
    a.info = {"cpu_model": "AMD EPYC", "query_vectors_sha256": "abc"}
    b = _report()
    b.info = {"cpu_model": "Intel Xeon", "query_vectors_sha256": "def"}
    assert evaluate(a, b, SPECS).passed
    assert GateReport.model_validate_json(a.model_dump_json()).info == a.info


def test_a_report_without_info_still_loads():
    # baselines written before `info` existed must keep working
    raw = '{"suite": "retrieval", "fingerprint": "f", "n_queries": 1, "metrics": {"recall@10": 0.5}}'
    assert GateReport.model_validate_json(raw).info == {}


def test_compare_files_shows_the_host_of_the_current_run(tmp_path):
    base = _report()
    cur = _report()
    cur.info = {"cpu_model": "AMD EPYC 7763", "query_vectors_sha256": "deadbeef"}
    (tmp_path / "base.json").write_text(base.model_dump_json())
    (tmp_path / "cur.json").write_text(cur.model_dump_json())
    md_path = tmp_path / "out.md"
    assert compare_files(tmp_path / "cur.json", tmp_path / "base.json", SPECS, "T", md_path) == 0
    md = md_path.read_text()
    assert "AMD EPYC 7763" in md and "deadbeef" in md
