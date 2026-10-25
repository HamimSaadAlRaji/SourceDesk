"""The CI gate's comparison logic (plan.md Phase 16, design record 16c): current
metrics against a committed baseline, with a tolerance band per metric.

The rule that matters is what counts as failure. A regression gate that passes
when it has no information is worse than no gate, so everything that is not a
clean comparison fails: a gated metric missing on either side, a NaN/inf value,
a baseline taken on a different corpus (`fingerprint`) or a different query set
(`n_queries`), and a gate with no metric specs at all. Only a clean comparison
inside the band passes.

Tolerances are absolute, not relative: the metrics are bounded fractions, and an
absolute band means the same thing at 0.44 recall as at 0.86 MRR.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Direction = Literal["higher", "lower"]
Status = Literal["pass", "fail", "improved"]

# Float slack so a drop of exactly the tolerance (0.44 -> 0.42 computes as
# 0.020000000000000018) is not failed by representation error.
_EPS = 1e-9


class MetricSpec(BaseModel):
    name: str
    tolerance: float
    direction: Direction = "higher"  # "lower" for latency and cost


class GateReport(BaseModel):
    """What one suite measured. `fingerprint` identifies the corpus and index the
    numbers came from (`ci_index.corpus_fingerprint`); `n_queries` the query set."""

    suite: str
    fingerprint: str
    n_queries: int
    metrics: dict[str, float] = Field(default_factory=dict)
    # Provenance (host CPU, query-vector hash). Never gated and never compared: it exists so a
    # metric that moved can be traced to hardware instead of guessed at.
    info: dict[str, str] = Field(default_factory=dict)


class MetricCheck(BaseModel):
    name: str
    baseline: float | None
    current: float | None
    delta: float | None  # current - baseline, signed as-is
    tolerance: float
    direction: Direction
    status: Status


class GateResult(BaseModel):
    passed: bool
    checks: list[MetricCheck]
    problems: list[str]


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def evaluate(current: GateReport, baseline: GateReport, specs: list[MetricSpec]) -> GateResult:
    problems: list[str] = []
    checks: list[MetricCheck] = []

    if not specs:
        problems.append("no metric specs: a gate with nothing to check cannot pass")
    if current.fingerprint != baseline.fingerprint:
        problems.append(
            f"fingerprint mismatch: current {current.fingerprint} vs baseline {baseline.fingerprint} "
            "(measured on a different corpus or index; rebaseline deliberately, do not compare)"
        )
    if current.n_queries != baseline.n_queries:
        problems.append(f"n_queries mismatch: current {current.n_queries} vs baseline {baseline.n_queries}")

    for spec in specs:
        cur, base = current.metrics.get(spec.name), baseline.metrics.get(spec.name)
        if not _finite(cur) or not _finite(base):
            which = "current run" if not _finite(cur) else "baseline"
            problems.append(f"{spec.name}: missing or non-finite in the {which} (cur={cur}, base={base})")
            checks.append(MetricCheck(
                name=spec.name, baseline=base, current=cur, delta=None,
                tolerance=spec.tolerance, direction=spec.direction, status="fail",
            ))
            continue
        assert cur is not None and base is not None
        delta = cur - base
        worse_by = -delta if spec.direction == "higher" else delta
        if worse_by > spec.tolerance + _EPS:
            status: Status = "fail"
        elif -worse_by > spec.tolerance + _EPS:
            status = "improved"
        else:
            status = "pass"
        checks.append(MetricCheck(
            name=spec.name, baseline=base, current=cur, delta=delta,
            tolerance=spec.tolerance, direction=spec.direction, status=status,
        ))

    passed = not problems and bool(checks) and all(c.status != "fail" for c in checks)
    return GateResult(passed=passed, checks=checks, problems=problems)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}" if abs(value) >= 100 else f"{value:.4f}"


_STATUS_LABEL = {"pass": "✅ pass", "fail": "❌ FAIL", "improved": "⬆️ improved"}


def render_markdown(title: str, result: GateResult) -> str:
    """The table a PR comment shows (plan.md Phase 16 step 5)."""
    verdict = "✅ passed" if result.passed else "❌ FAILED"
    lines = [f"### {title}: {verdict}", "", "| Metric | Baseline | Current | Δ | Band | Status |", "|---|---|---|---|---|---|"]
    for c in result.checks:
        delta = "n/a" if c.delta is None else (f"{c.delta:+.1f}" if abs(c.delta) >= 100 else f"{c.delta:+.4f}")
        band = f"±{_fmt(c.tolerance)} ({'higher' if c.direction == 'higher' else 'lower'} is better)"
        lines.append(f"| {c.name} | {_fmt(c.baseline)} | {_fmt(c.current)} | {delta} | {band} | {_STATUS_LABEL[c.status]} |")
    if result.problems:
        lines += ["", "**Problems:**", *[f"- {p}" for p in result.problems]]
    return "\n".join(lines) + "\n"


def compare_files(
    current_path: Path, baseline_path: Path, specs: list[MetricSpec], title: str, markdown_path: Path | None = None
) -> int:
    """Shared CLI tail for every suite: 0 = gate passed, 1 = failed. A missing
    baseline is a failure -- there is nothing to compare against, so no information."""
    current = GateReport.model_validate_json(current_path.read_text())
    if not baseline_path.exists():
        print(f"no baseline at {baseline_path}: a gate with nothing to compare against cannot pass", file=sys.stderr)
        return 1
    baseline = GateReport.model_validate_json(baseline_path.read_text())
    result = evaluate(current, baseline, specs)
    md = render_markdown(title, result)
    if current.info:
        md += "\n<sub>" + " · ".join(f"{k}: {v}" for k, v in current.info.items()) + "</sub>\n"
    print(md)
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(md)
    return 0 if result.passed else 1


def write_baseline(current_path: Path, baseline_path: Path) -> None:
    """Deliberate rebaseline: validates the report, then overwrites. A reviewable diff in the PR."""
    report = GateReport.model_validate_json(current_path.read_text())
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(report.model_dump_json(indent=2) + "\n")
