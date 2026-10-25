"""The router half of the CI gate (design record 16e): score the served router on
the 58 Phase 8 labeled dev queries and compare to a committed baseline.

    python -m aegis.eval.router_gate run     --out results/router.json
    python -m aegis.eval.router_gate compare --current results/router.json [--markdown results/router.md]
    python -m aegis.eval.router_gate baseline --current results/router.json

This is the LLM-backed tier: every query is one real call to the router model
(gpt-4.1-mini, ~$0.019 per run). The spend ledger enforces a per-run cap through
OPENAI_BUDGET_USD in CI. Needs no index, so it runs without Qdrant.

billing_account_legal recall is the safety metric (a missed billing/legal question
is a confident wrong answer to a question that must go to a human) and is gated
with zero tolerance. The classifier maps unparseable output to billing as a
fail-safe, which inflates that recall, so billing *precision* is scored too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from aegis.config import settings
from aegis.eval.gate import GateReport, MetricSpec, compare_files, write_baseline
from aegis.router.classifier import Route, RouterClassifier
from aegis.types import GoldenQuery, NegativeCategory, Stratum

BASELINE_NAME = "router.json"
_STRATUM_TO_ROUTE = {Stratum.HOW_TO: Route.DOCS_HOWTO, Stratum.ERROR_DEBUGGING: Route.KNOWN_ISSUE}
_MAX_WORKERS = 8


def load_labeled_queries() -> list[tuple[GoldenQuery, Route]]:
    """The Phase 8 label set (experiments/08a_router_eval.py): how_to -> docs_howto,
    error_debugging -> known_issue, dev billing_legal negatives -> billing."""
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        rows = [GoldenQuery(**json.loads(line)) for line in f]
    labeled: list[tuple[GoldenQuery, Route]] = []
    for q in rows:
        if not q.is_negative and q.stratum in _STRATUM_TO_ROUTE:
            labeled.append((q, _STRATUM_TO_ROUTE[q.stratum]))
        elif q.is_negative and q.negative_category == NegativeCategory.BILLING_LEGAL:
            labeled.append((q, Route.BILLING_ACCOUNT_LEGAL))
    return labeled


def query_set_fingerprint(labeled: Sequence[tuple[GoldenQuery, Route]]) -> str:
    """Identifies the *question set and its labels*, not the router under test: a
    prompt or model change must move the metrics, not the fingerprint."""
    h = hashlib.sha256()
    for q, route in sorted(labeled, key=lambda pair: pair[0].id):
        h.update(f"{q.id}\x00{q.text}\x00{route.value}\x00".encode())
    return h.hexdigest()[:16]


def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def score_routes(pairs: Sequence[tuple[Route, Route]]) -> dict[str, float]:
    """`pairs` = (true route, predicted route). An undefined metric is NaN, which the
    gate treats as a failure -- no support is not the same as a perfect score."""
    def correct(route: Route) -> int:
        return sum(1 for t, p in pairs if t == route and p == route)

    def true_count(route: Route) -> int:
        return sum(1 for t, _ in pairs if t == route)

    billing_predicted = sum(1 for _, p in pairs if p == Route.BILLING_ACCOUNT_LEGAL)
    return {
        "accuracy": _ratio(sum(1 for t, p in pairs if t == p), len(pairs)),
        "billing_recall": _ratio(correct(Route.BILLING_ACCOUNT_LEGAL), true_count(Route.BILLING_ACCOUNT_LEGAL)),
        "billing_precision": _ratio(correct(Route.BILLING_ACCOUNT_LEGAL), billing_predicted),
        "docs_howto_recall": _ratio(correct(Route.DOCS_HOWTO), true_count(Route.DOCS_HOWTO)),
        "known_issue_recall": _ratio(correct(Route.KNOWN_ISSUE), true_count(Route.KNOWN_ISSUE)),
    }


def gate_specs() -> list[MetricSpec]:
    g = settings.gate
    return [
        MetricSpec(name="billing_recall", tolerance=g.router_billing_recall_tolerance),
        MetricSpec(name="billing_precision", tolerance=g.router_billing_precision_tolerance),
        MetricSpec(name="accuracy", tolerance=g.router_accuracy_tolerance),
        MetricSpec(name="router_p95_ms", tolerance=g.router_p95_tolerance_ms, direction="lower"),
    ]


def run(model: str | None = None) -> GateReport:
    """`model` overrides the served router model for one-off candidate comparisons only;
    the CI gate always runs the served model."""
    labeled = load_labeled_queries()
    classifier = RouterClassifier(model=model)
    # A provider outage raises RouterUnavailableError and fails the job: an outage is
    # not a prediction, and must never be scored as one.
    latencies_ms: list[float] = []

    def timed(pair: tuple[GoldenQuery, Route]) -> Route:
        start = time.perf_counter()
        route = classifier.classify(pair[0].text)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        return route

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        predicted = list(pool.map(timed, labeled))
    metrics = score_routes([(true, pred) for (_, true), pred in zip(labeled, predicted)])
    # The router sits on every query's critical path. Per-call API latency is what CI can measure
    # honestly; end-to-end p95 cannot be (the CPU-bound confidence scorer would measure the runner).
    metrics["router_p95_ms"] = float(np.percentile(latencies_ms, 95))
    return GateReport(suite="router", fingerprint=query_set_fingerprint(labeled), n_queries=len(labeled), metrics=metrics)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--out", type=Path, required=True)
    p_run.add_argument("--model", default=None, help="candidate comparison only; the gate runs the served model")
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--current", type=Path, required=True)
    p_cmp.add_argument("--markdown", type=Path, default=None)
    p_base = sub.add_parser("baseline")
    p_base.add_argument("--current", type=Path, required=True)
    args = parser.parse_args(argv)

    baseline = settings.gate.baseline_dir / BASELINE_NAME
    if args.cmd == "run":
        report = run(args.model)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report.model_dump_json(indent=2))
        print(report.model_dump_json(indent=2))
        return 0
    if args.cmd == "compare":
        return compare_files(args.current, baseline, gate_specs(), f"Router ({settings.openai.router_model}, 58 labeled dev queries)", args.markdown)
    write_baseline(args.current, baseline)
    print(f"baseline written to {baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
