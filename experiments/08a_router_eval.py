"""Phase 8 router evaluation (plan.md Phase 8, step 3): "label ~60 golden
queries with correct routes; report a routing confusion matrix."

Ground-truth route labels reuse `strata.py`'s existing keyword-based
classification rather than inventing a fresh labeling pass (decision
confirmed with the user 2026-08-12):

    how_to (25 dev positives)          -> docs_howto
    error_debugging (25 dev positives) -> known_issue
    negative/billing_legal (dev split) -> billing_account_legal
    = ~60 labeled queries (58 measured: only 8 of the 10 billing_legal
      negatives are in the dev split -- the other 2 are sealed in the
      Phase 4 holdout split, same dev-only-until-Phase-19 convention every
      other phase follows)

strata.py's how_to/error_debugging cues are about *how the query is
phrased* ("how do I ..." vs "... raises/crashes/fails"), which is exactly
the router's own distinction -- not a repurposed signal, the same signal.
conceptual/version_specific/multi_hop/ambiguous strata are excluded: they
were checked (see docs/results/ ledger in progress.md) and don't separate
cleanly into a single route (~50-58% doc-chunk ground truth in every
stratum, a side effect of Phase 4's broad auto-labeling), so forcing them
into docs_howto/known_issue would just add label noise, not real coverage.
"""

from __future__ import annotations

import json
from collections import Counter

from aegis.config import REPO_ROOT, settings
from aegis.router.classifier import Route, RouterClassifier
from aegis.types import GoldenQuery, NegativeCategory, Stratum

RESULTS_MD = REPO_ROOT / "docs" / "results" / "routing-confusion-matrix.md"
RESULTS_RAW = REPO_ROOT / "docs" / "results" / "routing-eval-raw.json"

_STRATUM_TO_ROUTE = {
    Stratum.HOW_TO: Route.DOCS_HOWTO,
    Stratum.ERROR_DEBUGGING: Route.KNOWN_ISSUE,
}


def load_labeled_queries() -> list[tuple[GoldenQuery, Route]]:
    path = settings.paths.data_golden / "golden_set_v1.jsonl"
    rows = [GoldenQuery(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]

    labeled = []
    for q in rows:
        if not q.is_negative and q.stratum in _STRATUM_TO_ROUTE:
            labeled.append((q, _STRATUM_TO_ROUTE[q.stratum]))
        elif q.is_negative and q.negative_category == NegativeCategory.BILLING_LEGAL:
            labeled.append((q, Route.BILLING_ACCOUNT_LEGAL))
    return labeled


def main() -> None:
    labeled = load_labeled_queries()
    by_route = Counter(route.value for _, route in labeled)
    print(f"{len(labeled)} labeled queries: {dict(by_route)}")

    classifier = RouterClassifier()

    routes = list(Route)
    confusion: dict[str, Counter] = {r.value: Counter() for r in routes}
    rows_out = []
    for q, true_route in labeled:
        predicted = classifier.classify(q.text)
        confusion[true_route.value][predicted.value] += 1
        rows_out.append({"query_id": q.id, "true_route": true_route.value, "predicted_route": predicted.value})
        print(f"  {q.id:35s} true={true_route.value:24s} pred={predicted.value}")

    n = len(labeled)
    correct = sum(confusion[r.value][r.value] for r in routes)
    accuracy = correct / n

    per_class = {}
    for r in routes:
        tp = confusion[r.value][r.value]
        fn = sum(confusion[r.value][other.value] for other in routes if other != r)
        fp = sum(confusion[other.value][r.value] for other in routes if other != r)
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        per_class[r.value] = {"tp": tp, "fn": fn, "fp": fp, "precision": precision, "recall": recall}

    print(f"\nOverall accuracy: {accuracy:.4f} ({correct}/{n})")
    for r in routes:
        pc = per_class[r.value]
        print(f"  {r.value:24s} precision={pc['precision']:.4f} recall={pc['recall']:.4f} (tp={pc['tp']} fn={pc['fn']} fp={pc['fp']})")

    RESULTS_RAW.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_RAW.write_text(
        json.dumps(
            {
                "n_queries": n,
                "accuracy": accuracy,
                "confusion": {k: dict(v) for k, v in confusion.items()},
                "per_class": per_class,
                "rows": rows_out,
            },
            indent=2,
        )
    )
    print(f"Wrote {RESULTS_RAW}")

    _write_markdown(labeled, confusion, per_class, accuracy, correct, n, routes)
    print(f"Wrote {RESULTS_MD}")


def _write_markdown(labeled, confusion, per_class, accuracy, correct, n, routes) -> None:
    lines = [
        "# Routing confusion matrix",
        "",
        f"**Phase:** 8 · **Model:** `{settings.router.model}` (few-shot, JSON-mode, temperature={settings.router.temperature})",
        "",
        f"{n} labeled queries: 25 `how_to` dev positives -> `docs_howto`, 25 `error_debugging` dev positives -> "
        f"`known_issue`, {sum(confusion['billing_account_legal'].values())} `billing_legal` negatives (of 10 total -- "
        "the other 2 are sealed in the Phase 4 holdout split) -> `billing_account_legal`. See the experiment "
        "script's docstring for the labeling-methodology decision.",
        "",
        f"## Overall accuracy: {accuracy:.4f} ({correct}/{n})",
        "",
        "## Confusion matrix (rows = true route, columns = predicted route)",
        "",
        "| true \\ predicted | " + " | ".join(r.value for r in routes) + " |",
        "|---" * (len(routes) + 1) + "|",
    ]
    for r in routes:
        row = [str(confusion[r.value][c.value]) for c in routes]
        lines.append(f"| **{r.value}** | " + " | ".join(row) + " |")

    lines += ["", "## Per-class precision / recall", "", "| Route | Precision | Recall | TP | FN | FP |", "|---|---|---|---|---|---|"]
    for r in routes:
        pc = per_class[r.value]
        lines.append(f"| {r.value} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['tp']} | {pc['fn']} | {pc['fp']} |")

    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
