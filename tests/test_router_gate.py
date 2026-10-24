"""The router half of the CI gate (design record 16e). The metric that matters is
billing_account_legal recall: a billing/legal question routed to a technical route
is the Air Canada failure mode, so it is gated with zero tolerance. The classifier
fails safe by mapping unparseable output to billing, which would *inflate* that
recall, so billing precision is scored alongside it to expose the inflation."""

from __future__ import annotations

import math
from collections import Counter

from aegis.eval.router_gate import load_labeled_queries, query_set_fingerprint, score_routes
from aegis.router.classifier import Route

D, K, B = Route.DOCS_HOWTO, Route.KNOWN_ISSUE, Route.BILLING_ACCOUNT_LEGAL


def test_a_perfect_run_scores_one_everywhere():
    m = score_routes([(D, D), (K, K), (B, B)])
    assert m == {
        "accuracy": 1.0, "billing_recall": 1.0, "billing_precision": 1.0,
        "docs_howto_recall": 1.0, "known_issue_recall": 1.0,
    }


def test_billing_recall_counts_missed_billing_questions():
    # two true billing questions, one routed to docs -> recall 0.5
    m = score_routes([(B, B), (B, D), (D, D)])
    assert m["billing_recall"] == 0.5


def test_failsafe_inflation_shows_up_as_lower_billing_precision_not_higher_recall_alone():
    # Every query unparseable -> all predicted billing: recall is a perfect 1.0, precision collapses.
    m = score_routes([(B, B), (D, B), (K, B), (D, B)])
    assert m["billing_recall"] == 1.0
    assert m["billing_precision"] == 0.25


def test_per_route_recall():
    m = score_routes([(D, D), (D, K), (K, K), (K, K)])
    assert m["docs_howto_recall"] == 0.5
    assert m["known_issue_recall"] == 1.0
    assert m["accuracy"] == 0.75


def test_a_metric_with_no_support_is_nan_not_zero_or_one():
    # No true billing rows: recall is undefined. NaN makes the gate fail rather than pass on no information.
    m = score_routes([(D, D), (K, K)])
    assert math.isnan(m["billing_recall"])
    assert math.isnan(m["billing_precision"])  # nothing predicted billing either


def test_the_labeled_set_is_the_phase_8_set():
    labeled = load_labeled_queries()
    assert len(labeled) == 58
    assert Counter(r.value for _, r in labeled) == {"docs_howto": 25, "known_issue": 25, "billing_account_legal": 8}


def test_fingerprint_is_stable_and_sensitive_to_text_and_label():
    labeled = load_labeled_queries()
    base = query_set_fingerprint(labeled)
    assert base == query_set_fingerprint(list(labeled))
    q0, r0 = labeled[0]
    relabeled = [(q0, K if r0 != K else D), *labeled[1:]]
    assert query_set_fingerprint(relabeled) != base
    edited = [(q0.model_copy(update={"text": q0.text + " x"}), r0), *labeled[1:]]
    assert query_set_fingerprint(edited) != base
