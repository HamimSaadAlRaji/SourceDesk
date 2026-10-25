"""The escalation half of the CI gate (design record 16i). `air_canada_fn` -- a negative
(should-escalate) query that was answered anyway -- is the project's north-star failure,
so it is a count gated with zero tolerance, not a rate with a band. Over-caution is the
other side of the same dial and is a band on deflection."""

from __future__ import annotations

import math

from aegis.eval.escalation_gate import score_escalation


def test_perfect_decisions():
    m = score_escalation(escalated=[True, True, False, False], is_negative=[True, True, False, False])
    assert m == {"air_canada_fn": 0.0, "negative_recall": 1.0, "deflection_rate": 1.0}


def test_an_answered_negative_is_counted_as_an_air_canada_failure():
    m = score_escalation(escalated=[True, False, False], is_negative=[True, True, False])
    assert m["air_canada_fn"] == 1.0
    assert m["negative_recall"] == 0.5


def test_over_caution_lowers_deflection_but_not_recall():
    m = score_escalation(escalated=[True, True, True], is_negative=[True, False, False])
    assert m["negative_recall"] == 1.0
    assert m["deflection_rate"] == 0.0
    assert m["air_canada_fn"] == 0.0


def test_no_negatives_means_undefined_recall_not_a_pass():
    m = score_escalation(escalated=[False, False], is_negative=[False, False])
    assert math.isnan(m["negative_recall"])
    assert m["air_canada_fn"] == 0.0  # a count, honestly zero


def test_no_positives_means_undefined_deflection():
    assert math.isnan(score_escalation(escalated=[True], is_negative=[True])["deflection_rate"])
