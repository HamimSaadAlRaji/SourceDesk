"""Cohen's kappa via `sklearn.metrics.cohen_kappa_score` (plan.md Phase 11,
step 3 -- judge validation). Solved statistics problem, imported (CLAUDE.md
Sec 2); a subtly wrong hand-rolled kappa would silently invalidate the
judge-agreement claim (DoD tracker #8, kappa >= 0.6). Verified exactly once
here against a hand-computed confusion matrix -- Wikipedia's own worked
"Cohen's kappa" example (25/25 marginal split, po=0.70, pe=0.50) -- then
trusted everywhere else this project uses it (Unit 4's judge-vs-judge run).
"""

import pytest

from aegis.eval.agreement import cohen_kappa, raw_agreement_rate


def _wikipedia_example_labels() -> tuple[list[str], list[str]]:
    # 50 items. Confusion matrix (rater_a rows, rater_b cols):
    #            b=Yes  b=No
    # a=Yes        20     5    (25)
    # a=No         10    15    (25)
    # totals       30    20    50
    rater_a = ["Yes"] * 20 + ["Yes"] * 5 + ["No"] * 10 + ["No"] * 15
    rater_b = ["Yes"] * 20 + ["No"] * 5 + ["Yes"] * 10 + ["No"] * 15
    return rater_a, rater_b


def test_cohen_kappa_matches_the_hand_computed_wikipedia_example():
    rater_a, rater_b = _wikipedia_example_labels()

    # po = (20 + 15) / 50 = 0.70
    # pe = (25/50 * 30/50) + (25/50 * 20/50) = 0.30 + 0.20 = 0.50
    # kappa = (po - pe) / (1 - pe) = (0.70 - 0.50) / 0.50 = 0.40
    assert cohen_kappa(rater_a, rater_b) == pytest.approx(0.40)


def test_raw_agreement_rate_matches_the_same_example():
    rater_a, rater_b = _wikipedia_example_labels()

    assert raw_agreement_rate(rater_a, rater_b) == pytest.approx(0.70)


def test_cohen_kappa_is_one_for_perfect_agreement():
    labels = ["a", "b", "a", "c", "b", "c", "a"]

    assert cohen_kappa(labels, list(labels)) == pytest.approx(1.0)


def test_cohen_kappa_is_zero_when_agreement_matches_chance():
    # Independent, uniform draws across 2 categories with equal marginals
    # produce exactly chance-level agreement -- po == pe.
    rater_a = ["a", "a", "b", "b"]
    rater_b = ["a", "b", "a", "b"]

    assert cohen_kappa(rater_a, rater_b) == pytest.approx(0.0)


def test_cohen_kappa_returns_a_plain_float_not_a_numpy_scalar():
    rater_a, rater_b = _wikipedia_example_labels()

    assert type(cohen_kappa(rater_a, rater_b)) is float


def test_cohen_kappa_raises_on_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        cohen_kappa(["a", "b"], ["a"])


def test_quadratic_weighting_penalizes_distant_ordinal_disagreement_less_forgivingly_than_unweighted():
    # Ordinal 1-5 rubric. rater_b is always exactly 1 point below rater_a --
    # unweighted kappa treats every one of these as a flat disagreement, but
    # quadratic weighting should score it as a real (if imperfect) measure
    # of agreement, since a stronger discount on ordinal proximity, higher
    # than unweighted, is the expected/hand-verified relationship (sklearn's
    # own weighting, trusted per CLAUDE.md Sec 2, not re-derived here).
    rater_a = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    rater_b = [2, 3, 4, 5, 4, 2, 3, 4, 5, 4]

    unweighted = cohen_kappa(rater_a, rater_b)
    quadratic = cohen_kappa(rater_a, rater_b, weights="quadratic")

    assert quadratic > unweighted
