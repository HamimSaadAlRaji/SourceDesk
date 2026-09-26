"""Inter-rater agreement via `sklearn.metrics.cohen_kappa_score` (plan.md
Phase 11, step 3 -- judge validation). A solved statistics problem, imported
rather than hand-rolled (CLAUDE.md Sec 2): the formula itself is simple, but
a subtly wrong implementation would silently invalidate every downstream
"the judge is reliable" claim, and sklearn's version is the one every reader
can independently check against. Verified once against a hand-computed
confusion matrix (`tests/test_agreement.py`), same convention as
`eval/metrics.py`'s `ranx` wrapper.

`weights` (None / "linear" / "quadratic") matters for Unit 3's helpfulness
rubric if it ends up ordinal (e.g. a 1-5 Likert scale): unweighted kappa
treats a 1-vs-5 disagreement identically to a 4-vs-5 disagreement, which
overstates how badly two judges actually disagree on an ordinal scale.
Exposed here, not decided here -- Unit 3 picks the rubric shape and which
weighting fits it.

Judge-vs-judge, not judge-vs-hand-label (2026-09-25 decision, progress.md
Decision Log -- the user declined to hand-label; see design record 11+ for
the honest cost of this substitution).
"""

from __future__ import annotations

from sklearn.metrics import cohen_kappa_score


def cohen_kappa(labels_a: list, labels_b: list, weights: str | None = None) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError(f"labels_a and labels_b must have the same length, got {len(labels_a)} and {len(labels_b)}")
    return float(cohen_kappa_score(labels_a, labels_b, weights=weights))


def raw_agreement_rate(labels_a: list, labels_b: list) -> float:
    """Fraction of items where both raters gave the identical label --
    kappa's numerator before subtracting chance agreement. Reported
    alongside kappa (not a substitute for it): a high raw rate can still be
    mostly chance if one category dominates both raters' label distribution,
    which is exactly what kappa corrects for."""
    if len(labels_a) != len(labels_b):
        raise ValueError(f"labels_a and labels_b must have the same length, got {len(labels_a)} and {len(labels_b)}")
    if not labels_a:
        return float("nan")
    agreements = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b)
    return agreements / len(labels_a)
