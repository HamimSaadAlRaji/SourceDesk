"""TDD for Phase 11 Unit 5's custom metrics -- citation validity
aggregation, escalation appropriateness, code-block static syntax validity.
All three are deterministic and code-based (zero LLM cost); a silent bug
here would misreport exactly the numbers plan.md's Definition of Done
tracker and `make eval`'s scorecard are built to be trusted for, same
category as the citation validator and confidence scorer CLAUDE.md already
names for TDD.
"""

import pytest

from aegis.eval.custom_metrics import (
    citation_validity_summary,
    code_block_syntax_validity,
    escalation_appropriateness,
)
from aegis.guardrails.citations import ClaimResult, ClaimStatus, ValidationResult


def _validation(statuses: list[ClaimStatus]) -> ValidationResult:
    claims = [ClaimResult(sentence="s", citation_ids=[1], status=s) for s in statuses]
    return ValidationResult(claims=claims, tier2_calls=0)


# --- citation_validity_summary ---


def test_citation_validity_summary_averages_valid_citation_rate_across_answers():
    # Answer 1: 2/2 supported (rate 1.0). Answer 2: 1/2 supported (rate 0.5).
    results = [
        _validation([ClaimStatus.SUPPORTED_LEXICAL, ClaimStatus.SUPPORTED_ENTAILMENT]),
        _validation([ClaimStatus.SUPPORTED_LEXICAL, ClaimStatus.UNSUPPORTED]),
    ]

    summary = citation_validity_summary(results)

    assert summary.n_answers == 2
    assert summary.mean_valid_citation_rate == pytest.approx(0.75)


def test_citation_validity_summary_counts_claim_statuses_across_the_whole_batch():
    results = [
        _validation([ClaimStatus.SUPPORTED_LEXICAL, ClaimStatus.MISSING_CITATION]),
        _validation([ClaimStatus.SUPPORTED_LEXICAL, ClaimStatus.EXEMPT]),
    ]

    summary = citation_validity_summary(results)

    assert summary.status_counts[ClaimStatus.SUPPORTED_LEXICAL] == 2
    assert summary.status_counts[ClaimStatus.MISSING_CITATION] == 1
    assert summary.status_counts[ClaimStatus.EXEMPT] == 1
    assert summary.status_counts.get(ClaimStatus.UNSUPPORTED, 0) == 0


def test_citation_validity_summary_handles_empty_batch():
    summary = citation_validity_summary([])

    assert summary.n_answers == 0
    assert summary.mean_valid_citation_rate == 0.0


# --- escalation_appropriateness ---


def test_escalation_appropriateness_matches_the_hand_computed_confusion_matrix():
    # 4 items: escalated=[True, True, False, False], is_negative=[True, False, False, True]
    # -> TP=1 (escalated on negative), FP=1 (escalated on positive),
    #    FN=1 (didn't escalate on negative), TN=1 (didn't escalate on positive)
    result = escalation_appropriateness(
        escalated=[True, True, False, False],
        is_negative=[True, False, False, True],
    )

    assert (result.tp, result.fp, result.fn, result.tn) == (1, 1, 1, 1)
    assert result.precision == pytest.approx(0.5)  # tp / (tp + fp)
    assert result.recall == pytest.approx(0.5)  # tp / (tp + fn)
    assert result.deflection_rate == pytest.approx(0.5)  # tn / (tn + fp)


def test_escalation_appropriateness_matches_phase_10s_real_chosen_tau_numbers():
    # Reproduces docs/results/confusion-matrix.md's real chosen-tau result
    # (TP=27 FP=24 FN=3 TN=126) from raw escalate/is_negative labels, as a
    # regression guard that this reimplementation agrees with the number
    # already measured and shipped in Phase 10.
    escalated = [True] * (27 + 24) + [False] * (3 + 126)
    is_negative = [True] * 27 + [False] * 24 + [True] * 3 + [False] * 126

    result = escalation_appropriateness(escalated=escalated, is_negative=is_negative)

    assert (result.tp, result.fp, result.fn, result.tn) == (27, 24, 3, 126)
    assert result.precision == pytest.approx(27 / 51)
    assert result.recall == pytest.approx(0.90)
    assert result.deflection_rate == pytest.approx(0.84)


def test_escalation_appropriateness_raises_on_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        escalation_appropriateness(escalated=[True], is_negative=[True, False])


def test_escalation_appropriateness_precision_is_nan_when_nothing_escalated():
    result = escalation_appropriateness(escalated=[False, False], is_negative=[True, False])

    assert result.tp == 0 and result.fp == 0
    assert result.precision != result.precision  # NaN != NaN


# --- code_block_syntax_validity ---


def test_code_block_syntax_validity_accepts_valid_python():
    answer = "Here's how:\n\n```python\ndef f(x):\n    return x + 1\n```\n"

    result = code_block_syntax_validity(answer)

    assert result.n_checked == 1
    assert result.n_valid == 1
    assert result.valid_rate == pytest.approx(1.0)


def test_code_block_syntax_validity_rejects_invalid_python():
    answer = "Try this:\n\n```python\ndef f(x)\n    return x + 1\n```\n"  # missing colon

    result = code_block_syntax_validity(answer)

    assert result.n_checked == 1
    assert result.n_valid == 0
    assert result.valid_rate == pytest.approx(0.0)
    assert len(result.invalid_blocks) == 1


def test_code_block_syntax_validity_checks_untagged_blocks():
    answer = "```\nx = 1\ny = 2\n```"

    result = code_block_syntax_validity(answer)

    assert result.n_checked == 1
    assert result.n_valid == 1


def test_code_block_syntax_validity_skips_explicitly_non_python_blocks():
    answer = "```bash\npip install fastapi\n```\n\n```json\n{not valid json or python}\n```"

    result = code_block_syntax_validity(answer)

    assert result.n_checked == 0
    assert result.n_skipped_non_python == 2


def test_code_block_syntax_validity_handles_no_code_blocks():
    # Vacuous truth, same convention as `ValidationResult.valid_citation_rate`
    # for an all-EXEMPT answer (guardrails/citations.py) -- nothing to be
    # wrong about, so this doesn't get penalized as 0.
    result = code_block_syntax_validity("Just prose, no code here.")

    assert result.n_checked == 0
    assert result.n_valid == 0
    assert result.valid_rate == 1.0


def test_code_block_syntax_validity_handles_multiple_mixed_blocks():
    answer = (
        "```python\nx = 1\n```\n\n"
        "```python\ndef broken(\n```\n\n"
        "```bash\necho hi\n```"
    )

    result = code_block_syntax_validity(answer)

    assert result.n_checked == 2
    assert result.n_valid == 1
    assert result.n_skipped_non_python == 1
