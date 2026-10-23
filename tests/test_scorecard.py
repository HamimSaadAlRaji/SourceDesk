"""TDD for Phase 11 Unit 6 -- the `make eval` scorecard. Pure aggregation
over already-measured data (Unit 4's batch rows), zero LLM cost, but a
silent bug here would misreport exactly the numbers the Definition of Done
tracker is built to be trusted for -- same risk class CLAUDE.md names the
citation validator and confidence scorer for, now the top-level consumer
of both.
"""


import pytest

from aegis.config import REPO_ROOT, settings
from aegis.eval.scorecard import (
    build_dod_checks,
    build_scorecard,
    load_rows,
    summarize_citation_validity,
    summarize_code_blocks,
    summarize_judge_agreement,
    summarize_ragas,
)


def _row(query_id: str, *, faithfulness=None, groq_score=None, gemini_score=None, citation_rate=0.5, answer_text="prose only") -> dict:
    row: dict = {
        "query_id": query_id,
        "generation": {"valid_citation_rate": citation_rate, "answer_text": answer_text},
        "ragas": {},
    }
    if faithfulness is not None:
        row["ragas"]["faithfulness"] = {"score": faithfulness}
    if groq_score is not None:
        row["judge_groq"] = {"score": groq_score}
    if gemini_score is not None:
        row["judge_gemini"] = {"score": gemini_score}
    return row


def _errored_row(query_id: str) -> dict:
    return {"query_id": query_id, "generation": {"error": "boom"}}


# --- summarize_ragas ---


def test_summarize_ragas_computes_mean_over_successful_scores_only():
    rows = [_row("a", faithfulness=0.8), _row("b", faithfulness=0.6), _row("c")]

    summary = summarize_ragas(rows)

    assert summary["faithfulness"].n == 2
    assert summary["faithfulness"].mean == pytest.approx(0.7)


def test_summarize_ragas_reports_zero_n_and_none_mean_when_all_missing():
    rows = [_row("a"), _row("b")]

    summary = summarize_ragas(rows)

    assert summary["faithfulness"].n == 0
    assert summary["faithfulness"].mean is None


def test_summarize_ragas_excludes_generation_errors():
    rows = [_row("a", faithfulness=0.9), _errored_row("b")]
    # An errored generation row wouldn't carry a "ragas" key at all in the
    # real batch script -- confirm that shape doesn't blow up the summary.

    summary = summarize_ragas(rows)

    assert summary["faithfulness"].n == 1
    assert summary["faithfulness"].mean == pytest.approx(0.9)


# --- summarize_judge_agreement ---


def test_summarize_judge_agreement_computes_kappa_over_paired_rows_only():
    rows = [
        _row("a", groq_score=5, gemini_score=5),
        _row("b", groq_score=1, gemini_score=1),
        _row("c", groq_score=5, gemini_score=5),
        _row("d", groq_score=1, gemini_score=1),
        _row("e", groq_score=3),  # gemini missing -- unpaired, excluded
    ]

    summary = summarize_judge_agreement(rows)

    assert summary is not None
    assert summary.n_paired == 4
    assert summary.n_a == 5  # groq succeeded on all 5
    assert summary.n_b == 4  # gemini succeeded on 4
    assert summary.cohen_kappa == pytest.approx(1.0)  # perfect agreement on the paired subset
    assert summary.raw_agreement_rate == pytest.approx(1.0)


def test_summarize_judge_agreement_returns_none_when_fewer_than_two_paired():
    rows = [_row("a", groq_score=5, gemini_score=5), _row("b", groq_score=3)]

    summary = summarize_judge_agreement(rows)

    assert summary is not None
    assert summary.n_paired == 1
    assert summary.cohen_kappa is None
    assert summary.raw_agreement_rate is None


def test_summarize_judge_agreement_returns_none_object_when_no_rows_have_either_judge():
    rows = [_row("a")]

    summary = summarize_judge_agreement(rows)

    assert summary is None


# --- summarize_code_blocks ---


def test_summarize_code_blocks_aggregates_across_generation_ok_rows_only():
    rows = [
        _row("a", answer_text="```python\nx = 1\n```"),
        _row("b", answer_text="```python\ndef broken(\n```"),
        _errored_row("c"),
    ]

    result = summarize_code_blocks(rows)

    assert result.n_checked == 2
    assert result.n_valid == 1
    assert len(result.invalid_blocks) == 1


def test_summarize_code_blocks_handles_no_code_anywhere():
    rows = [_row("a", answer_text="just prose")]

    result = summarize_code_blocks(rows)

    assert result.n_checked == 0
    assert result.valid_rate == 1.0


# --- summarize_citation_validity ---


def test_summarize_citation_validity_averages_the_stored_scalar_rate():
    rows = [_row("a", citation_rate=1.0), _row("b", citation_rate=0.5), _errored_row("c")]

    summary = summarize_citation_validity(rows)

    assert summary.n == 2
    assert summary.mean == pytest.approx(0.75)


# --- build_dod_checks ---


def test_build_dod_checks_flags_faithfulness_below_target():
    rows = [_row("a", faithfulness=0.5)]
    ragas = summarize_ragas(rows)
    judge_agreement = summarize_judge_agreement(rows)

    checks = build_dod_checks(ragas, judge_agreement)

    faithfulness_check = next(c for c in checks if "faithfulness" in c.criterion.lower())
    assert faithfulness_check.met is False


def test_build_dod_checks_flags_faithfulness_at_or_above_target():
    rows = [_row("a", faithfulness=settings.eval.ragas_faithfulness_target)]
    ragas = summarize_ragas(rows)

    checks = build_dod_checks(ragas, summarize_judge_agreement(rows))

    faithfulness_check = next(c for c in checks if "faithfulness" in c.criterion.lower())
    assert faithfulness_check.met is True


def test_build_dod_checks_marks_kappa_unmeasurable_when_judge_agreement_is_none():
    checks = build_dod_checks(summarize_ragas([]), None)

    kappa_check = next(c for c in checks if "kappa" in c.criterion.lower())
    assert kappa_check.met is None


# --- build_scorecard: real-data regression guard ---


def test_build_scorecard_matches_the_real_shipped_unit_4_batch_numbers():
    # Reproduces docs/results/judge-agreement.md's real published numbers
    # from the actual raw batch file -- same regression-guard convention as
    # test_custom_metrics.py's confusion-matrix guard. If this ever drifts,
    # either the aggregation broke or the published doc is stale -- both
    # worth knowing immediately.
    rows = load_rows(REPO_ROOT / "docs" / "results" / "judge-agreement-batch-raw.json")

    scorecard = build_scorecard(rows)

    assert scorecard.n_queries == 30
    assert scorecard.n_generation_ok == 29

    assert scorecard.ragas["faithfulness"].n == 20
    assert scorecard.ragas["faithfulness"].mean == pytest.approx(0.7977396755291493)
    assert scorecard.ragas["response_relevancy"].n == 27
    assert scorecard.ragas["id_based_context_precision"].n == 29
    assert scorecard.ragas["id_based_context_recall"].n == 29

    assert scorecard.judge_agreement is not None
    assert scorecard.judge_agreement.n_paired == 9
    assert scorecard.judge_agreement.cohen_kappa == pytest.approx(-0.0161, abs=1e-3)
    assert scorecard.judge_agreement.cohen_kappa_quadratic == pytest.approx(0.0906, abs=1e-3)
    assert scorecard.judge_agreement.raw_agreement_rate == pytest.approx(0.2222, abs=1e-3)

    assert scorecard.mean_valid_citation_rate.n == 29
    assert scorecard.mean_valid_citation_rate.mean == pytest.approx(0.7068776051534672)

    # Real, small-sample positive result: every Python code block in this
    # batch's real generated answers parsed clean.
    assert scorecard.code_blocks.n_checked == 8
    assert scorecard.code_blocks.n_valid == 8

    faithfulness_check = next(c for c in scorecard.checks if "faithfulness" in c.criterion.lower())
    assert faithfulness_check.met is False  # 0.7977 < 0.90 target, real and disclosed

    kappa_check = next(c for c in scorecard.checks if "kappa" in c.criterion.lower())
    assert kappa_check.met is False  # -0.0161 < 0.6 target, real and disclosed


def test_load_rows_reads_the_real_checkpoint_file_shape():
    rows = load_rows(settings.eval.generation_batch_raw_path)

    assert len(rows) == 30
    assert all("query_id" in r for r in rows)
