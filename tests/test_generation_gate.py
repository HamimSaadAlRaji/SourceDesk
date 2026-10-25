"""The generation half of the CI gate (design record 16j). valid_citation_rate is
vacuously 1.0 for an answer with nothing to check, so on its own it can *rise* when the
model stops answering or stops citing. These pin the guards: an empty answer scores zero,
and cited-answer rate and empty-answer rate are tracked as their own metrics."""

from __future__ import annotations

import math

from aegis.eval.generation_gate import AnswerRow, load_queries, score_generation


def _row(rate=1.0, supported=1, checked=0, valid=0, empty=False, ms=1000.0) -> AnswerRow:
    return AnswerRow(valid_citation_rate=rate, n_supported_citations=supported, code_checked=checked,
                     code_valid=valid, empty=empty, generation_ms=ms)


def test_means_and_rates_over_ordinary_answers():
    m = score_generation([_row(rate=1.0), _row(rate=0.5), _row(rate=0.0, supported=0)])
    assert m["valid_citation_rate"] == 0.5
    assert m["cited_answer_rate"] == 2 / 3
    assert m["empty_answer_rate"] == 0.0


def test_an_empty_answer_scores_zero_not_the_vacuous_one():
    # A model that returns "" has nothing to validate: valid_citation_rate would be 1.0.
    m = score_generation([_row(rate=1.0), _row(rate=1.0, supported=0, empty=True)])
    assert m["valid_citation_rate"] == 0.5
    assert m["empty_answer_rate"] == 0.5
    assert m["cited_answer_rate"] == 0.5


def test_code_syntax_rate_is_blocks_valid_over_blocks_checked_across_all_answers():
    m = score_generation([_row(checked=3, valid=3), _row(checked=1, valid=0), _row()])
    assert m["code_syntax_valid_rate"] == 3 / 4


def test_no_code_blocks_at_all_is_nan_not_a_pass():
    assert math.isnan(score_generation([_row(), _row()])["code_syntax_valid_rate"])


def test_latency_p95_uses_generation_time():
    rows = [_row(ms=float(ms)) for ms in range(1, 101)]
    assert abs(score_generation(rows)["generation_p95_ms"] - 95.05) < 1e-9


def test_no_answers_is_all_nan():
    m = score_generation([])
    assert all(math.isnan(v) for v in m.values())


def test_the_query_sample_is_the_seeded_twenty_positive_dev_queries():
    queries = load_queries()
    assert len(queries) == 20
    assert all(not q.is_negative for q in queries)
    assert [q.id for q in queries] == [q.id for q in load_queries()]  # deterministic
