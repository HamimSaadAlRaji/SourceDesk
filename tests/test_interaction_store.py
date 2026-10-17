from __future__ import annotations

import pytest

from aegis.api.store import InteractionStore


def _store(tmp_path):
    return InteractionStore(tmp_path / "i.sqlite")


def test_feedback_requires_a_recorded_interaction(tmp_path):
    s = _store(tmp_path)
    assert s.add_feedback("a" * 32, "up", None) is False
    assert s.list_feedback() == []


def test_feedback_roundtrip_joins_query_and_answer(tmp_path):
    s = _store(tmp_path)
    s.record_interaction("t1", "how do I x?", "do y [1]", escalated=False, cache_status="miss")
    assert s.add_feedback("t1", "down", "wrong version") is True
    [row] = s.list_feedback()
    assert (row.trace_id, row.rating, row.comment) == ("t1", "down", "wrong version")
    assert (row.query_text, row.answer_text, row.escalated) == ("how do I x?", "do y [1]", False)


def test_second_vote_replaces_first(tmp_path):
    s = _store(tmp_path)
    s.record_interaction("t1", "q", "a", escalated=True, cache_status="miss")
    s.add_feedback("t1", "up", None)
    s.add_feedback("t1", "down", "changed my mind")
    [row] = s.list_feedback()
    assert row.rating == "down" and row.comment == "changed my mind"


def test_persists_across_instances(tmp_path):
    s = _store(tmp_path)
    s.record_interaction("t1", "q", "a", escalated=False, cache_status="miss")
    s.add_feedback("t1", "up", None)
    assert [r.trace_id for r in _store(tmp_path).list_feedback()] == ["t1"]


def test_rejects_unknown_rating(tmp_path):
    s = _store(tmp_path)
    s.record_interaction("t1", "q", "a", escalated=False, cache_status="miss")
    with pytest.raises(ValueError):
        s.add_feedback("t1", "meh", None)


def test_rewritten_query_roundtrip(tmp_path):
    s = _store(tmp_path)
    s.record_interaction("t1", "and the free tier?", "a", escalated=False, cache_status="miss", rewritten_query="free tier limits?")
    s.add_feedback("t1", "up", None)
    assert s.list_feedback()[0].rewritten_query == "free tier limits?"
