from __future__ import annotations

import pytest

from aegis.api.store import InteractionStore


def _store(tmp_path):
    return InteractionStore(tmp_path / "i.sqlite")


def test_feedback_requires_a_recorded_interaction(tmp_path):
    s = _store(tmp_path)
    assert s.add_feedback("a" * 32, "up", None, owner="o") is False
    assert s.list_feedback() == []


def test_feedback_roundtrip_joins_query_and_answer(tmp_path):
    s = _store(tmp_path)
    s.record_interaction("t1", "how do I x?", "do y [1]", escalated=False, cache_status="miss", owner="o")
    assert s.add_feedback("t1", "down", "wrong version", owner="o") is True
    [row] = s.list_feedback()
    assert (row.trace_id, row.rating, row.comment) == ("t1", "down", "wrong version")
    assert (row.query_text, row.answer_text, row.escalated) == ("how do I x?", "do y [1]", False)


def test_second_vote_replaces_first(tmp_path):
    s = _store(tmp_path)
    s.record_interaction("t1", "q", "a", escalated=True, cache_status="miss", owner="o")
    s.add_feedback("t1", "up", None, owner="o")
    s.add_feedback("t1", "down", "changed my mind", owner="o")
    [row] = s.list_feedback()
    assert row.rating == "down" and row.comment == "changed my mind"


def test_persists_across_instances(tmp_path):
    s = _store(tmp_path)
    s.record_interaction("t1", "q", "a", escalated=False, cache_status="miss", owner="o")
    s.add_feedback("t1", "up", None, owner="o")
    assert [r.trace_id for r in _store(tmp_path).list_feedback()] == ["t1"]


def test_rejects_unknown_rating(tmp_path):
    s = _store(tmp_path)
    s.record_interaction("t1", "q", "a", escalated=False, cache_status="miss", owner="o")
    with pytest.raises(ValueError):
        s.add_feedback("t1", "meh", None, owner="o")


def test_rewritten_query_roundtrip(tmp_path):
    s = _store(tmp_path)
    s.record_interaction("t1", "and the free tier?", "a", escalated=False, cache_status="miss", owner="o", rewritten_query="free tier limits?")
    s.add_feedback("t1", "up", None, owner="o")
    assert s.list_feedback()[0].rewritten_query == "free tier limits?"


# ---- Phase 15 (design record 15c): ownership ----------------------------------------


def test_only_the_owner_can_vote(tmp_path):
    s = InteractionStore(tmp_path / "i.sqlite")
    s.record_interaction("t1", "q", "a", escalated=False, cache_status="miss", owner="session:alice")
    assert s.add_feedback("t1", "down", "nope", owner="session:mallory") is False
    assert s.list_feedback() == []
    assert s.add_feedback("t1", "up", None, owner="session:alice") is True


def test_owner_is_required_on_both_writes(tmp_path):
    s = InteractionStore(tmp_path / "i.sqlite")
    with pytest.raises(TypeError):
        s.record_interaction("t1", "q", "a", escalated=False, cache_status="miss")
    with pytest.raises(TypeError):
        s.add_feedback("t1", "up", None)


def test_rows_from_before_ownership_are_migrated_and_votable_by_nobody(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as c:  # the Phase 14 schema, no owner column
        c.execute("CREATE TABLE interactions (trace_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, query_text TEXT NOT NULL, "
                  "answer_text TEXT NOT NULL, escalated INTEGER NOT NULL, cache_status TEXT NOT NULL, rewritten_query TEXT)")
        c.execute("INSERT INTO interactions VALUES ('old','2026-10-01','q','a',0,'miss',NULL)")
    s = InteractionStore(path)
    assert s.add_feedback("old", "up", None, owner="session:anyone") is False
    s.record_interaction("new", "q", "a", escalated=False, cache_status="miss", owner="session:x")
    assert s.add_feedback("new", "up", None, owner="session:x") is True
