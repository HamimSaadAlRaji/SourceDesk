"""Persistence for /chat interactions and /feedback votes (sqlite, stdlib).

Why record the interaction itself, not just the vote: feedback -> review queue ->
golden-set candidate (plan.md Phase 14 step 5) needs the query and the answer
the user actually saw. Leaving them in Langfuse would tie that loop to its
30-day retention, and trusting the client to send them back would let anyone
inject arbitrary "answers" into the review queue. So the server records them at
`done` time, and a vote is accepted only for a trace_id the server issued.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

_RATINGS = ("up", "down")
_DECISIONS = ("positive", "negative", "reject")


class FeedbackRow(BaseModel):
    trace_id: str
    rating: str
    comment: str | None
    created_at: str
    query_text: str
    answer_text: str
    escalated: bool
    cache_status: str
    rewritten_query: str | None = None


class InteractionStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS interactions (trace_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
                "query_text TEXT NOT NULL, answer_text TEXT NOT NULL, escalated INTEGER NOT NULL, cache_status TEXT NOT NULL, rewritten_query TEXT)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS feedback (trace_id TEXT PRIMARY KEY REFERENCES interactions(trace_id), "
                "rating TEXT NOT NULL CHECK (rating IN ('up','down')), comment TEXT, created_at TEXT NOT NULL)"
            )
            # One human decision per voted interaction (aegis.eval.review_queue).
            # Kept beside the vote, not in the golden set: the golden set is
            # frozen and versioned; promotion only produces candidates.
            c.execute(
                "CREATE TABLE IF NOT EXISTS reviews (trace_id TEXT PRIMARY KEY REFERENCES feedback(trace_id), "
                "decision TEXT NOT NULL CHECK (decision IN ('positive','negative','reject')), stratum TEXT, "
                "negative_category TEXT, note TEXT, reviewed_at TEXT NOT NULL)"
            )

    def _conn(self) -> sqlite3.Connection:
        # One connection per call: requests run on a thread pool and sqlite
        # connections are thread-bound by default.
        return sqlite3.connect(self._path)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def record_interaction(self, trace_id: str, query_text: str, answer_text: str, *, escalated: bool, cache_status: str,
                           rewritten_query: str | None = None) -> None:
        with self._lock, closing(self._conn()) as c, c:
            c.execute(
                "INSERT OR REPLACE INTO interactions VALUES (?,?,?,?,?,?,?)",
                (trace_id, self._now(), query_text, answer_text, int(escalated), cache_status, rewritten_query),
            )

    def add_feedback(self, trace_id: str, rating: str, comment: str | None) -> bool:
        """False if the server never issued this trace_id. A second vote replaces the first."""
        if rating not in _RATINGS:
            raise ValueError(f"rating must be one of {_RATINGS}")
        with self._lock, closing(self._conn()) as c, c:
            if c.execute("SELECT 1 FROM interactions WHERE trace_id=?", (trace_id,)).fetchone() is None:
                return False
            c.execute("INSERT OR REPLACE INTO feedback VALUES (?,?,?,?)", (trace_id, rating, comment, self._now()))
            return True

    _FEEDBACK_SELECT = (
        "SELECT f.trace_id, f.rating, f.comment, f.created_at, i.query_text, i.answer_text, i.escalated, i.cache_status, "
        "i.rewritten_query FROM feedback f JOIN interactions i USING (trace_id) "
    )

    @staticmethod
    def _feedback_row(r: tuple) -> FeedbackRow:
        return FeedbackRow(trace_id=r[0], rating=r[1], comment=r[2], created_at=r[3], query_text=r[4],
                           answer_text=r[5], escalated=bool(r[6]), cache_status=r[7], rewritten_query=r[8])

    def list_feedback(self) -> list[FeedbackRow]:
        with closing(self._conn()) as c:
            rows = c.execute(self._FEEDBACK_SELECT + "ORDER BY f.created_at").fetchall()
        return [self._feedback_row(r) for r in rows]

    def pending_review(self) -> list[FeedbackRow]:
        """Voted, not yet reviewed. Down-votes first: they are where a real
        failure is most likely, and the golden set is weakest on real failures."""
        with closing(self._conn()) as c:
            rows = c.execute(
                self._FEEDBACK_SELECT + "WHERE f.trace_id NOT IN (SELECT trace_id FROM reviews) "
                "ORDER BY f.rating = 'up', f.created_at"
            ).fetchall()
        return [self._feedback_row(r) for r in rows]

    def record_review(self, trace_id: str, decision: str, *, stratum: str | None = None,
                      negative_category: str | None = None, note: str | None = None) -> bool:
        """False if there is no vote on this trace_id. A second review replaces the first."""
        if decision not in _DECISIONS:
            raise ValueError(f"decision must be one of {_DECISIONS}")
        if decision == "positive" and not stratum:
            raise ValueError("a positive candidate needs a stratum")
        if decision == "negative" and not negative_category:
            raise ValueError("a negative candidate needs a negative_category")
        with self._lock, closing(self._conn()) as c, c:
            if c.execute("SELECT 1 FROM feedback WHERE trace_id=?", (trace_id,)).fetchone() is None:
                return False
            c.execute(
                "INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?,?)",
                (trace_id, decision, stratum, negative_category, note, self._now()),
            )
            return True

    def reviewed(self) -> list[dict]:
        """Every review joined to its vote and interaction, ordered by trace_id (stable export order)."""
        with closing(self._conn()) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT r.trace_id, r.decision, r.stratum, r.negative_category, r.note, r.reviewed_at, "
                "f.rating, f.comment, i.query_text, i.answer_text, i.escalated, i.cache_status, i.rewritten_query, "
                "i.created_at AS asked_at FROM reviews r JOIN feedback f USING (trace_id) "
                "JOIN interactions i USING (trace_id) ORDER BY r.trace_id"
            ).fetchall()
        return [dict(r) for r in rows]
