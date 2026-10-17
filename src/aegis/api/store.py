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

    def list_feedback(self) -> list[FeedbackRow]:
        with closing(self._conn()) as c:
            rows = c.execute(
                "SELECT f.trace_id, f.rating, f.comment, f.created_at, i.query_text, i.answer_text, i.escalated, i.cache_status, i.rewritten_query "
                "FROM feedback f JOIN interactions i USING (trace_id) ORDER BY f.created_at"
            ).fetchall()
        return [FeedbackRow(trace_id=r[0], rating=r[1], comment=r[2], created_at=r[3], query_text=r[4],
                            answer_text=r[5], escalated=bool(r[6]), cache_status=r[7], rewritten_query=r[8]) for r in rows]
