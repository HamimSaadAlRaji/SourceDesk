"""Paid-API spend ledger with a hard budget (Phase 15 step 0, design record 15g).

The user's cap is $10 for the whole project. A dashboard limit stops spend
only after the fact; this stops it before the call. Every paid call:

1. `reserve()` -- its worst case (counted input tokens + the full
   `max_output_tokens`, at list price) is checked against
   budget - spent - outstanding reservations, inside one IMMEDIATE sqlite
   transaction, so concurrent threads *and* separate processes (server +
   an experiment) cannot jointly overshoot. Over budget -> BudgetExceededError,
   and the provider is never contacted.
2. `settle()` -- the provider's billed usage replaces the reservation.
   `release()` -- the call failed; nothing was billed.

Reservations older than `reservation_ttl_s` are ignored (a crashed process
must not lock the budget forever). Known gap: a crash *after* the provider
billed and before `settle` loses that one call's record; worst case is one
call's cost, bounded by its reservation.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from collections import defaultdict
from contextlib import closing
from pathlib import Path


class BudgetExceededError(RuntimeError):
    """The call's worst-case cost does not fit in the remaining budget."""


class SpendLedger:
    def __init__(self, path: Path, budget_usd: float, prices: dict[str, dict[str, float]],
                 reservation_ttl_s: float = 600.0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self.budget_usd = budget_usd
        self.prices = prices
        self.ttl = reservation_ttl_s
        with closing(self._conn()) as c, c:
            c.execute("CREATE TABLE IF NOT EXISTS spend (id TEXT PRIMARY KEY, ts REAL NOT NULL, model TEXT NOT NULL, "
                      "purpose TEXT NOT NULL, input_tokens INTEGER, cached_input_tokens INTEGER, "
                      "output_tokens INTEGER, cost_usd REAL NOT NULL)")
            c.execute("CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, ts REAL NOT NULL, model TEXT NOT NULL, "
                      "purpose TEXT NOT NULL, worst_usd REAL NOT NULL)")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=30, isolation_level=None)

    def cost(self, model: str, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> float:
        p = self.prices[model]  # KeyError on an unpriced model: never assume free
        uncached = max(0, input_tokens - cached_input_tokens)
        return (uncached * p["input"] + cached_input_tokens * p["cached_input"] + output_tokens * p["output"]) / 1e6

    def reserve(self, model: str, input_tokens: int, max_output_tokens: int, purpose: str) -> str:
        worst = self.cost(model, input_tokens, 0, max_output_tokens)
        rid = uuid.uuid4().hex
        with closing(self._conn()) as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                spent = c.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM spend").fetchone()[0]
                held = c.execute("SELECT COALESCE(SUM(worst_usd), 0) FROM reservations WHERE ts > ?",
                                 (time.time() - self.ttl,)).fetchone()[0]
                if spent + held + worst > self.budget_usd:
                    raise BudgetExceededError(
                        f"{purpose}: worst case ${worst:.4f} exceeds remaining ${self.budget_usd - spent - held:.4f} "
                        f"of the ${self.budget_usd:.2f} budget (spent ${spent:.4f}, reserved ${held:.4f})")
                c.execute("INSERT INTO reservations VALUES (?,?,?,?,?)", (rid, time.time(), model, purpose, worst))
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise
        return rid

    def settle(self, reservation_id: str, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> float:
        with closing(self._conn()) as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT model, purpose FROM reservations WHERE id=?", (reservation_id,)).fetchone()
            if row is None:
                c.execute("ROLLBACK")
                raise KeyError(f"unknown reservation {reservation_id}")
            model, purpose = row
            cost = self.cost(model, input_tokens, cached_input_tokens, output_tokens)
            c.execute("INSERT INTO spend VALUES (?,?,?,?,?,?,?,?)",
                      (reservation_id, time.time(), model, purpose, input_tokens, cached_input_tokens, output_tokens, cost))
            c.execute("DELETE FROM reservations WHERE id=?", (reservation_id,))
            c.execute("COMMIT")
        return cost

    def release(self, reservation_id: str) -> None:
        with closing(self._conn()) as c, c:
            c.execute("DELETE FROM reservations WHERE id=?", (reservation_id,))

    def spent_usd(self) -> float:
        with closing(self._conn()) as c:
            return c.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM spend").fetchone()[0]

    def reserved_usd(self) -> float:
        with closing(self._conn()) as c:
            return c.execute("SELECT COALESCE(SUM(worst_usd), 0) FROM reservations WHERE ts > ?",
                             (time.time() - self.ttl,)).fetchone()[0]

    def summary(self) -> dict:
        with closing(self._conn()) as c:
            rows = c.execute("SELECT model, purpose, input_tokens, cached_input_tokens, output_tokens, cost_usd FROM spend").fetchall()
        by_purpose: dict[str, float] = defaultdict(float)
        by_model: dict[str, float] = defaultdict(float)
        for model, purpose, *_rest, cost in rows:
            by_purpose[purpose] += cost
            by_model[model] += cost
        spent = sum(r[-1] for r in rows)
        return {"budget_usd": self.budget_usd, "spent_usd": round(spent, 6), "remaining_usd": round(self.budget_usd - spent, 6),
                "calls": len(rows), "input_tokens": sum(r[2] or 0 for r in rows), "output_tokens": sum(r[4] or 0 for r in rows),
                "by_purpose": {k: round(v, 6) for k, v in by_purpose.items()},
                "by_model": {k: round(v, 6) for k, v in by_model.items()}}
