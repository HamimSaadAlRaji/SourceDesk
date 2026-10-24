"""The PR comment (design record 16f). A comment that says "passed" when a gate was
skipped, cancelled or crashed before writing its table is worse than no comment, so
the verdict logic is pinned: only a failure is a failure, but only real successes
are described as passes, and anything not run is said to be not run."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

# The script is stdlib-only and run by path in CI (no dependency install), so load it by path too.
_SPEC = importlib.util.spec_from_file_location(
    "pr_comment", Path(__file__).resolve().parents[1] / "src" / "aegis" / "eval" / "pr_comment.py"
)
pr_comment = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pr_comment)

RET = "### Retrieval: ✅ passed\n\n| Metric | Baseline | Current |\n|---|---|---|\n| recall@10 | 0.4435 | 0.4435 |\n"
RTR = "### Router: ✅ passed\n\n| Metric | Baseline | Current |\n|---|---|---|\n| accuracy | 0.7759 | 0.7759 |\n"


def _build(retrieval="success", llm="success", ret_md=RET, rtr_md=RTR, spend=0.01536):
    return pr_comment.build_comment(
        sha="a1b2c3d4e5f6", run_url="https://example.test/run/1",
        jobs={"retrieval": retrieval, "router": llm},
        sections={"retrieval": ret_md, "router": rtr_md}, spend_usd=spend,
    )


def test_all_green_says_so_and_includes_both_tables_the_link_and_the_sha():
    body = _build()
    assert body.startswith(pr_comment.MARKER)
    assert "✅ All evaluation gates passed" in body
    assert "recall@10" in body and "accuracy" in body
    assert "https://example.test/run/1" in body and "a1b2c3d" in body
    assert "$0.0154" in body


def test_a_failed_gate_is_the_headline_and_names_the_gate():
    body = _build(retrieval="failure", ret_md=RET.replace("passed", "FAILED"))
    assert "❌" in body.splitlines()[1]
    assert "Retrieval" in body.splitlines()[1]
    assert "All evaluation gates passed" not in body


def test_a_failure_with_no_table_still_fails_and_says_no_result_was_produced():
    body = _build(llm="failure", rtr_md=None)
    assert "❌" in body.splitlines()[1]
    assert "no result was produced" in body.lower()


def test_a_skipped_gate_is_reported_as_not_run_not_as_passed():
    body = _build(llm="skipped", rtr_md=None)
    assert "not run" in body.lower()
    assert "All evaluation gates passed" not in body
    assert "✅" in body.splitlines()[1]  # nothing failed...
    assert "1 gate not run" in body       # ...but the headline discloses the gap


def test_a_cancelled_gate_is_incomplete_not_green():
    body = _build(retrieval="cancelled", ret_md=None)
    assert "⚠️" in body.splitlines()[1]
    assert "✅ All" not in body


def test_failure_beats_cancelled_in_the_headline():
    body = _build(retrieval="cancelled", llm="failure", ret_md=None, rtr_md=None)
    assert "❌" in body.splitlines()[1]


def test_the_comment_stays_under_githubs_size_limit():
    body = _build(ret_md="x" * 200_000)
    assert len(body) < 60_000
    assert "truncated" in body.lower()


def test_spend_is_read_from_the_ledger_file(tmp_path):
    db = tmp_path / "ci_spend.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE spend (id TEXT, ts REAL, model TEXT, purpose TEXT, input_tokens INT, "
                "cached_input_tokens INT, output_tokens INT, cost_usd REAL)")
    con.executemany("INSERT INTO spend VALUES (?,?,?,?,?,?,?,?)", [("a", 0, "m", "router", 1, 0, 1, 0.01), ("b", 0, "m", "router", 1, 0, 1, 0.005)])
    con.commit()
    con.close()
    assert abs(pr_comment.read_spend(db) - 0.015) < 1e-12
    assert pr_comment.read_spend(tmp_path / "missing.sqlite") is None
