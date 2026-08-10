import json

from aegis.eval.harvest import harvest_candidates, is_low_signal_resolution, load_resolved_issues
from aegis.types import Stratum


def test_low_signal_resolution_detects_pure_redirects():
    assert is_low_signal_resolution("Please open a new discussion with a minimal reproducible example.") is True
    assert is_low_signal_resolution("Thanks!") is True
    assert is_low_signal_resolution("Closing this as stale.") is True


def test_low_signal_resolution_keeps_terse_real_answers():
    # Short but a complete, real technical answer -- must not be discarded
    # just for being brief (this maintainer is famously terse).
    assert is_low_signal_resolution("Use `response_model=None` to disable response validation entirely.") is False


def test_low_signal_resolution_keeps_long_answers_even_with_a_polite_opener():
    text = (
        "Thanks for reporting! The fix is to set `Response.headers['X-Custom']` explicitly in your "
        "path operation function, since middleware runs after the response object is already built "
        "in this case. See the docs on custom headers for the full example and caveats."
    )
    assert is_low_signal_resolution(text) is False


def _write_jsonl(path, docs):
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")


def test_load_resolved_issues_splits_title_body_resolution(tmp_path):
    path = tmp_path / "documents.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "issue::1",
                "doc_type": "issue",
                "text": "# How do I do X?\n\nBody text here.\n\n**Resolution (@tiangolo):**\n\nUse Y instead.",
                "content_hash": "abc",
                "title": "How do I do X?",
                "resolution_comment_id": "123",
            },
            {
                "id": "issue::2",
                "doc_type": "issue",
                "text": "# Unresolved issue\n\nNo resolution here.",
                "content_hash": "def",
                "title": "Unresolved issue",
                "resolution_comment_id": None,
            },
            {
                "id": "doc::readme",
                "doc_type": "documentation",
                "text": "Some doc text",
                "content_hash": "ghi",
                "title": "Readme",
            },
        ],
    )
    records = load_resolved_issues(path)
    assert len(records) == 1
    assert records[0]["issue_id"] == "issue::1"
    assert records[0]["title"] == "How do I do X?"
    assert records[0]["body"] == "Body text here."
    assert records[0]["resolution"] == "Use Y instead."


def test_load_resolved_issues_skips_issues_without_resolution_marker(tmp_path):
    # A malformed/legacy record with resolution_comment_id set but no
    # matching text marker shouldn't crash the harvest or silently produce
    # a garbage split -- it's dropped.
    path = tmp_path / "documents.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "issue::3",
                "doc_type": "issue",
                "text": "# Title only\n\nNo resolution marker present.",
                "content_hash": "abc",
                "title": "Title only",
                "resolution_comment_id": "999",
            },
        ],
    )
    assert load_resolved_issues(path) == []


def test_harvest_candidates_samples_up_to_target_per_stratum(tmp_path):
    path = tmp_path / "documents.jsonl"
    docs = [
        {
            "id": f"issue::{i}",
            "doc_type": "issue",
            "text": (
                f"# How do I configure thing {i}?\n\n"
                "What's the best way to set this up?\n\n"
                "**Resolution (@x):**\n\nYou can do it like this, by setting the config option explicitly in your app."
            ),
            "content_hash": str(i),
            "title": f"How do I configure thing {i}?",
            "resolution_comment_id": "1",
        }
        for i in range(5)
    ]
    _write_jsonl(path, docs)

    sampled = harvest_candidates(path, target_per_stratum=3, seed=1)
    assert len(sampled[Stratum.HOW_TO]) == 3
    assert sampled[Stratum.CONCEPTUAL] == []
    assert all(record["stratum"] == Stratum.HOW_TO for record in sampled[Stratum.HOW_TO])


def test_harvest_candidates_is_deterministic_for_a_fixed_seed(tmp_path):
    path = tmp_path / "documents.jsonl"
    docs = [
        {
            "id": f"issue::{i}",
            "doc_type": "issue",
            "text": (
                f"# How do I do thing {i}?\n\nHow to set it up?\n\n"
                "**Resolution (@x):**\n\nYou can set it up like this, by passing the option explicitly."
            ),
            "content_hash": str(i),
            "title": f"How do I do thing {i}?",
            "resolution_comment_id": "1",
        }
        for i in range(10)
    ]
    _write_jsonl(path, docs)

    first = harvest_candidates(path, target_per_stratum=4, seed=42)
    second = harvest_candidates(path, target_per_stratum=4, seed=42)
    assert [r["issue_id"] for r in first[Stratum.HOW_TO]] == [r["issue_id"] for r in second[Stratum.HOW_TO]]
