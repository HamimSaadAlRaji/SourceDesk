"""TDD for the feedback -> review queue -> golden-set candidate loop (plan.md
Phase 14 step 5, design record 14g). The export feeds future evaluation data,
so a silent bug here (a holdout query leaking into dev, a follow-up exported
without the text it resolved to, the system's own citations becoming
"ground truth") would corrupt numbers later -- hence TDD, CLAUDE.md SS6."""

import json

import pytest

from aegis.api.store import InteractionStore
from aegis.eval.review_queue import export_candidates, review_session
from aegis.types import GoldenQuery, NegativeCategory, Stratum


def _store(tmp_path) -> InteractionStore:
    return InteractionStore(tmp_path / "i.sqlite")


def _voted(store, trace_id, rating, query="how do I declare a query parameter", *, escalated=False,
           rewritten=None, comment=None):
    store.record_interaction(trace_id, query, f"answer for {trace_id}", escalated=escalated,
                             cache_status="miss", rewritten_query=rewritten)
    store.add_feedback(trace_id, rating, comment)


# --- the queue ---


def test_queue_holds_only_voted_unreviewed_interactions_downvotes_first(tmp_path):
    store = _store(tmp_path)
    _voted(store, "t-up", "up")
    _voted(store, "t-down", "down")
    store.record_interaction("t-novote", "q", "a", escalated=False, cache_status="miss")

    assert [r.trace_id for r in store.pending_review()] == ["t-down", "t-up"]


def test_reviewed_items_leave_the_queue(tmp_path):
    store = _store(tmp_path)
    _voted(store, "t1", "down")
    store.record_review("t1", "reject", note="spam")

    assert store.pending_review() == []


def test_review_requires_a_voted_interaction_and_a_valid_decision(tmp_path):
    store = _store(tmp_path)
    _voted(store, "t1", "down")

    assert store.record_review("never-issued", "reject") is False
    with pytest.raises(ValueError):
        store.record_review("t1", "maybe")
    with pytest.raises(ValueError):
        store.record_review("t1", "positive")  # a positive candidate needs a stratum
    with pytest.raises(ValueError):
        store.record_review("t1", "negative")  # a negative candidate needs a category


# --- the export ---


def _export(tmp_path, store, golden=()) -> list[dict]:
    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text("".join(json.dumps(g) + "\n" for g in golden))
    out = tmp_path / "candidates_feedback.jsonl"
    export_candidates(store, out, golden_paths=[golden_path])
    return [json.loads(line) for line in out.read_text().splitlines()]


def test_positive_candidate_is_a_valid_unlabeled_golden_query_with_provenance(tmp_path):
    store = _store(tmp_path)
    _voted(store, "t1", "down", comment="wrong param")
    store.record_review("t1", "positive", stratum=Stratum.HOW_TO)

    (row,) = _export(tmp_path, store)

    q = GoldenQuery(**row)
    assert q.id == "feedback::t1" and q.is_negative is False and q.stratum == Stratum.HOW_TO
    assert q.ground_truth_chunk_ids == []  # labeled later from a permissive pool, never from the system's citations
    assert row["split"] == "candidate"
    assert row["feedback"]["rating"] == "down" and row["feedback"]["comment"] == "wrong param"
    assert row["feedback"]["answer_shown"] == "answer for t1"
    assert "citations" not in json.dumps(row)


def test_negative_candidate_carries_the_category_expected_behavior(tmp_path):
    store = _store(tmp_path)
    _voted(store, "t1", "down", query="refund my invoice")
    store.record_review("t1", "negative", negative_category=NegativeCategory.BILLING_LEGAL)

    (row,) = _export(tmp_path, store)

    q = GoldenQuery(**row)
    assert q.is_negative and q.negative_category == NegativeCategory.BILLING_LEGAL
    assert q.expected_behavior == "hard_escalate"


def test_follow_up_exports_the_standalone_rewritten_text(tmp_path):
    # "what about making it required?" means nothing without its history; the
    # rewritten text is what the system actually searched for.
    store = _store(tmp_path)
    _voted(store, "t1", "down", query="what about making it required?",
           rewritten="how do I make a query parameter required in FastAPI")
    store.record_review("t1", "positive", stratum=Stratum.HOW_TO)

    (row,) = _export(tmp_path, store)

    assert row["text"] == "how do I make a query parameter required in FastAPI"
    assert row["feedback"]["original_query"] == "what about making it required?"


def test_rejected_and_unreviewed_items_are_not_exported(tmp_path):
    store = _store(tmp_path)
    _voted(store, "t1", "down")
    _voted(store, "t2", "up")
    store.record_review("t1", "reject")

    assert _export(tmp_path, store) == []


def test_duplicates_of_existing_golden_queries_are_flagged_with_their_split(tmp_path):
    store = _store(tmp_path)
    _voted(store, "t1", "down", query="  How do I declare a Query parameter ")
    store.record_review("t1", "positive", stratum=Stratum.HOW_TO)
    golden = [{"id": "golden::issue::9", "text": "how do I declare a query parameter", "is_negative": False,
               "split": "holdout"}]

    (row,) = _export(tmp_path, store, golden)

    assert row["duplicate_of"] == {"id": "golden::issue::9", "split": "holdout"}


def test_export_is_deterministic_and_idempotent(tmp_path):
    store = _store(tmp_path)
    for t in ("t2", "t1"):
        _voted(store, t, "down")
        store.record_review(t, "positive", stratum=Stratum.CONCEPTUAL)

    first = _export(tmp_path, store)
    second = _export(tmp_path, store)

    assert first == second and [r["id"] for r in first] == ["feedback::t1", "feedback::t2"]


# --- the interactive session (scripted input) ---


def test_session_records_decisions_from_scripted_input(tmp_path):
    store = _store(tmp_path)
    _voted(store, "t-a", "down", query="how do I declare a query parameter")
    _voted(store, "t-b", "down", query="refund my invoice")
    _voted(store, "t-c", "up")
    answers = iter(["p", "how_to", "n", "billing_legal", "r"])
    shown: list[str] = []

    review_session(store, input_fn=lambda prompt: next(answers), print_fn=shown.append)

    rows = {r["trace_id"]: r for r in store.reviewed()}
    assert rows["t-a"]["decision"] == "positive" and rows["t-a"]["stratum"]
    assert rows["t-b"]["decision"] == "negative" and rows["t-b"]["negative_category"] == "billing_legal"
    assert rows["t-c"]["decision"] == "reject"
    assert store.pending_review() == []


def test_session_quit_and_skip_leave_items_pending(tmp_path):
    store = _store(tmp_path)
    _voted(store, "t-a", "down")
    _voted(store, "t-b", "down", query="another question")
    answers = iter(["s", "q"])

    review_session(store, input_fn=lambda prompt: next(answers), print_fn=lambda s: None)

    assert {r.trace_id for r in store.pending_review()} == {"t-a", "t-b"}


def test_session_reprompts_on_an_invalid_choice(tmp_path):
    store = _store(tmp_path)
    _voted(store, "t-a", "down")
    answers = iter(["x", "n", "not_a_category", "out_of_corpus"])

    review_session(store, input_fn=lambda prompt: next(answers), print_fn=lambda s: None)

    (row,) = store.reviewed()
    assert row["negative_category"] == "out_of_corpus"


def test_stratum_has_no_enter_to_accept_default(tmp_path):
    # The text-only heuristic agrees with the frozen strata on 138/180, so a
    # blank answer must re-prompt rather than silently take its guess.
    store = _store(tmp_path)
    _voted(store, "t-a", "down")
    answers = iter(["p", "", "conceptual"])

    review_session(store, input_fn=lambda prompt: next(answers), print_fn=lambda s: None)

    (row,) = store.reviewed()
    assert row["stratum"] == "conceptual"
