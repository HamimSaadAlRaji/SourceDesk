"""HTTP layer for Phase 14 Unit 2. The pipeline is a fake that yields canned
events (or raises if it must not be reached), so these tests pin the wire
format and the request-validation boundary, not retrieval or generation."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from aegis.api.app import create_app
from aegis.config import settings
from aegis.events import CitationEvent, DoneEvent, ErrorEvent, EscalationEvent, TokenEvent


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.api, "interactions_db_path", tmp_path / "default.sqlite")


class _FakePipeline:
    def __init__(self, events=()):
        self.events = list(events)
        self.queries = []
        self.warmed = False

    def warm_up(self):
        self.warmed = True
        return {}

    def answer_stream(self, query, history=None):
        self.queries.append(query)
        self.histories = getattr(self, "histories", []) + [history]
        yield from self.events


class _MustNotRun:
    def warm_up(self):
        return {}

    def answer_stream(self, query, history=None):
        raise AssertionError("pipeline reached despite invalid request")


def _frames(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = block.split("\n")
        assert lines[0].startswith("event: ") and lines[1].startswith("data: "), block
        out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


_DONE = DoneEvent(trace_id="t1", cache_status="miss", escalate=False)


def test_chat_streams_typed_events_in_order_as_sse():
    pipe = _FakePipeline([
        TokenEvent(text="Hello "), TokenEvent(text="world [1]"),
        CitationEvent(citation_id=1, chunk_id="c1", doc_id="d1", heading_path=["A"], text="src"),
        _DONE,
    ])
    with TestClient(create_app(pipe)) as client:
        r = client.post("/chat", json={"query": "how do I x?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    frames = _frames(r.text)
    assert [n for n, _ in frames] == ["token", "token", "citation", "done"]
    assert frames[2][1]["chunk_id"] == "c1" and frames[2][1]["text"] == "src"
    assert frames[3][1]["trace_id"] == "t1"
    assert pipe.queries[0].text == "how do I x?" and pipe.queries[0].id


def test_escalation_is_its_own_event_type():
    pipe = _FakePipeline([EscalationEvent(cause="low_confidence", reason="low_confidence", message="m", after_retrieval=True), _DONE])
    with TestClient(create_app(pipe)) as client:
        frames = _frames(client.post("/chat", json={"query": "q"}).text)
    assert [n for n, _ in frames] == ["escalation", "done"]


def test_pipeline_error_event_is_delivered_with_http_200():
    pipe = _FakePipeline([TokenEvent(text="partial"), ErrorEvent(code="APIError", message="x", trace_id="t")])
    with TestClient(create_app(pipe)) as client:
        r = client.post("/chat", json={"query": "q"})
    assert r.status_code == 200
    assert [n for n, _ in _frames(r.text)] == ["token", "error"]


def test_invalid_requests_rejected_before_pipeline():
    with TestClient(create_app(_MustNotRun())) as client:
        assert client.post("/chat", json={"query": ""}).status_code == 422
        assert client.post("/chat", json={"query": "   "}).status_code == 422
        assert client.post("/chat", json={}).status_code == 422
        assert client.post("/chat", json={"query": "x" * (settings.api.max_query_chars + 1)}).status_code == 422


def test_health_and_startup_warm_up():
    pipe = _FakePipeline()
    with TestClient(create_app(pipe)) as client:
        r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert pipe.warmed is True


# ---- Unit 3: /trace/{id}, /feedback -------------------------------------------------

from aegis.api.store import InteractionStore  # noqa: E402

_TID = "ab" * 16


class _FakeAudit:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, trace_id):
        self.calls.append(trace_id)
        return self.result


def _app(tmp_path, events, audit=None):
    store = InteractionStore(tmp_path / "i.sqlite")
    return create_app(_FakePipeline(events), store=store, audit=audit), store


def test_chat_records_interaction_so_feedback_can_attach(tmp_path):
    done = DoneEvent(trace_id=_TID, cache_status="miss", escalate=False)
    app, store = _app(tmp_path, [TokenEvent(text="do "), TokenEvent(text="y [1]"), done])
    with TestClient(app) as c:
        c.post("/chat", json={"query": "how?"})
        r = c.post("/feedback", json={"trace_id": _TID, "rating": "up"})
    assert r.status_code == 200
    [row] = store.list_feedback()
    assert (row.query_text, row.answer_text, row.rating) == ("how?", "do y [1]", "up")


def test_escalation_message_is_recorded_as_the_answer(tmp_path):
    esc = EscalationEvent(cause="low_confidence", reason="low_confidence", message="escalated", after_retrieval=True)
    app, store = _app(tmp_path, [esc, DoneEvent(trace_id=_TID, cache_status="miss", escalate=True)])
    with TestClient(app) as c:
        c.post("/chat", json={"query": "q"})
        c.post("/feedback", json={"trace_id": _TID, "rating": "down", "comment": "should have answered"})
    [row] = store.list_feedback()
    assert row.escalated is True and row.answer_text == "escalated"


def test_stream_that_errors_is_not_recorded(tmp_path):
    app, _ = _app(tmp_path, [TokenEvent(text="partial"), ErrorEvent(code="X", message="m", trace_id=_TID)])
    with TestClient(app) as c:
        c.post("/chat", json={"query": "q"})
        assert c.post("/feedback", json={"trace_id": _TID, "rating": "up"}).status_code == 404


def test_feedback_validation(tmp_path):
    app, _ = _app(tmp_path, [])
    with TestClient(app) as c:
        assert c.post("/feedback", json={"trace_id": _TID, "rating": "meh"}).status_code == 422
        assert c.post("/feedback", json={"trace_id": "nope", "rating": "up"}).status_code == 422
        assert c.post("/feedback", json={"trace_id": _TID, "rating": "up", "comment": "x" * 5000}).status_code == 422
        assert c.post("/feedback", json={"trace_id": _TID, "rating": "up"}).status_code == 404


_OPERATOR_KEY = "o" * 43


@pytest.fixture
def _operator_key(monkeypatch):
    # /trace is operator-only since Phase 15 (design record 15c).
    from aegis.api.auth import hash_key
    from aegis.config import ApiKeyGrant

    monkeypatch.setattr(settings.api, "api_keys",
                        {hash_key(_OPERATOR_KEY): ApiKeyGrant(name="op", partition="full", can_trace=True)})
    return {"Authorization": f"Bearer {_OPERATOR_KEY}"}


def test_trace_endpoint_returns_audit_and_validates_id(tmp_path, _operator_key):
    audit = _FakeAudit({"trail": {"trace_id": _TID}, "evidence": []})
    app, _ = _app(tmp_path, [], audit=audit)
    with TestClient(app) as c:
        assert c.get(f"/trace/{_TID}", headers=_operator_key).json()["trail"]["trace_id"] == _TID
        assert c.get("/trace/not-a-trace-id", headers=_operator_key).status_code == 422
    assert audit.calls == [_TID]


def test_trace_not_found_is_404(tmp_path, _operator_key):
    app, _ = _app(tmp_path, [], audit=_FakeAudit(None))
    with TestClient(app) as c:
        assert c.get(f"/trace/{_TID}", headers=_operator_key).status_code == 404


# ---- Unit 4: conversation history ---------------------------------------------------


def test_history_is_passed_as_role_text_pairs():
    pipe = _FakePipeline([_DONE])
    hist = [{"role": "user", "text": "q1"}, {"role": "assistant", "text": "a1"}]
    with TestClient(create_app(pipe)) as c:
        assert c.post("/chat", json={"query": "and then?", "history": hist}).status_code == 200
    assert pipe.histories == [[("user", "q1"), ("assistant", "a1")]]


def test_no_history_means_none():
    pipe = _FakePipeline([_DONE])
    with TestClient(create_app(pipe)) as c:
        c.post("/chat", json={"query": "q"})
    assert pipe.histories == [None]


def test_history_is_windowed_to_the_most_recent_messages_and_truncated(monkeypatch):
    monkeypatch.setattr(settings.api, "history_window_messages", 2)
    monkeypatch.setattr(settings.api, "history_message_chars", 5)
    pipe = _FakePipeline([_DONE])
    hist = [{"role": "user", "text": "old"}, {"role": "assistant", "text": "x" * 50}, {"role": "user", "text": "newest question"}]
    with TestClient(create_app(pipe)) as c:
        c.post("/chat", json={"query": "q", "history": hist})
    assert pipe.histories == [[("assistant", "xxxxx"), ("user", "newes")]]


def test_history_validation():
    with TestClient(create_app(_MustNotRun())) as c:
        assert c.post("/chat", json={"query": "q", "history": [{"role": "system", "text": "x"}]}).status_code == 422
        too_many = [{"role": "user", "text": "x"}] * (settings.api.max_history_messages + 1)
        assert c.post("/chat", json={"query": "q", "history": too_many}).status_code == 422


def test_rewritten_query_is_recorded_with_the_interaction(tmp_path):
    done = DoneEvent(trace_id=_TID, cache_status="miss", escalate=False, rewritten_query="standalone q")
    app, store = _app(tmp_path, [TokenEvent(text="a"), done])
    with TestClient(app) as c:
        c.post("/chat", json={"query": "follow up"})
        c.post("/feedback", json={"trace_id": _TID, "rating": "up"})
    [row] = store.list_feedback()
    assert (row.query_text, row.rewritten_query) == ("follow up", "standalone q")


# --- rate limiting (Phase 14 build item 6, design record 14f) ---


def _limited_app(pipe, capacity=2, per_minute=6.0):
    from aegis.api.ratelimit import TokenBucketLimiter

    return create_app(pipe, limiter=TokenBucketLimiter(capacity, per_minute, max_clients=100))


def test_chat_over_the_limit_is_429_with_retry_after_and_never_reaches_the_pipeline():
    pipe = _FakePipeline([_DONE])
    with TestClient(_limited_app(pipe, capacity=2)) as client:
        codes = [client.post("/chat", json={"query": f"q{i}"}).status_code for i in range(2)]
        r = client.post("/chat", json={"query": "q3"})
    assert codes == [200, 200]
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) == 10  # one token per 10 s at 6/min, rounded up to whole seconds
    assert len(pipe.queries) == 2


def test_invalid_requests_do_not_spend_rate_limit_tokens():
    pipe = _FakePipeline([_DONE])
    with TestClient(_limited_app(pipe, capacity=1)) as client:
        assert client.post("/chat", json={"query": "   "}).status_code == 422
        assert client.post("/chat", json={"query": "real question"}).status_code == 200


def test_rate_limit_does_not_apply_to_health_or_feedback():
    pipe = _FakePipeline([_DONE])
    with TestClient(_limited_app(pipe, capacity=1)) as client:
        client.post("/chat", json={"query": "q"})
        assert client.post("/chat", json={"query": "q"}).status_code == 429
        assert client.get("/health").status_code == 200
        assert client.post("/feedback", json={"trace_id": "0" * 32, "rating": "up"}).status_code == 404  # unknown trace, not 429
