"""Attacks on the HTTP surface built in Phase 14 (design records 14b, 14c, 14f,
14g). Fake pipeline and fake audit service throughout: these pin what the
*server* allows, independent of retrieval or generation.

Two different callers are simulated with TestClient's `client=(host, port)`,
which sets the socket peer address the rate limiter keys on, and separate
TestClients, which hold separate cookie jars (= separate browsers).

Before (design record 15): A5a-e and A5g all succeeded (xfail). After (15c,
user decision "anonymous chat + per-browser ownership + /trace behind API
keys"): these assert that policy. A5a's original assertion ("every endpoint
401s anonymous callers") was written before the decision; anonymous /chat is
now intended, so the test asserts the decided policy instead -- recorded in
design record 15c rather than silently rewritten.
"""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from aegis.api.app import create_app
from aegis.api.auth import hash_key
from aegis.api.ratelimit import TokenBucketLimiter
from aegis.api.store import InteractionStore
from aegis.config import ApiKeyGrant, settings
from aegis.eval.review_queue import review_session
from aegis.events import DoneEvent, TokenEvent

_TID = "a" * 32
_ALICE = ("10.0.0.1", 40001)
_MALLORY = ("10.0.0.66", 40066)
_TRACE_KEY = "t" * 43
_PLAIN_KEY = "p" * 43
_ALICE_QUERY = "my staging DB is at postgres://admin:hunter2@10.1.2.3/app, why does Depends hang?"


class _Pipe:
    def __init__(self, events):
        self.events = events

    def warm_up(self):
        return {}

    def answer_stream(self, query, history=None):
        yield from self.events


class _RecordingAudit:
    """Returns the same shape as the real audit service: the trail carries the
    query text, evidence carries full chunk text (audit_service.py)."""

    def __call__(self, trace_id):
        return {"trail": {"trace_id": trace_id, "query": _ALICE_QUERY},
                "evidence": [{"chunk_id": "issue::1::heading_aware::0", "text": "full chunk text"}]}


def _app(tmp_path, audit=None, limiter=None):
    store = InteractionStore(tmp_path / "i.sqlite")
    events = [TokenEvent(text="answer"), DoneEvent(trace_id=_TID, cache_status="miss", escalate=False)]
    return create_app(_Pipe(events), store=store, audit=audit or _RecordingAudit(), limiter=limiter), store


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    monkeypatch.setattr(settings.api, "api_keys", {
        hash_key(_TRACE_KEY): ApiKeyGrant(name="operator", partition="full", can_trace=True),
        hash_key(_PLAIN_KEY): ApiKeyGrant(name="partner", partition="full", can_trace=False),
    })


def _bearer(key):
    return {"Authorization": f"Bearer {key}"}


def test_access_policy(tmp_path):
    """The decided policy (15c): chat is public, the debug trail is operator-only,
    and a bad key is refused -- never treated as anonymous."""
    app, _ = _app(tmp_path)
    with TestClient(app, client=_MALLORY) as c:
        assert c.post("/chat", json={"query": "q"}).status_code == 200
        assert c.get(f"/trace/{_TID}").status_code == 401
        assert c.get(f"/trace/{_TID}", headers=_bearer(_PLAIN_KEY)).status_code == 403
        assert c.get(f"/trace/{_TID}", headers=_bearer(_TRACE_KEY)).status_code == 200
        for r in (c.post("/chat", json={"query": "q"}, headers=_bearer("wrong")),
                  c.get(f"/trace/{_TID}", headers=_bearer("wrong")),
                  c.post("/feedback", json={"trace_id": _TID, "rating": "up"}, headers=_bearer("wrong"))):
            assert r.status_code == 401


def test_chat_issues_an_httponly_strict_ownership_cookie(tmp_path):
    app, _ = _app(tmp_path)
    with TestClient(app, client=_ALICE) as c:
        first = c.post("/chat", json={"query": "q"})
        second = c.post("/chat", json={"query": "q"})
    cookie = first.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "path=/" in cookie
    assert "set-cookie" not in second.headers  # the returning browser keeps its token


def test_trace_is_not_readable_by_another_anonymous_caller(tmp_path):
    app, _ = _app(tmp_path)
    with TestClient(app, client=_ALICE) as alice:
        alice.post("/chat", json={"query": _ALICE_QUERY})
        own = alice.get(f"/trace/{_TID}")
    with TestClient(app, client=_MALLORY) as mallory:
        r = mallory.get(f"/trace/{_TID}")
    assert r.status_code == 401 and _ALICE_QUERY not in r.text
    assert own.status_code == 401  # operator-only: not even the asker reads the debug trail


def test_feedback_only_from_the_caller_that_created_the_trace(tmp_path):
    app, store = _app(tmp_path)
    with TestClient(app, client=_ALICE) as alice:
        alice.post("/chat", json={"query": "q"})
        alice.post("/feedback", json={"trace_id": _TID, "rating": "up"})
    with TestClient(app, client=_MALLORY) as mallory:
        r = mallory.post("/feedback", json={"trace_id": _TID, "rating": "down", "comment": "terrible"})
    assert r.status_code in (403, 404)
    assert store.list_feedback()[0].rating == "up"


def test_spoofed_x_forwarded_for_does_not_reset_the_rate_limit(tmp_path):
    """Defended today (14f keys on the socket peer). Pinned so a future
    trusted-proxy change can't reopen it."""
    app, _ = _app(tmp_path, limiter=TokenBucketLimiter(2, 6.0, max_clients=100))
    with TestClient(app, client=_MALLORY) as c:
        codes = [c.post("/chat", json={"query": f"q{i}"}, headers={"X-Forwarded-For": f"203.0.113.{i}"}).status_code
                 for i in range(4)]
    assert codes == [200, 200, 429, 429]


def test_feedback_is_rate_limited(tmp_path):
    app, _ = _app(tmp_path)
    with TestClient(app, client=_MALLORY) as c:
        c.post("/chat", json={"query": "q"})
        codes = [c.post("/feedback", json={"trace_id": _TID, "rating": "down", "comment": f"spam {i}"}).status_code
                 for i in range(200)]
    assert 429 in codes


class _BlockingAudit:
    """Stands in for a /trace call waiting out Langfuse ingestion (up to
    `trace_max_wait_s` = 20 s in the real service)."""

    def __init__(self):
        self.release = threading.Event()

    def __call__(self, trace_id):
        self.release.wait(timeout=30)
        return None


def test_slow_trace_calls_cannot_starve_other_endpoints(tmp_path):
    """Attacked with a *valid* trace key (a leaked operator key is the worst
    case): anonymous calls no longer reach the slow path at all, so attacking
    anonymously would pass for the wrong reason."""
    audit = _BlockingAudit()
    app, _ = _app(tmp_path, audit=audit)
    health: dict = {}
    answered_while_blocked: dict = {}
    with TestClient(app, client=_MALLORY) as c:
        codes: list[int] = []
        workers = [threading.Thread(target=lambda i=i: codes.append(
            c.get(f"/trace/{i:032x}", headers=_bearer(_TRACE_KEY)).status_code)) for i in range(40)]
        try:
            for w in workers:
                w.start()
            threading.Event().wait(1.0)  # let all 40 occupy pool threads
            probe = threading.Thread(target=lambda: health.update(code=c.get("/health").status_code))
            probe.start()
            probe.join(timeout=2.0)
            # Snapshot *before* releasing: once the /trace calls are released the
            # probe completes, and reading it afterwards measures nothing (the
            # first version of this test did exactly that and falsely XPASSed).
            answered_while_blocked.update(health)
        finally:
            audit.release.set()
            for w in workers:
                w.join(timeout=10)
    assert answered_while_blocked.get("code") == 200, "/health did not answer within 2 s while /trace calls were in flight"
    assert codes.count(503) == 40 - settings.api.trace_max_concurrent


def test_review_queue_neutralizes_terminal_escape_sequences(tmp_path):
    store = InteractionStore(tmp_path / "r.sqlite")
    # Query: a bidi override that would visually reorder the text. Comment: an
    # ANSI clear-screen + window-title escape, and a newline faking a queue line.
    store.record_interaction(_TID, "how do I\u202e evil", "a", escalated=False, cache_status="miss", owner="o")
    store.add_feedback(_TID, "down", "fine\x1b[2J\x1b]0;owned\x07 answer\n[2/2] vote: 👍 up", owner="o")
    printed: list[str] = []
    review_session(store, input_fn=lambda _: "q", print_fn=printed.append)
    assert not any(ch in line for line in printed for ch in ("\x1b", "\x07", "\u202e", "\n"))
    assert any("\\x1b[2J" in line for line in printed)  # shown, escaped, so the reviewer sees the attempt


class _Tagged(_Pipe):
    def __init__(self, tag):
        super().__init__([TokenEvent(text=tag), DoneEvent(trace_id=_TID, cache_status="miss", escalate=False)])


def test_the_principal_selects_the_partition_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.api, "api_keys", {
        hash_key(_PLAIN_KEY): ApiKeyGrant(name="docs-tenant", partition="docs", can_trace=False)})
    app = create_app(pipelines={"full": _Tagged("FULL"), "docs": _Tagged("DOCS")},
                     store=InteractionStore(tmp_path / "i.sqlite"), audit=_RecordingAudit())
    with TestClient(app, client=_ALICE) as c:
        assert "DOCS" in c.post("/chat", json={"query": "q"}, headers=_bearer(_PLAIN_KEY)).text
        assert "FULL" in c.post("/chat", json={"query": "q"}).text


def test_server_refuses_to_start_without_a_pipeline_for_a_configured_partition(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.api, "api_keys", {
        hash_key(_PLAIN_KEY): ApiKeyGrant(name="docs-tenant", partition="docs", can_trace=False)})
    app = create_app(_Tagged("FULL"), store=InteractionStore(tmp_path / "i.sqlite"))
    with pytest.raises(RuntimeError, match="docs"):
        with TestClient(app):
            pass


def test_trace_backend_failure_is_a_502_not_a_500(tmp_path):
    def broken(trace_id):
        raise TimeoutError("langfuse read timed out")

    app, _ = _app(tmp_path, audit=broken)
    with TestClient(app, raise_server_exceptions=False) as c:
        assert c.get(f"/trace/{_TID}", headers=_bearer(_TRACE_KEY)).status_code == 502


# ---- 15d: input boundary --------------------------------------------------------------

_SECRET_Q = "my DATABASE_URL is postgresql://admin:S3cretPass@db.internal/app and alice@example.com gets 500s"


class _CapturingPipe(_Pipe):
    def __init__(self):
        super().__init__([TokenEvent(text="a"), DoneEvent(trace_id=_TID, cache_status="miss", escalate=False)])
        self.seen: list[tuple[str, list]] = []

    def answer_stream(self, query, history=None):
        self.seen.append((query.text, history or []))
        yield from self.events


def test_secrets_never_reach_the_pipeline_or_the_store(tmp_path):
    pipe = _CapturingPipe()
    store = InteractionStore(tmp_path / "i.sqlite")
    app = create_app(pipe, store=store, audit=_RecordingAudit())
    with TestClient(app, client=_ALICE) as c:
        c.post("/chat", json={"query": _SECRET_Q,
                              "history": [{"role": "user", "text": "token ghp_" + "a1B2c3D4e5" * 4}]})
        c.post("/feedback", json={"trace_id": _TID, "rating": "down", "comment": "email me at bob@example.com"})
    text, history = pipe.seen[0]
    everything = text + " ".join(t for _, t in history) + " ".join(
        f"{r.query_text} {r.answer_text} {r.comment}" for r in store.list_feedback())
    for secret in ("S3cretPass", "alice@example.com", "ghp_a1B2", "bob@example.com"):
        assert secret not in everything
    assert "[REDACTED:password]" in text and "[REDACTED:email]" in store.list_feedback()[0].comment
