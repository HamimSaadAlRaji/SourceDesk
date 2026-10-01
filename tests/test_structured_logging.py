"""TDD for structured logging (Phase 13 build item 2): JSON records with a
trace ID threaded through every line of one request. The router-failure test
is the motivating case -- a Groq quota outage silently became "escalate every
query" with no log line anywhere (found 2026-09-30)."""

import io
import json
import logging

import httpx
import pytest

from aegis.observability.logging import (
    JsonFormatter,
    bind_trace_id,
    get_logger,
    log_event,
)


def _capture(logger_name: str) -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(logger_name)
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, stream


def _records(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_record_is_valid_json_with_standard_and_custom_fields():
    logger, stream = _capture("aegis.test.a")

    log_event(logger, logging.INFO, "cache.lookup", cache_status="miss", corpus_version="v1")

    (rec,) = _records(stream)
    assert rec["event"] == "cache.lookup"
    assert rec["level"] == "INFO"
    assert rec["logger"] == "aegis.test.a"
    assert rec["cache_status"] == "miss" and rec["corpus_version"] == "v1"
    assert "ts" in rec


def test_trace_id_is_attached_inside_bind_and_absent_outside():
    logger, stream = _capture("aegis.test.b")

    log_event(logger, logging.INFO, "before")
    with bind_trace_id("trace-abc"):
        log_event(logger, logging.INFO, "inside")
    log_event(logger, logging.INFO, "after")

    before, inside, after = _records(stream)
    assert "trace_id" not in before and "trace_id" not in after
    assert inside["trace_id"] == "trace-abc"


def test_bind_trace_id_restores_the_previous_value_when_nested():
    logger, stream = _capture("aegis.test.c")

    with bind_trace_id("outer"):
        with bind_trace_id("inner"):
            log_event(logger, logging.INFO, "a")
        log_event(logger, logging.INFO, "b")

    a, b = _records(stream)
    assert a["trace_id"] == "inner" and b["trace_id"] == "outer"


def test_non_json_serializable_field_values_do_not_break_logging():
    logger, stream = _capture("aegis.test.d")

    log_event(logger, logging.INFO, "odd", value=object())

    (rec,) = _records(stream)
    assert rec["event"] == "odd" and isinstance(rec["value"], str)


def test_get_logger_is_namespaced_under_aegis():
    assert get_logger("pipeline").name == "aegis.pipeline"


def test_router_api_failure_is_logged_not_silently_swallowed(monkeypatch):
    import groq

    from aegis.router.classifier import RouterClassifier, Route

    class _FakeCompletions:
        def create(self, **kwargs):
            raise groq.APIError("quota exhausted", httpx.Request("POST", "http://x"), body=None)

    class _FakeGroq:
        def __init__(self, api_key=None):
            self.chat = type("C", (), {"completions": _FakeCompletions()})()

    monkeypatch.setattr(groq, "Groq", _FakeGroq)
    logger, stream = _capture("aegis.router")

    with bind_trace_id("trace-router"):
        route = RouterClassifier().classify("how do I declare a query parameter")

    assert route == Route.BILLING_ACCOUNT_LEGAL  # fail-safe behavior unchanged
    events = [r for r in _records(stream) if r["event"] == "router.api_error"]
    assert len(events) == 1
    assert events[0]["level"] == "WARNING"
    assert events[0]["trace_id"] == "trace-router"
    assert events[0]["error_type"] == "APIError"


def test_router_unparseable_output_is_logged_with_a_snippet_and_still_fails_safe():
    from aegis.router.classifier import Route, RouterClassifier

    logger, stream = _capture("aegis.router")

    route = RouterClassifier(classify_fn=lambda q: "not json at all").classify("some query")

    assert route == Route.BILLING_ACCOUNT_LEGAL
    (rec,) = [r for r in _records(stream) if r["event"] == "router.unparseable_output"]
    assert rec["level"] == "WARNING" and rec["raw"] == "not json at all"


def test_router_empty_output_after_an_api_error_is_not_double_logged():
    from aegis.router.classifier import RouterClassifier

    logger, stream = _capture("aegis.router")

    RouterClassifier(classify_fn=lambda q: "").classify("some query")

    assert [r["event"] for r in _records(stream)] == []  # the API-error path already logged; "" is not re-reported
