"""TDD for the multi-provider fallback client (plan.md Phase 9, step 1).
Providers are faked as spies -- no network -- same treatment as
`RouterClassifier`'s `classify_fn` and `HydeGenerator`'s `hyde_fn`.

The one test that matters most here (mirroring `test_router_policy.py`'s
"generation must be structurally unreachable" test) is
`test_propagates_error_when_primary_fails_mid_stream_and_secondary_never_called`
-- it proves a mid-stream failure can never result in two providers'
output getting spliced into one answer.
"""

import logging

import pytest

from aegis.generation.llm.fallback import AllProvidersFailedError, FallbackLLMClient


class _FakeProvider:
    def __init__(self, name: str, chunks: list[str] | None = None, fail_before_first: bool = False, fail_after: int | None = None):
        self.name = name
        self.chunks = chunks or []
        self.fail_before_first = fail_before_first
        self.fail_after = fail_after
        self.call_count = 0

    def stream(self, messages, temperature, top_p, max_tokens):
        self.call_count += 1
        if self.fail_before_first:
            raise RuntimeError(f"{self.name} failed before first token")
        for i, chunk in enumerate(self.chunks):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError(f"{self.name} failed mid-stream")
            yield chunk


def _messages():
    return [{"role": "user", "content": "hello"}]


def test_uses_primary_when_it_succeeds_and_never_calls_secondary():
    primary = _FakeProvider("primary", chunks=["hello", " world"])
    secondary = _FakeProvider("secondary", chunks=["should never appear"])
    client = FallbackLLMClient([primary, secondary])

    result = "".join(client.stream(_messages()))

    assert result == "hello world"
    assert primary.call_count == 1
    assert secondary.call_count == 0


def test_falls_back_when_primary_fails_before_first_token():
    primary = _FakeProvider("primary", fail_before_first=True)
    secondary = _FakeProvider("secondary", chunks=["fallback", " answer"])
    client = FallbackLLMClient([primary, secondary])

    result = "".join(client.stream(_messages()))

    assert result == "fallback answer"
    assert primary.call_count == 1
    assert secondary.call_count == 1


def test_falls_back_through_multiple_providers_in_order():
    a = _FakeProvider("a", fail_before_first=True)
    b = _FakeProvider("b", fail_before_first=True)
    c = _FakeProvider("c", chunks=["c answered"])
    client = FallbackLLMClient([a, b, c])

    result = "".join(client.stream(_messages()))

    assert result == "c answered"
    assert a.call_count == 1
    assert b.call_count == 1
    assert c.call_count == 1


def test_propagates_error_when_primary_fails_mid_stream_and_secondary_never_called():
    primary = _FakeProvider("primary", chunks=["partial ", "output"], fail_after=1)
    secondary = _FakeProvider("secondary", chunks=["should never run"])
    client = FallbackLLMClient([primary, secondary])

    with pytest.raises(RuntimeError, match="failed mid-stream"):
        list(client.stream(_messages()))

    assert secondary.call_count == 0


def test_raises_all_providers_failed_when_every_provider_fails_before_first_token():
    a = _FakeProvider("a", fail_before_first=True)
    b = _FakeProvider("b", fail_before_first=True)
    client = FallbackLLMClient([a, b])

    with pytest.raises(AllProvidersFailedError) as exc_info:
        list(client.stream(_messages()))

    names = [name for name, _ in exc_info.value.attempts]
    assert names == ["a", "b"]


def test_empty_stream_from_primary_is_a_success_not_a_failure():
    primary = _FakeProvider("primary", chunks=[])
    secondary = _FakeProvider("secondary", chunks=["should not run"])
    client = FallbackLLMClient([primary, secondary])

    result = "".join(client.stream(_messages()))

    assert result == ""
    assert secondary.call_count == 0


def test_generate_joins_the_stream_into_one_string():
    primary = _FakeProvider("primary", chunks=["a", "b", "c"])
    client = FallbackLLMClient([primary])

    assert client.generate(_messages()) == "abc"


def test_constructor_rejects_an_empty_provider_list():
    with pytest.raises(ValueError, match="at least one provider"):
        FallbackLLMClient([])


def test_generate_with_provider_reports_the_winning_provider_name():
    primary = _FakeProvider("primary", fail_before_first=True)
    secondary = _FakeProvider("secondary", chunks=["it was ", "me"])
    client = FallbackLLMClient([primary, secondary])

    text, provider_name = client.generate_with_provider(_messages())

    assert text == "it was me"
    assert provider_name == "secondary"


def test_generate_with_provider_is_none_when_the_winning_stream_is_empty():
    primary = _FakeProvider("primary", chunks=[])
    client = FallbackLLMClient([primary])

    text, provider_name = client.generate_with_provider(_messages())

    assert text == ""
    assert provider_name is None


# --- Phase 14 build item 6: failover must be visible (design record 14f) ---
# The cold-browser run in design record 14e was answered by Gemini and nothing
# recorded why Groq was skipped. Every failed attempt and every failover is a
# structured log event; the client's behavior is unchanged.


def _events(caplog, name: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == name]


def test_failed_attempt_is_logged_with_provider_and_error_type(caplog):
    primary = _FakeProvider("primary", fail_before_first=True)
    secondary = _FakeProvider("secondary", chunks=["ok"])
    client = FallbackLLMClient([primary, secondary])

    with caplog.at_level(logging.WARNING, logger="aegis"):
        "".join(client.stream(_messages()))

    (rec,) = _events(caplog, "llm.provider_failed")
    fields = rec.aegis_fields
    assert fields["provider"] == "primary" and fields["error_type"] == "RuntimeError"
    assert fields["attempt"] == 1 and fields["elapsed_ms"] >= 0


def test_failover_names_the_provider_that_answered_and_the_ones_skipped(caplog):
    primary = _FakeProvider("primary", fail_before_first=True)
    secondary = _FakeProvider("secondary", chunks=["ok"])
    client = FallbackLLMClient([primary, secondary])

    with caplog.at_level(logging.WARNING, logger="aegis"):
        "".join(client.stream(_messages()))

    (rec,) = _events(caplog, "llm.failover")
    assert rec.aegis_fields["provider"] == "secondary" and rec.aegis_fields["skipped"] == ["primary"]


def test_primary_success_logs_nothing_at_warning(caplog):
    client = FallbackLLMClient([_FakeProvider("primary", chunks=["ok"])])

    with caplog.at_level(logging.WARNING, logger="aegis"):
        "".join(client.stream(_messages()))

    assert caplog.records == []


def test_all_failed_logs_each_attempt_then_raises(caplog):
    client = FallbackLLMClient([_FakeProvider("a", fail_before_first=True), _FakeProvider("b", fail_before_first=True)])

    with caplog.at_level(logging.WARNING, logger="aegis"), pytest.raises(AllProvidersFailedError):
        "".join(client.stream(_messages()))

    assert [r.aegis_fields["provider"] for r in _events(caplog, "llm.provider_failed")] == ["a", "b"]
    assert _events(caplog, "llm.failover") == []
