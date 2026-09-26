"""TDD for the multi-provider fallback client (plan.md Phase 9, step 1).
Providers are faked as spies -- no network -- same treatment as
`RouterClassifier`'s `classify_fn` and `HydeGenerator`'s `hyde_fn`.

The one test that matters most here (mirroring `test_router_policy.py`'s
"generation must be structurally unreachable" test) is
`test_propagates_error_when_primary_fails_mid_stream_and_secondary_never_called`
-- it proves a mid-stream failure can never result in two providers'
output getting spliced into one answer.
"""

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
