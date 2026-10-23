"""TDD for the OpenAI provider and the small-call dispatcher (Phase 15 step 0,
design record 15g). A fake client stands in for the SDK, so these pin what
we send, what we record, and when we refuse -- without spending anything.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from aegis.cost.ledger import BudgetExceededError, SpendLedger
from aegis.generation.llm.chat import LLMCallError, chat_completion
from aegis.generation.llm.fallback import FallbackLLMClient
from aegis.generation.llm.openai_provider import OpenAIProvider, openai_chat

PRICES = {"gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
          "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00}}
MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "how do I use Depends?"}]


def _usage(prompt=120, cached=20, completion=30):
    return NS(prompt_tokens=prompt, completion_tokens=completion,
              prompt_tokens_details=NS(cached_tokens=cached))


class _FakeClient:
    def __init__(self, tokens=("Use ", "Depends [1]."), fail=None):
        self.tokens, self.fail, self.calls = tokens, fail, []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise self.fail
        if kw.get("stream"):
            chunks = [NS(choices=[NS(delta=NS(content=t))], usage=None) for t in self.tokens]
            return iter(chunks + [NS(choices=[], usage=_usage())])
        return NS(choices=[NS(message=NS(content="".join(self.tokens)))], usage=_usage())


@pytest.fixture
def ledger(tmp_path):
    return SpendLedger(tmp_path / "s.sqlite", budget_usd=10.0, prices=PRICES)


def test_stream_yields_tokens_and_settles_billed_usage(ledger):
    client = _FakeClient()
    p = OpenAIProvider("gpt-4.1-mini", purpose="generation", ledger=ledger, client_factory=lambda: client)
    assert "".join(p.stream(MSGS, temperature=0.0, top_p=1.0, max_tokens=400)) == "Use Depends [1]."
    assert client.calls[0]["stream_options"] == {"include_usage": True}
    assert ledger.spent_usd() == pytest.approx(ledger.cost("gpt-4.1-mini", 120, 20, 30))
    assert ledger.reserved_usd() == 0.0
    assert ledger.summary()["by_purpose"] == {"generation": pytest.approx(ledger.spent_usd())}


def test_over_budget_refuses_before_contacting_the_provider(tmp_path):
    tiny = SpendLedger(tmp_path / "t.sqlite", budget_usd=0.0001, prices=PRICES)
    client = _FakeClient()
    p = OpenAIProvider("gpt-4.1-mini", purpose="generation", ledger=tiny, client_factory=lambda: client)
    with pytest.raises(BudgetExceededError):
        next(p.stream(MSGS, temperature=0.0, top_p=1.0, max_tokens=400))
    assert client.calls == []


def test_budget_refusal_falls_back_to_the_next_provider(tmp_path):
    tiny = SpendLedger(tmp_path / "t.sqlite", budget_usd=0.0001, prices=PRICES)

    class _Free:
        name = "groq"

        def stream(self, messages, temperature, top_p, max_tokens):
            yield "free answer"

    paid = OpenAIProvider("gpt-4.1-mini", purpose="generation", ledger=tiny, client_factory=_FakeClient)
    text, provider = FallbackLLMClient([paid, _Free()]).generate_with_provider(MSGS, 0.0, 1.0, 400)
    assert (text, provider) == ("free answer", "groq")


def test_a_failed_call_releases_its_reservation(ledger):
    p = OpenAIProvider("gpt-4.1-mini", purpose="generation", ledger=ledger,
                       client_factory=lambda: _FakeClient(fail=RuntimeError("503")))
    with pytest.raises(RuntimeError):
        next(p.stream(MSGS, temperature=0.0, top_p=1.0, max_tokens=400))
    assert ledger.reserved_usd() == 0.0 and ledger.spent_usd() == 0.0


def test_reasoning_models_get_no_sampling_params_and_a_completion_cap(ledger):
    client = _FakeClient()
    openai_chat("gpt-5-mini", MSGS, temperature=0.0, max_tokens=300, purpose="judge", ledger=ledger,
                client_factory=lambda: client)
    sent = client.calls[0]
    assert "temperature" not in sent and "top_p" not in sent
    assert sent["max_completion_tokens"] == 300 and "reasoning_effort" in sent


def test_non_reasoning_models_get_temperature(ledger):
    client = _FakeClient()
    openai_chat("gpt-4.1-mini", MSGS, temperature=0.0, max_tokens=50, purpose="router", ledger=ledger,
                client_factory=lambda: client, json_mode=True)
    sent = client.calls[0]
    assert sent["temperature"] == 0.0 and sent["max_completion_tokens"] == 50
    assert sent["response_format"] == {"type": "json_object"}


def test_dispatch_wraps_provider_and_budget_errors(tmp_path):
    tiny = SpendLedger(tmp_path / "t.sqlite", budget_usd=0.0001, prices=PRICES)
    with pytest.raises(LLMCallError) as e:
        # worst case 500 output tokens x $1.60/1M = $0.0008 > the $0.0001 budget
        chat_completion("openai", "gpt-4.1-mini", MSGS, temperature=0.0, max_tokens=500, purpose="router",
                        ledger=tiny, client_factory=_FakeClient)
    assert isinstance(e.value.__cause__, BudgetExceededError)


def test_dispatch_rejects_unknown_providers():
    with pytest.raises(ValueError):
        chat_completion("mystery", "m", MSGS, temperature=0.0, max_tokens=5, purpose="x")
