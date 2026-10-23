"""Suite-wide guards.

No test may reach a paid API (Phase 15 step 0): the first run after OpenAI
became the default provider made ONE real router call ($0.000046, recorded by
the spend ledger) through two tests that only mocked Groq. Every test now gets
an OpenAI client factory that raises, and a throwaway spend ledger, so a test
that forgets to inject a fake fails loudly instead of spending money.
"""

from __future__ import annotations

import pytest


class RealOpenAICallInTest(AssertionError):
    pass


@pytest.fixture(autouse=True)
def _no_real_openai(monkeypatch, tmp_path):
    from aegis.config import settings
    from aegis.cost.ledger import SpendLedger
    from aegis.generation.llm import openai_provider

    def _refuse():
        raise RealOpenAICallInTest("a test tried to construct a real OpenAI client; inject client_factory=")

    monkeypatch.setattr(openai_provider, "_default_client", _refuse)
    ledger = SpendLedger(tmp_path / "test_spend.sqlite", settings.openai.budget_usd, settings.openai.prices)
    monkeypatch.setattr(openai_provider, "default_ledger", lambda: ledger)
