"""One entry point for short, non-streaming LLM calls (router, rewriter,
citation entailment), so each role picks its provider from config instead of
hard-coding an SDK (Phase 15 step 0, design record 15g).

Every provider-side failure -- an SDK API error, or the spend ledger refusing
a call over budget -- surfaces as `LLMCallError` (original as `__cause__`), so
callers keep one fail-safe branch: the router raises RouterUnavailableError,
the rewriter falls back to the raw query, entailment fails closed.
"""

from __future__ import annotations

from collections.abc import Callable

from aegis.config import settings
from aegis.cost.ledger import BudgetExceededError, SpendLedger


class LLMCallError(RuntimeError):
    def __init__(self, provider: str, exc: BaseException) -> None:
        self.provider = provider
        self.status_code = getattr(exc, "status_code", None)
        super().__init__(f"{provider} call failed: {type(exc).__name__}")


def _groq_chat(model, messages, temperature, max_tokens, json_mode) -> str:
    from groq import Groq

    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    completion = Groq(api_key=settings.groq.api_key).chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens, messages=messages, **extra)
    return completion.choices[0].message.content or ""


def chat_completion(provider: str, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int,
                    purpose: str, json_mode: bool = False, ledger: SpendLedger | None = None,
                    client_factory: Callable | None = None) -> str:
    if provider == "openai":
        import openai

        from aegis.generation.llm.openai_provider import openai_chat

        try:
            return openai_chat(model, messages, temperature, max_tokens, purpose, json_mode=json_mode,
                               ledger=ledger, client_factory=client_factory)
        except (openai.APIError, BudgetExceededError) as exc:
            raise LLMCallError(provider, exc) from exc
    if provider == "groq":
        import groq

        try:
            return _groq_chat(model, messages, temperature, max_tokens, json_mode)
        except groq.APIError as exc:
            raise LLMCallError(provider, exc) from exc
    raise ValueError(f"unknown provider {provider!r}")


def resolve_role(provider: str, model: str | None, openai_default: str, groq_default: str) -> tuple[str, str]:
    """`(provider, model)` for a role. No explicit model: the configured
    provider's default for that role. Explicit model: its provider is inferred
    from the name -- every Groq model this project uses is namespaced
    (`qwen/...`, `openai/gpt-oss-...`) and no OpenAI model is -- so experiments
    that pin a Groq model keep running on Groq."""
    if model is None:
        return provider, (openai_default if provider == "openai" else groq_default)
    return ("groq" if "/" in model else "openai"), model
