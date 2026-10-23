"""OpenAI provider (Phase 15 step 0, design record 15g) -- the primary paid
provider. Every call goes through the spend ledger: a worst-case reservation
before the request (refused over the $10 budget, without contacting OpenAI),
then the billed `usage` once the response is complete.

Reasoning models (gpt-5*, o*) reject temperature/top_p and bill hidden
reasoning tokens as output, so they get `max_completion_tokens` (which caps
reasoning + answer together, so the reservation stays a true worst case) and
`reasoning_effort` instead.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import lru_cache

from aegis.config import settings
from aegis.cost.ledger import SpendLedger

_MSG_OVERHEAD_TOKENS = 4  # per message, plus 3 for the reply priming: OpenAI's published counting rule


@lru_cache(maxsize=1)
def default_ledger() -> SpendLedger:
    return SpendLedger(settings.openai.ledger_path, settings.openai.budget_usd, settings.openai.prices)


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("o200k_base")  # the gpt-4.1 / gpt-4o / gpt-5 tokenizer


def count_message_tokens(messages: list[dict[str, str]]) -> int:
    enc = _encoding()
    return sum(len(enc.encode(m["content"])) + _MSG_OVERHEAD_TOKENS for m in messages) + 3


def _is_reasoning(model: str) -> bool:
    return any(model.startswith(p) for p in settings.openai.reasoning_model_prefixes)


def _params(model: str, temperature: float, top_p: float | None, max_tokens: int) -> dict:
    if _is_reasoning(model):
        return {"max_completion_tokens": max_tokens, "reasoning_effort": settings.openai.reasoning_effort}
    params = {"temperature": temperature, "max_completion_tokens": max_tokens}
    if top_p is not None:
        params["top_p"] = top_p
    return params


def _default_client():
    from openai import OpenAI

    return OpenAI(api_key=settings.openai.api_key)


def _settle(ledger: SpendLedger, reservation: str, usage, messages, produced_text: str) -> None:
    if usage is not None:
        details = getattr(usage, "prompt_tokens_details", None)
        ledger.settle(reservation, usage.prompt_tokens, getattr(details, "cached_tokens", 0) or 0, usage.completion_tokens)
    else:  # no usage reported (stream cut short): record a conservative estimate, never nothing
        ledger.settle(reservation, count_message_tokens(messages), 0, len(_encoding().encode(produced_text)))


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str = settings.openai.generation_model, purpose: str = "generation",
                 ledger: SpendLedger | None = None, client_factory: Callable | None = None) -> None:
        self.model = model
        self.purpose = purpose
        self._ledger = ledger
        self._client_factory = client_factory or _default_client

    def stream(self, messages: list[dict[str, str]], temperature: float, top_p: float, max_tokens: int) -> Iterator[str]:
        ledger = self._ledger or default_ledger()
        reservation = ledger.reserve(self.model, count_message_tokens(messages), max_tokens, self.purpose)
        produced: list[str] = []
        usage = None
        try:
            response = self._client_factory().chat.completions.create(
                model=self.model, messages=messages, stream=True, stream_options={"include_usage": True},
                **_params(self.model, temperature, top_p, max_tokens),
            )
            for chunk in response:
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        produced.append(delta)
                        yield delta
        except BaseException:
            if not produced and usage is None:
                ledger.release(reservation)  # failed before any output: nothing billed
            else:
                _settle(ledger, reservation, usage, messages, "".join(produced))
            raise
        _settle(ledger, reservation, usage, messages, "".join(produced))


def openai_chat(model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int, purpose: str,
                json_mode: bool = False, ledger: SpendLedger | None = None,
                client_factory: Callable | None = None) -> str:
    """Non-streaming call for short tasks (router, rewriter, entailment, judge)."""
    ledger = ledger or default_ledger()
    reservation = ledger.reserve(model, count_message_tokens(messages), max_tokens, purpose)
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    try:
        completion = (client_factory or _default_client)().chat.completions.create(
            model=model, messages=messages, **_params(model, temperature, None, max_tokens), **extra)
    except BaseException:
        ledger.release(reservation)
        raise
    text = completion.choices[0].message.content or ""
    _settle(ledger, reservation, getattr(completion, "usage", None), messages, text)
    return text
