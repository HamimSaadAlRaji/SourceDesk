"""Groq provider (primary, plan.md Phase 9, step 1) -- thin streaming
wrapper over the Groq SDK, shaped to `ProviderClient`. The SDK import is
deferred into `stream()`, same convention as `router/classifier.py` and
`query/hyde.py`'s default LLM-call wrappers.
"""

from __future__ import annotations

from collections.abc import Iterator

from aegis.config import settings


class GroqProvider:
    name = "groq"

    def __init__(self, model: str = settings.groq.model, api_key: str = settings.groq.api_key) -> None:
        self.model = model
        self._api_key = api_key

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Iterator[str]:
        from groq import Groq

        client = Groq(api_key=self._api_key)
        completion = client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            messages=messages,
            stream=True,
        )
        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
