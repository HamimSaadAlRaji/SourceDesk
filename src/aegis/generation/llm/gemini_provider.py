"""Gemini provider (fallback, plan.md Phase 9, step 1) -- thin streaming
wrapper over `google-genai`, shaped to `ProviderClient`.

Gemini's SDK doesn't accept OpenAI-style `{"role": "system"/"user"/
"assistant"}` message lists directly: the system message becomes
`GenerateContentConfig.system_instruction` (config, not a content turn),
and remaining turns map "user" -> "user", "assistant" -> "model" (Gemini's
own name for the model's turn). Confirmed against the installed
`google-genai==2.18.1` SDK's actual type fields before writing this, not
assumed from memory of an older API shape.

`thinking_budget=0` is not optional: the default model (`gemini-3.5-flash`,
and `gemini-flash-latest`, tested identically) is an extended-thinking
model by default, and a real call at `max_tokens=20` came back with
`finish_reason=MAX_TOKENS` and empty visible text -- the whole budget spent
on hidden `thought` tokens before any answer text. Confirmed directly
against the live API (`docs/explainers/09-generation-prompting-and-
citations.md` §6), not assumed. This is the right default for this
provider's role (a fast fallback for a support-answer task, not multi-step
reasoning) regardless of the token-budget symptom.
"""

from __future__ import annotations

from collections.abc import Iterator

from aegis.config import settings

_ROLE_MAP = {"user": "user", "assistant": "model"}


def _split_system_and_turns(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    system = None
    turns = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            turns.append(m)
    return system, turns


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str = settings.gemini.model, api_key: str = settings.gemini.api_key) -> None:
        self.model = model
        self._api_key = api_key

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Iterator[str]:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        system, turns = _split_system_and_turns(messages)
        contents = [types.Content(role=_ROLE_MAP[m["role"]], parts=[types.Part(text=m["content"])]) for m in turns]
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            system_instruction=system,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        for chunk in client.models.generate_content_stream(model=self.model, contents=contents, config=config):
            if chunk.text:
                yield chunk.text
