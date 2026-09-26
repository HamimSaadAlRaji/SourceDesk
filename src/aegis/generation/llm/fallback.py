"""Multi-provider LLM client with fallback (plan.md Phase 9, step 1): Groq
primary, Gemini secondary, tried in `GenerationSettings.provider_order`.
Ollama stays documented as a third, optional provider (plan.md's own
wording) but isn't built -- no local Ollama install on this machine to
exercise a real call path against, same "design-record-only" treatment
plan.md already gives ColBERT and LLM-based chunking elsewhere in this
project.

`.generate()` is just `"".join(.stream())` -- one streaming interface per
plan.md step 1, so fallback logic only has to be correct once, not
duplicated between a streaming and non-streaming path.

Fallback boundary is deliberately narrow: a provider is only skipped in
favor of the next one if it fails *before* yielding its first token (auth
errors, rate limits, and most model-unavailable errors surface at request
start, not mid-stream -- this is also exactly the failure plan.md's own
risk table names: "Groq rate limits during eval | 429s mid-sweep"). A
failure *after* some output has already been produced propagates instead
of silently switching providers -- splicing a second provider's fresh
generation onto a first provider's partial output would produce an answer
no single model actually wrote, undermining the citation-grounding premise
this whole phase exists for.
"""

from __future__ import annotations

from collections.abc import Iterator

from aegis.generation.llm.base import ProviderClient


class AllProvidersFailedError(Exception):
    def __init__(self, attempts: list[tuple[str, Exception]]) -> None:
        self.attempts = attempts
        summary = "; ".join(f"{name}: {exc!r}" for name, exc in attempts)
        super().__init__(f"All providers failed before producing any output: {summary}")


class FallbackLLMClient:
    def __init__(self, providers: list[ProviderClient]) -> None:
        if not providers:
            raise ValueError("FallbackLLMClient needs at least one provider")
        self.providers = providers

    def _stream_with_provider(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Iterator[tuple[str, str]]:
        """Shared core for `stream`/`generate`/`generate_with_provider` --
        yields `(provider_name, token)` pairs so callers that care which
        provider actually answered (the bake-off script, future request
        logging) can get that without the client holding any mutable
        per-call state of its own (no `self.last_provider` field -- Phase
        6's `EmbeddingCache` thread-safety bug is exactly the failure mode
        that pattern invites on a shared instance)."""
        attempts: list[tuple[str, Exception]] = []

        for provider in self.providers:
            gen = provider.stream(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            try:
                first = next(gen)
            except StopIteration:
                return  # provider produced a well-formed, merely empty response -- not a failure
            except Exception as exc:  # noqa: BLE001 -- deliberately broad: any provider SDK's own exception type
                attempts.append((provider.name, exc))
                continue

            # First token obtained -- this provider is committed to for the
            # rest of the stream. A later exception propagates rather than
            # triggering fallback (see module docstring).
            yield provider.name, first
            for token in gen:
                yield provider.name, token
            return

        raise AllProvidersFailedError(attempts)

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        for _, token in self._stream_with_provider(messages, temperature, top_p, max_tokens):
            yield token

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
    ) -> str:
        return "".join(self.stream(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens))

    def generate_with_provider(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
    ) -> tuple[str, str | None]:
        """Returns `(text, provider_name)` -- `provider_name` is `None`
        only when the winning provider's stream was empty (a well-formed
        but content-free response)."""
        provider_name: str | None = None
        parts: list[str] = []
        for name, token in self._stream_with_provider(messages, temperature, top_p, max_tokens):
            provider_name = name
            parts.append(token)
        return "".join(parts), provider_name
