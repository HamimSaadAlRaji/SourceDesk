from __future__ import annotations

from aegis.config import settings
from aegis.generation.llm.fallback import AllProvidersFailedError, FallbackLLMClient
from aegis.generation.llm.gemini_provider import GeminiProvider
from aegis.generation.llm.groq_provider import GroqProvider
from aegis.generation.llm.openai_provider import OpenAIProvider

_PROVIDER_FACTORIES = {"openai": OpenAIProvider, "groq": GroqProvider, "gemini": GeminiProvider}


def default_client(provider_order: list[str] | None = None) -> FallbackLLMClient:
    order = provider_order or settings.generation.provider_order
    providers = [_PROVIDER_FACTORIES[name]() for name in order]
    return FallbackLLMClient(providers)


__all__ = [
    "AllProvidersFailedError",
    "FallbackLLMClient",
    "GeminiProvider",
    "GroqProvider",
    "OpenAIProvider",
    "default_client",
]
