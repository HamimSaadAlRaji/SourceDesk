"""HyDE -- Hypothetical Document Embeddings (plan.md Phase 8, step 6):
generate a hypothetical answer to the query with an LLM, then embed *that*
text for dense retrieval instead of the raw query.

Works because dense retrieval's cosine similarity measures how much two
pieces of text sit in the same region of embedding space, and a question
and its answer are different genres of text (a question, an explanation)
even when topically identical -- two answers to the same underlying
question are much closer to each other in that space than a question is to
its own answer. Generating a plausible-sounding (possibly wrong, that's
fine -- it's never shown to the user or cited) answer and embedding *that*
searches from the answer's neighborhood instead of the question's.

`HydeGenerator` is the I/O-bound LLM wrapper. `HydeRetriever` composes it
with any `_SearchableRetriever` (`.search(query, top_k, **kwargs)` already
takes a raw string and embeds it internally -- see
`aegis.retrieval.dense.DenseIndex.search` -- so no interface change is
needed; the hypothetical answer text is just passed as the query).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from aegis.config import settings
from aegis.types import Chunk

HydeFn = Callable[[str], str]

SYSTEM_PROMPT = """Write a short, plausible-sounding paragraph that answers \
the user's question about FastAPI, as if it were an excerpt from \
documentation or a resolved GitHub issue. It does not need to be factually \
correct -- it will never be shown to a user -- it only needs to read like \
the kind of text that would actually answer this question, using realistic \
FastAPI terminology (parameter names, decorators, exception classes).

Respond with the paragraph only. No preamble."""


def _default_hyde_fn(model: str, temperature: float) -> HydeFn:
    from groq import APIError, Groq

    client = Groq(api_key=settings.groq.api_key)

    def hyde(query: str) -> str:
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
            )
        except APIError:
            return ""
        return completion.choices[0].message.content or ""

    return hyde


class HydeGenerator:
    def __init__(
        self,
        hyde_fn: HydeFn | None = None,
        model: str = settings.query.hyde_model,
        temperature: float = settings.query.temperature,
    ) -> None:
        self._hyde_fn = hyde_fn or _default_hyde_fn(model, temperature)

    def generate(self, query: str) -> str:
        """Falls back to the raw query on a failed/empty generation --
        degrades to plain dense retrieval on the original text rather than
        embedding an empty string."""
        hypothetical = self._hyde_fn(query).strip()
        return hypothetical or query


class _SearchableRetriever(Protocol):
    def search(self, query: str, top_k: int, **kwargs) -> list[tuple[Chunk, float]]: ...


class HydeRetriever:
    def __init__(
        self,
        base_retriever: _SearchableRetriever,
        generator: HydeGenerator,
        enabled: bool = True,
    ) -> None:
        self.base_retriever = base_retriever
        self.generator = generator
        self.enabled = enabled

    def search(self, query: str, top_k: int = 10, **kwargs) -> list[tuple[Chunk, float]]:
        if not self.enabled:
            return self.base_retriever.search(query, top_k=top_k, **kwargs)

        hypothetical = self.generator.generate(query)
        return self.base_retriever.search(hypothetical, top_k=top_k, **kwargs)
