"""Multi-query expansion (plan.md Phase 8, step 5): generate 2-3 paraphrases
of the query, retrieve each independently, fuse the ranked lists with RRF.
The *parallel* agentic workflow (course topic 27, plan.md Module 4) --
several independent retrieval attempts run concurrently and combined, as
opposed to the router's *conditional* branch or iterative retrieval's
*sequential* retry loop.

`QueryExpander` is the I/O-bound LLM wrapper (dependency-injected, tested
with a fake `expand_fn`). `ExpandedRetriever` is pure composition over any
`_SearchableRetriever` + `QueryExpander`, the same shape as
`aegis.rerank.pipeline.RerankedRetriever` -- toggleable, same interface,
same reasoning for reusing `reciprocal_rank_fusion` rather than a new
fusion rule (Phase 5 already built and measured it; a second one here would
just be an untested duplicate).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from aegis.config import settings
from aegis.retrieval.fusion import reciprocal_rank_fusion
from aegis.types import Chunk

ExpandFn = Callable[[str, int], list[str]]

SYSTEM_PROMPT = """Generate {n} alternative phrasings of the user's question \
that preserve its exact meaning but vary the wording, structure, or level \
of technical detail -- the goal is to surface documents a single phrasing \
might miss. Do not answer the question. Do not include the original \
phrasing.

Respond with exactly {n} lines, one paraphrase per line, no numbering, no \
extra commentary."""


def _default_expand_fn(model: str, temperature: float) -> ExpandFn:
    from groq import APIError, Groq

    client = Groq(api_key=settings.groq.api_key)

    def expand(query: str, n: int) -> list[str]:
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(n=n)},
                    {"role": "user", "content": query},
                ],
            )
        except APIError:
            return []
        text = completion.choices[0].message.content or ""
        lines = [line.strip("-* \t") for line in text.splitlines() if line.strip()]
        return lines[:n]

    return expand


class QueryExpander:
    def __init__(
        self,
        expand_fn: ExpandFn | None = None,
        model: str | None = None,
        temperature: float | None = None,
        n: int | None = None,
    ) -> None:
        self.n = n if n is not None else settings.query.n_expansions
        self._expand_fn = expand_fn or _default_expand_fn(
            model or settings.query.expansion_model, temperature if temperature is not None else settings.query.temperature
        )

    def expand(self, query: str) -> list[str]:
        """Returns `[query] + up to n paraphrases`, deduplicated, original
        always first and always present -- a failed/empty LLM call degrades
        to exactly the no-expansion baseline rather than an empty result
        set."""
        paraphrases = self._expand_fn(query, self.n)
        seen = {query}
        result = [query]
        for raw in paraphrases:
            p = raw.strip()
            if p and p not in seen:
                seen.add(p)
                result.append(p)
        return result


class _SearchableRetriever(Protocol):
    def search(self, query: str, top_k: int, **kwargs) -> list[tuple[Chunk, float]]: ...


class ExpandedRetriever:
    def __init__(
        self,
        base_retriever: _SearchableRetriever,
        expander: QueryExpander,
        enabled: bool = True,
    ) -> None:
        self.base_retriever = base_retriever
        self.expander = expander
        self.enabled = enabled

    def search(self, query: str, top_k: int = 10, **kwargs) -> list[tuple[Chunk, float]]:
        if not self.enabled:
            return self.base_retriever.search(query, top_k=top_k, **kwargs)

        queries = self.expander.expand(query)
        ranked_lists = [self.base_retriever.search(q, top_k=top_k, **kwargs) for q in queries]
        fused = reciprocal_rank_fusion(ranked_lists)
        return fused[:top_k]
