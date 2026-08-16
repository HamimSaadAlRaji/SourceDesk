"""Iterative retrieval (plan.md Phase 8, step 8): on low confidence,
reformulate the query once and retry -- capped at one retry per plan.md's
explicit wording. The *iterative* agentic workflow pattern (course topic
27), alongside the router's conditional branch (`router/policy.py`) and
multi-query expansion's parallel branch (`query/expand.py`).

Confidence here is a **provisional proxy**: top-1 fused retrieval score
below `settings.query.low_confidence_score_threshold`. Phase 10 hasn't
built the real confidence scorer yet (reranker top score + rank-1/rank-2
gap + inter-chunk agreement + score-distribution shape, plan.md Phase 10
step 1) -- this borrows one cheap piece of that future signal so Phase 8
can measure whether the retry *pattern* helps at all, and is explicitly
superseded once Phase 10 exists (same provisional-then-superseded pattern
Phase 3 used persisting heading-aware chunks before Phase 6 picked a
winner -- see progress.md's decision log, 2026-08-02).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from aegis.config import settings
from aegis.types import Chunk

ReformulateFn = Callable[[str], str]

SYSTEM_PROMPT = """The user's question returned low-confidence search \
results from a FastAPI documentation + issue-tracker search engine. \
Reformulate it to be more specific and search-friendly: add likely \
technical terms, expand any acronyms, and make an implicit subject \
explicit. Preserve the original intent -- don't invent new intent.

Respond with the reformulated question only. No preamble."""


def _default_reformulate_fn(model: str, temperature: float) -> ReformulateFn:
    from groq import APIError, Groq

    client = Groq(api_key=settings.groq.api_key)

    def reformulate(query: str) -> str:
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

    return reformulate


class QueryReformulator:
    def __init__(
        self,
        reformulate_fn: ReformulateFn | None = None,
        model: str = settings.query.rewrite_model,
        temperature: float = settings.query.temperature,
    ) -> None:
        self._reformulate_fn = reformulate_fn or _default_reformulate_fn(model, temperature)

    def reformulate(self, query: str) -> str:
        reformulated = self._reformulate_fn(query).strip()
        return reformulated or query


class _SearchableRetriever(Protocol):
    def search(self, query: str, top_k: int, **kwargs) -> list[tuple[Chunk, float]]: ...


class IterativeRetriever:
    def __init__(
        self,
        base_retriever: _SearchableRetriever,
        reformulator: QueryReformulator,
        score_threshold: float = settings.query.low_confidence_score_threshold,
        max_retries: int = settings.query.iterative_retrieval_max_retries,
        enabled: bool = True,
    ) -> None:
        self.base_retriever = base_retriever
        self.reformulator = reformulator
        self.score_threshold = score_threshold
        self.max_retries = max_retries
        self.enabled = enabled

    def search(self, query: str, top_k: int = 10, **kwargs) -> list[tuple[Chunk, float]]:
        results = self.base_retriever.search(query, top_k=top_k, **kwargs)
        if not self.enabled:
            return results

        current_query = query
        retries = 0
        while retries < self.max_retries and self._is_low_confidence(results):
            current_query = self.reformulator.reformulate(current_query)
            results = self.base_retriever.search(current_query, top_k=top_k, **kwargs)
            retries += 1
        return results

    def _is_low_confidence(self, results: list[tuple[Chunk, float]]) -> bool:
        return not results or results[0][1] < self.score_threshold
