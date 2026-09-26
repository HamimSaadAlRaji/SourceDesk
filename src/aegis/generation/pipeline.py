"""Composes the context budgeter, a versioned prompt template, the
multi-provider LLM client, and the citation validator into one
`.answer(query, retrieved)` call -- the object Phase 9's own experiments
(sampling sweep, provider bake-off) and Phase 10's abstention gate both
call. Pure composition, same shape as `RerankedRetriever`/`HydeRetriever`:
every collaborator is dependency-injected, so this is unit-testable with
fakes (`tests/test_generation_pipeline.py`); the real path is exercised by
`experiments/09*.py`.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from aegis.config import settings
from aegis.generation.budgeter import PackedContext, format_context, pack_context
from aegis.generation.prompts import PromptTemplate, get_prompt
from aegis.guardrails.citations import CitationValidator, ClaimStatus, ValidationResult
from aegis.types import Query, Retrieved


class _LLMClient(Protocol):
    def generate_with_provider(
        self, messages: list[dict[str, str]], temperature: float, top_p: float, max_tokens: int
    ) -> tuple[str, str | None]: ...


class GenerationResult(BaseModel):
    query: Query
    text: str
    context: PackedContext
    validation: ValidationResult
    provider: str | None
    citations: list[str]  # chunk ids actually cited *and* validated as supported


class GenerationPipeline:
    def __init__(
        self,
        llm: _LLMClient,
        validator: CitationValidator,
        prompt: PromptTemplate | None = None,
        budget_tokens: int = settings.generation.context_budget_tokens,
        temperature: float = settings.groq.temperature,
        top_p: float = settings.groq.top_p,
        max_tokens: int = settings.groq.max_tokens,
    ) -> None:
        self.llm = llm
        self.validator = validator
        self.prompt = prompt or get_prompt(settings.generation.prompt_version)
        self.budget_tokens = budget_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def answer(self, query: Query, retrieved: list[Retrieved]) -> GenerationResult:
        context = pack_context(retrieved, self.budget_tokens)
        context_block = format_context(context)
        messages = self.prompt.build_messages(query.text, context_block)

        text, provider = self.llm.generate_with_provider(
            messages, temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens
        )

        validation = self.validator.validate(text, context)
        citations = _supported_citation_chunk_ids(context, validation)

        return GenerationResult(
            query=query, text=text, context=context, validation=validation, provider=provider, citations=citations
        )


_SUPPORTED = {ClaimStatus.SUPPORTED_LEXICAL, ClaimStatus.SUPPORTED_ENTAILMENT}


def _supported_citation_chunk_ids(context: PackedContext, validation: ValidationResult) -> list[str]:
    citation_to_chunk_id = {pc.citation_id: pc.chunk.id for pc in context.chunks}
    return sorted(
        {
            citation_to_chunk_id[cid]
            for claim in validation.claims
            if claim.status in _SUPPORTED
            for cid in claim.citation_ids
        }
    )
