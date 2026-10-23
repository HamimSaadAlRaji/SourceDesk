"""Query rewriting (plan.md Phase 8, step 4): resolve pronouns and ellipsis
against conversation history so a follow-up like "what about on the free
tier?" becomes a standalone query a retriever (which sees one query string
at a time, with no memory of the conversation) can actually use.

I/O-bound wrapper over the Groq API, same dependency-injection shape as
`aegis.router.classifier.RouterClassifier` -- `tests/test_rewrite.py`
injects a fake `rewrite_fn`, the real Groq path is exercised by
`experiments/08b_query_understanding_ablation.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from aegis.config import settings
from aegis.observability.logging import get_logger, log_event

_log = get_logger("rewrite")

# (role, text) turns, oldest first. Only the user's own prior turns matter
# for resolving "it"/"that"/ellipsis -- assistant turns are included too
# since a pronoun can refer to something the assistant said back.
History = list[tuple[str, str]]
RewriteFn = Callable[[History, str], str]

SYSTEM_PROMPT = """You rewrite a user's follow-up message into a standalone \
question, using the conversation history to resolve any pronoun ("it", \
"that", "this") or ellipsis (an omitted subject, e.g. "what about the free \
tier?"). Preserve the user's actual intent and level of detail -- don't add \
information that isn't implied by the history. If the follow-up is already \
standalone, return it unchanged.

Respond with the rewritten question only. No preamble, no quotes."""


def _default_rewrite_fn(provider: str, model: str, temperature: float) -> RewriteFn:
    from aegis.generation.llm.chat import LLMCallError, chat_completion

    def rewrite(history: History, query: str) -> str:
        history_text = "\n".join(f"{role}: {text}" for role, text in history)
        try:
            return chat_completion(
                provider, model,
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": f"Conversation so far:\n{history_text}\n\nFollow-up: {query}"}],
                temperature=temperature, max_tokens=settings.query.rewrite_max_tokens, purpose="rewrite",
            )
        except LLMCallError as exc:
            # Falling back to the raw query is the right fail-safe, but it must
            # not be silent: the router once hid a quota outage the same way.
            cause = exc.__cause__
            log_event(
                _log, logging.WARNING, "rewrite.api_error", provider=provider,
                error_type=type(cause).__name__, status_code=exc.status_code, detail=str(cause)[:300],
            )
            return ""

    return rewrite


class QueryRewriter:
    def __init__(
        self,
        rewrite_fn: RewriteFn | None = None,
        model: str | None = None,
        temperature: float = settings.query.temperature,
        provider: str = settings.query.rewrite_provider,
    ) -> None:
        if rewrite_fn is None:
            from aegis.generation.llm.chat import resolve_role

            provider, model = resolve_role(provider, model, settings.openai.rewrite_model, settings.query.rewrite_model)
            rewrite_fn = _default_rewrite_fn(provider, model, temperature)
        self._rewrite_fn = rewrite_fn

    def rewrite(self, history: History, query: str) -> str:
        """No history means nothing to resolve against -- skip the LLM call
        entirely rather than asking it to rewrite a query it has no context
        to rewrite. A blank/failed rewrite falls back to the raw query
        (fail safe: worse retrieval on the raw pronoun-laden text is
        recoverable; silently returning an empty query is not)."""
        if not history:
            return query
        rewritten = self._rewrite_fn(history, query).strip()
        if not rewritten:
            log_event(_log, logging.WARNING, "rewrite.blank_output", query_chars=len(query))
        return rewritten or query
