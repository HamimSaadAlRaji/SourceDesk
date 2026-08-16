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

from collections.abc import Callable

from aegis.config import settings

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


def _default_rewrite_fn(model: str, temperature: float) -> RewriteFn:
    from groq import APIError, Groq

    client = Groq(api_key=settings.groq.api_key)

    def rewrite(history: History, query: str) -> str:
        history_text = "\n".join(f"{role}: {text}" for role, text in history)
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Conversation so far:\n{history_text}\n\nFollow-up: {query}"},
                ],
            )
        except APIError:
            return ""
        return completion.choices[0].message.content or ""

    return rewrite


class QueryRewriter:
    def __init__(
        self,
        rewrite_fn: RewriteFn | None = None,
        model: str = settings.query.rewrite_model,
        temperature: float = settings.query.temperature,
    ) -> None:
        self._rewrite_fn = rewrite_fn or _default_rewrite_fn(model, temperature)

    def rewrite(self, history: History, query: str) -> str:
        """No history means nothing to resolve against -- skip the LLM call
        entirely rather than asking it to rewrite a query it has no context
        to rewrite. A blank/failed rewrite falls back to the raw query
        (fail safe: worse retrieval on the raw pronoun-laden text is
        recoverable; silently returning an empty query is not)."""
        if not history:
            return query
        rewritten = self._rewrite_fn(history, query).strip()
        return rewritten or query
