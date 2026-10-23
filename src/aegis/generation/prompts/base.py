"""Shared shape for versioned prompt templates (plan.md Phase 9, step 3).

A `PromptTemplate` is just a named system prompt plus the (deliberately
trivial) logic for turning a query + a budgeted context block into chat
messages. Versioning lives at the module level (`v1.py`, `v2.py`, ...) so
each version's exact wording is a diffable, reviewable artifact -- and so
the design record's "prompt iteration history with scores" section can
literally point at a git-diffable file per version rather than an inline
string that changed in place.
"""

from __future__ import annotations

from dataclasses import dataclass

from aegis.generation.budgeter import PackedContext, format_context


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    system_prompt: str
    # Whether retrieved text can forge a context block in this version's
    # rendering (design record 15, A3). Only v3+ is resistant.
    forgery_resistant: bool = False

    def messages_and_key(self, query: str, context: PackedContext) -> tuple[list[dict[str, str]], str | None]:
        """What the generation pipeline calls. v1/v2: exactly the Phase 9 prompt
        (no key), so their published numbers still describe what they run."""
        return self.build_messages(query, format_context(context)), None

    def build_messages(self, query: str, context_block: str) -> list[dict[str, str]]:
        user_content = f"Context:\n{context_block}\n\nQuestion: {query}"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
