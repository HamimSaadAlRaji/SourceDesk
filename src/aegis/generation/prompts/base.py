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


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    system_prompt: str

    def build_messages(self, query: str, context_block: str) -> list[dict[str, str]]:
        user_content = f"Context:\n{context_block}\n\nQuestion: {query}"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
