"""Output-side commitment check (Phase 15, design record 15e).

Retrieved chunks are user-written GitHub comments, so anything the model is
told by context can be attacker text. A citation proves a sentence came from
context; it does not make the sentence something the company said. For
money and accounts that distinction is the whole Air Canada failure, so this
is a code path, not a prompt rule (prompt v2 rule 3 already asks for it; this
is what holds when the model doesn't comply).

Deliberately conjunctive -- business term AND user/company framing AND a
commitment AND not a refusal -- because each word alone is common in
technical FastAPI answers ("OAuth2 scopes are granted", "GraphQL
subscriptions", "the API contract"). False-positive rate on real generated
answers: design record 15e §5.
"""

from __future__ import annotations

import re

from aegis.config import settings

_c = settings.commitment
_TERM = re.compile(_c.business_term, re.IGNORECASE)
_FRAMING = re.compile(_c.framing, re.IGNORECASE)
_COMMITMENT = re.compile(_c.commitment, re.IGNORECASE)
_REFUSAL = re.compile(_c.refusal, re.IGNORECASE)


def is_unauthorized_commitment(sentence: str) -> bool:
    return bool(
        _TERM.search(sentence) and _FRAMING.search(sentence) and _COMMITMENT.search(sentence)
        and not _REFUSAL.search(sentence)
    )
