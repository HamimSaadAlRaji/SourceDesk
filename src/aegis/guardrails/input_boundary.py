"""Input boundary (Phase 15, design record 15d): what happens to user text
before anything else sees it.

- `redact`: high-precision secret/PII shapes become `[REDACTED:<kind>]`. Run
  at the HTTP boundary on the query, every history turn and the feedback
  comment, so one call keeps a pasted password or token out of the pipeline,
  both caches, the interactions store, the golden-candidate export, Langfuse
  Cloud and the LLM providers -- all of which saw raw text before.
- `injection_flags`: a regex heuristic that only *flags* (logged, never
  blocks). Design record 15d measures why: cheap rephrasings evade it, so
  blocking on it would be a false sense of security with a real
  false-positive cost; the structural defenses (router hard-route,
  partition, output-side checks) are what must hold.
"""

from __future__ import annotations

import re

from aegis.config import settings

_PII = [(kind, re.compile(p)) for kind, p in settings.input_boundary.pii_patterns.items()]
_URL_PASSWORD = re.compile(settings.input_boundary.url_password_pattern)
_INJECTION = re.compile(settings.input_boundary.injection_pattern, re.IGNORECASE)


def redact(text: str) -> tuple[str, dict[str, int]]:
    """`(redacted_text, {kind: count})`. Idempotent: placeholders match no pattern."""
    counts: dict[str, int] = {}

    def _sub(kind: str):
        def repl(_m: re.Match) -> str:
            counts[kind] = counts.get(kind, 0) + 1
            return f"[REDACTED:{kind}]"
        return repl

    def _url(m: re.Match) -> str:
        if m.group(2) == "[REDACTED:password]":
            return m.group(0)
        counts["url_with_credentials"] = counts.get("url_with_credentials", 0) + 1
        return f"{m.group(1)}:[REDACTED:password]"

    # URL credentials first: `user:pass@host.tld` otherwise reads as an email
    # address `pass@host.tld` and the redaction eats the host (caught by
    # test_url_credentials_keep_user_and_host_but_lose_the_password).
    text = _URL_PASSWORD.sub(_url, text)
    for kind, pattern in _PII:
        text = pattern.sub(_sub(kind), text)
    return text, counts


def injection_flags(text: str) -> list[str]:
    """The matched phrases, lower-cased; empty when nothing matched."""
    return sorted({m.group(0).lower() for m in _INJECTION.finditer(text)})
