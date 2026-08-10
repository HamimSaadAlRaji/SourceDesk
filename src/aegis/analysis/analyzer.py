"""The custom analyzer (plan.md Phase 5, step 1).

Off-the-shelf tokenizers destroy exactly the tokens a support query depends
on: `status_code` gets shredded into noise, `app.get` gets read as two
sentences, `HTTP_401_UNAUTHORIZED` gets case-folded into something no
longer greppable, and `3.12` looks like two unrelated integers. This is
the one component that determines whether lexical (BM25) retrieval works
at all on this corpus -- see docs/explainers/05-lexical-dense-and-fusion.md
for the derivation and the measured effect.

Rule, uniformly applied: a compound identifier (snake_case, a dotted path,
a SCREAMING_CASE constant, a hyphenated flag) is emitted whole -- so an
exact-identifier query still matches -- *and* decomposed into its parts --
so a natural-language query ("response model", "cache dir") also matches.
The one exception is version/IP-style numeric sequences (`0.100.0`,
`127.0.0.1`), which are emitted whole only: their "parts" are bare integers
with no standalone meaning, and splitting them would just add noise.
"""

from __future__ import annotations

import re

# First character must be alphanumeric, or the run may start with one or
# two dashes (a CLI flag). Continuation characters allow the punctuation
# that makes compound identifiers meaningful: `_` (snake_case), `.`
# (dotted paths, versions), `-` (multi-word flags).
_TOKEN_RE = re.compile(r"-{0,2}[A-Za-z0-9][A-Za-z0-9_.\-]*")

# Trailing sentence punctuation that the continuation charset above would
# otherwise swallow (e.g. "...on 3.8.10." must not keep the final period).
_TRAILING_PUNCT_RE = re.compile(r"[.,;:!?]+$")

# v?<digits>(.<digits>){1,3} -- covers "3.12", "0.100.0", "127.0.0.1".
_VERSION_RE = re.compile(r"^v?\d+(\.\d+){1,3}$", re.IGNORECASE)

_SPLIT_RE = re.compile(r"[_.\-]+")


def analyze(text: str) -> list[str]:
    """Tokenize `text` into lowercase terms for BM25 indexing/querying."""
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text):
        raw = _TRAILING_PUNCT_RE.sub("", match)
        if raw:
            tokens.extend(_expand(raw))
    return tokens


def _expand(raw: str) -> list[str]:
    lowered = raw.lower()
    if _VERSION_RE.match(raw):
        return [lowered]

    core = lowered.lstrip("-")
    parts = [p for p in _SPLIT_RE.split(core) if p]

    tokens = [lowered]
    if len(parts) > 1:
        tokens.extend(parts)
    elif core != lowered:
        # A single-word flag ("--reload"): no internal separator to split
        # on, but the bare word (without leading dashes) still matters.
        tokens.append(core)
    return _dedupe(tokens)


def _dedupe(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
