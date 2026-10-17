"""Two-tier citation validator (plan.md Phase 9, step 4): parses a
generated answer into per-sentence claims, then checks whether each cited
chunk actually supports the sentence it's attached to.

Two tiers, cheapest first ("cost control matters" per plan.md):
  1. Lexical overlap between the claim sentence and its cited chunk's text
     -- free, no LLM call. Same "fraction of tokens found in the other
     text" technique Phase 4's golden-set auto-labeling already validated
     on this corpus (`eval/auto_label.py`'s `_text_overlap`), reimplemented
     rather than imported: guardrails/ is a production runtime path and
     shouldn't depend on eval/'s offline-labeling script.
  2. LLM entailment, only for claims tier 1 couldn't confidently pass -- a
     small/fast model (same "cheap task, cheap model" convention as
     RouterSettings/QuerySettings), asked a direct yes/no entailment
     question. Most real claims should clear tier 1 for free; tier 2 is the
     expensive fallback, not the common path.

Sentences that don't need a citation at all (short connective lead-ins,
e.g. "Here's how to do that:") are EXEMPT, not penalized -- plan.md's
requirement is "every factual claim carries a citation," not "every
sentence." Sentence splitting reuses `chunking/semantic.py`'s regex
splitter (documented there as misfiring on abbreviations/numbered lists --
that limitation applies here too, see design record 09 §6).

Fenced code blocks are stripped out *before* sentence splitting, via
`chunking/segments.py`'s `split_code_segments` (Phase 2, already hardened
against CRLF line endings and fences sharing a line with trailing HTML).
Without this, a lead-in ending in ":" with no following period before a
code fence gets glued to the entire code block and whatever prose follows
it into one giant "sentence" -- a real bug found in the Phase 9
full-pipeline smoke test, not a hypothetical (design record 09 §6). Code
segments themselves produce no claim: they're not prose, and any citation
for a code example lives in the surrounding text, not attached to the code
itself.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterator
from enum import StrEnum

from pydantic import BaseModel

from aegis.chunking.segments import split_code_segments
from aegis.chunking.semantic import split_sentences
from aegis.config import settings
from aegis.generation.budgeter import PackedContext
from aegis.observability.logging import get_logger, log_event

_log = get_logger("citations")

_CITATION_RE = re.compile(r"\[(\d+)\]")
_WORD_RE = re.compile(r"[a-z0-9]+")

EntailmentFn = Callable[[str, str], bool]

ENTAILMENT_SYSTEM_PROMPT = """You judge whether a piece of context supports a claim. \
Respond with JSON only: {"supported": true} or {"supported": false}. "Supported" means \
the context contains information that directly backs the claim -- not just related, not \
just plausible, actually stated or clearly implied. When genuinely unsure, respond false."""


class ClaimStatus(StrEnum):
    SUPPORTED_LEXICAL = "supported_lexical"
    SUPPORTED_ENTAILMENT = "supported_entailment"
    UNSUPPORTED = "unsupported"
    MISSING_CITATION = "missing_citation"
    INVALID_CITATION = "invalid_citation"
    EXEMPT = "exempt"


_PASSING = {ClaimStatus.SUPPORTED_LEXICAL, ClaimStatus.SUPPORTED_ENTAILMENT, ClaimStatus.EXEMPT}


class ClaimResult(BaseModel):
    sentence: str
    citation_ids: list[int]
    status: ClaimStatus
    overlap: float | None = None


class ValidationResult(BaseModel):
    claims: list[ClaimResult]
    tier2_calls: int

    @property
    def valid_citation_rate(self) -> float:
        """Fraction of citation-*requiring* claims (EXEMPT ones excluded
        from both halves of the ratio) that ended up supported -- the
        number the Phase 9 exit criterion ("100% of factual claims carry a
        parseable citation") is measured against."""
        checked = [c for c in self.claims if c.status != ClaimStatus.EXEMPT]
        if not checked:
            return 1.0
        passing = sum(1 for c in checked if c.status in _PASSING)
        return passing / len(checked)


def _iter_claim_sentences(answer_text: str) -> Iterator[str]:
    for segment in split_code_segments(answer_text):
        if segment.kind == "code":
            continue  # not prose; any citation for it lives in the surrounding text
        yield from split_sentences(segment.content)


def _requires_citation(sentence: str) -> bool:
    stripped = sentence.strip()
    if not stripped or stripped.endswith(":"):
        return False  # lead-in / connective, e.g. "Here's how to do that:"
    return len(stripped.split()) > 4  # too short to carry a checkable factual claim otherwise


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _lexical_overlap(claim: str, chunk_text: str) -> float:
    """Exact word-boundary token-set intersection, deliberately *not*
    `eval/auto_label.py`'s `token in candidate_text` substring check that
    this technique is otherwise modeled on: substring containment let short
    common tokens like "on"/"in" false-hit inside unrelated longer words
    (e.g. "on" inside "injection"), which would let tier 1 silently pass
    claims tier 1 exists to catch. Found via this module's own test suite,
    not by inspection -- see design record 09 §6."""
    claim_tokens = _tokenize(_CITATION_RE.sub("", claim))
    if not claim_tokens:
        return 0.0
    chunk_tokens = _tokenize(chunk_text)
    return len(claim_tokens & chunk_tokens) / len(claim_tokens)


def _default_entailment_fn(model: str, temperature: float) -> EntailmentFn:
    from groq import APIError, Groq

    client = Groq(api_key=settings.groq.api_key)

    def entails(claim: str, chunk_text: str) -> bool:
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=10,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": ENTAILMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{chunk_text}\n\nClaim:\n{claim}"},
                ],
            )
        except APIError as exc:
            # Logged: otherwise a quota outage is indistinguishable from "every
            # claim unsupported" (same lesson as the router, design record 14f).
            log_event(
                _log, logging.WARNING, "citation.entailment_api_error",
                error_type=type(exc).__name__, status_code=getattr(exc, "status_code", None), detail=str(exc)[:300],
            )
            return False  # fail closed: an unverifiable citation counts as unsupported, not passed
        raw = completion.choices[0].message.content or ""
        try:
            return bool(json.loads(raw)["supported"])
        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    return entails


class CitationValidator:
    def __init__(
        self,
        entailment_fn: EntailmentFn | None = None,
        lexical_overlap_threshold: float = settings.citation.lexical_overlap_threshold,
        model: str = settings.citation.entailment_model,
        temperature: float = settings.citation.entailment_temperature,
    ) -> None:
        self._entailment_fn = entailment_fn or _default_entailment_fn(model, temperature)
        self._threshold = lexical_overlap_threshold

    def validate(self, answer_text: str, context: PackedContext) -> ValidationResult:
        chunk_text_by_citation = {pc.citation_id: pc.chunk.text for pc in context.chunks}

        claims: list[ClaimResult] = []
        tier2_calls = 0

        for sentence in _iter_claim_sentences(answer_text):
            citation_ids = [int(n) for n in _CITATION_RE.findall(sentence)]

            if not _requires_citation(sentence):
                claims.append(ClaimResult(sentence=sentence, citation_ids=citation_ids, status=ClaimStatus.EXEMPT))
                continue

            if not citation_ids:
                claims.append(ClaimResult(sentence=sentence, citation_ids=[], status=ClaimStatus.MISSING_CITATION))
                continue

            invalid = [cid for cid in citation_ids if cid not in chunk_text_by_citation]
            if invalid:
                claims.append(
                    ClaimResult(sentence=sentence, citation_ids=citation_ids, status=ClaimStatus.INVALID_CITATION)
                )
                continue

            cited_text = "\n".join(chunk_text_by_citation[cid] for cid in citation_ids)
            overlap = _lexical_overlap(sentence, cited_text)

            if overlap >= self._threshold:
                claims.append(
                    ClaimResult(
                        sentence=sentence,
                        citation_ids=citation_ids,
                        status=ClaimStatus.SUPPORTED_LEXICAL,
                        overlap=overlap,
                    )
                )
                continue

            tier2_calls += 1
            supported = self._entailment_fn(sentence, cited_text)
            status = ClaimStatus.SUPPORTED_ENTAILMENT if supported else ClaimStatus.UNSUPPORTED
            claims.append(
                ClaimResult(sentence=sentence, citation_ids=citation_ids, status=status, overlap=overlap)
            )

        return ValidationResult(claims=claims, tier2_calls=tier2_calls)
