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
from aegis.guardrails.commitments import is_unauthorized_commitment
from aegis.guardrails.risky_advice import Vouched, risky_reasons
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
    # A company-voiced money/account commitment (aegis.guardrails.commitments):
    # never passing, whatever it cites -- a citation shows the sentence came from
    # context, not that the company said it (design record 15e).
    UNAUTHORIZED_COMMITMENT = "unauthorized_commitment"
    # Install / link / secret-request advice the documentation corpus doesn't vouch
    # for (aegis.guardrails.risky_advice): never passing, whatever it cites -- an
    # issue comment saying "pip install x" is support only in the lexical sense
    # (design record 15l, Open Question #20).
    UNVOUCHED_RISKY_ADVICE = "unvouched_risky_advice"


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

    @property
    def has_unauthorized_commitment(self) -> bool:
        return any(c.status == ClaimStatus.UNAUTHORIZED_COMMITMENT for c in self.claims)

    @property
    def has_unvouched_risky_advice(self) -> bool:
        return any(c.status == ClaimStatus.UNVOUCHED_RISKY_ADVICE for c in self.claims)


# A list item starts a line: "-", "*", "+", "•", "1.", "1)" then whitespace.
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")


def _prose_blocks(text: str) -> Iterator[str]:
    """Paragraphs and list items, each a claim boundary (Open Question #18,
    design record 15k). A plain newline is NOT a boundary: a hard-wrapped
    sentence must stay one claim, or its first half becomes an uncited
    fragment. Before this, a bulleted answer was a few very long "claims"
    that passed or failed as one unit on word overlap."""
    for paragraph in _PARAGRAPH_BREAK_RE.split(text):
        item: list[str] = []
        for line in paragraph.splitlines():
            marker = _LIST_ITEM_RE.match(line)
            if marker and item:
                yield " ".join(item)
                item = []
            item.append(line[marker.end():] if marker else line.strip())
        if item:
            yield " ".join(item)


def _iter_claim_sentences(answer_text: str) -> Iterator[str]:
    for segment in split_code_segments(answer_text):
        if segment.kind == "code":
            continue  # not prose; any citation for it lives in the surrounding text
        for block in _prose_blocks(segment.content):
            yield from split_sentences(block)


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


def _default_entailment_fn(provider: str, model: str, temperature: float) -> EntailmentFn:
    from aegis.generation.llm.chat import LLMCallError, chat_completion

    def entails(claim: str, chunk_text: str) -> bool:
        try:
            raw = chat_completion(
                provider, model,
                [{"role": "system", "content": ENTAILMENT_SYSTEM_PROMPT},
                 {"role": "user", "content": f"Context:\n{chunk_text}\n\nClaim:\n{claim}"}],
                temperature=temperature, max_tokens=10, purpose="entailment", json_mode=True,
            )
        except LLMCallError as exc:
            # Logged: otherwise a quota outage is indistinguishable from "every
            # claim unsupported" (same lesson as the router, design record 14f).
            cause = exc.__cause__
            log_event(
                _log, logging.WARNING, "citation.entailment_api_error", provider=provider,
                error_type=type(cause).__name__, status_code=exc.status_code, detail=str(cause)[:300],
            )
            return False  # fail closed: an unverifiable citation counts as unsupported, not passed
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
        model: str | None = None,
        temperature: float = settings.citation.entailment_temperature,
        provider: str = settings.citation.entailment_provider,
        vouched: Vouched | None = None,  # None: the documentation corpus (risky_advice.documentation_vouched)
    ) -> None:
        if entailment_fn is None:
            from aegis.generation.llm.chat import resolve_role

            provider, model = resolve_role(provider, model, settings.openai.entailment_model,
                                           settings.citation.entailment_model)
            entailment_fn = _default_entailment_fn(provider, model, temperature)
        self._entailment_fn = entailment_fn
        self._threshold = lexical_overlap_threshold
        self._vouched = vouched

    def validate(self, answer_text: str, context: PackedContext) -> ValidationResult:
        chunk_text_by_citation = {pc.citation_id: pc.chunk.text for pc in context.chunks}

        claims: list[ClaimResult] = []
        tier2_calls = 0

        for sentence in _iter_claim_sentences(answer_text):
            citation_ids = [int(n) for n in _CITATION_RE.findall(sentence)]

            # Before every citation check, and before the short-sentence
            # exemption: "Refund approved [1]." is short but is exactly the case.
            if is_unauthorized_commitment(sentence):
                log_event(_log, logging.WARNING, "citation.unauthorized_commitment", sentence=sentence[:200])
                claims.append(ClaimResult(sentence=sentence, citation_ids=citation_ids,
                                          status=ClaimStatus.UNAUTHORIZED_COMMITMENT))
                continue

            if reasons := risky_reasons(sentence, self._vouched):
                log_event(_log, logging.WARNING, "citation.unvouched_risky_advice", reasons=reasons,
                          sentence=sentence[:200])
                claims.append(ClaimResult(sentence=sentence, citation_ids=citation_ids,
                                          status=ClaimStatus.UNVOUCHED_RISKY_ADVICE))
                continue

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
