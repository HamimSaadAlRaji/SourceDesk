"""TDD for the two-tier citation validator (plan.md Phase 9, step 4). The
LLM entailment tier is dependency-injected out via `entailment_fn` -- same
treatment as `RouterClassifier`'s `classify_fn` -- so these tests never hit
the network. The exit criterion this file exists to prove is "the validator
catches deliberately corrupted citations"
(`test_invalid_citation_when_marker_points_to_a_chunk_not_in_context`).
"""

from aegis.chunking.tokens import count_tokens
from aegis.generation.budgeter import PackedChunk, PackedContext
from aegis.guardrails.citations import CitationValidator, ClaimStatus
from aegis.types import Chunk, DocType


def _chunk(id: str, text: str) -> Chunk:
    return Chunk(
        id=id,
        doc_id=id,
        text=text,
        doc_type=DocType.DOCUMENTATION,
        content_hash="hash",
        token_count=count_tokens(text),
        embedding_text=text,
    )


def _context(texts_by_citation: dict[int, str]) -> PackedContext:
    packed = [
        PackedChunk(citation_id=cid, chunk=_chunk(f"chunk::{cid}", text), score=1.0, token_count=count_tokens(text))
        for cid, text in texts_by_citation.items()
    ]
    return PackedContext(
        chunks=packed, total_tokens=sum(pc.token_count for pc in packed), requested_count=len(packed), dropped_count=0
    )


def _validator(entailment_fn=None, threshold=0.4):
    return CitationValidator(entailment_fn=entailment_fn, lexical_overlap_threshold=threshold)


def test_short_connective_lead_in_is_exempt():
    context = _context({1: "FastAPI uses Pydantic for validation."})
    validator = _validator()

    result = validator.validate("Here's how to do that:", context)

    assert result.claims[0].status == ClaimStatus.EXEMPT
    assert result.valid_citation_rate == 1.0


def test_factual_sentence_with_no_citation_marker_is_missing_citation():
    context = _context({1: "FastAPI uses Pydantic for request body validation."})
    validator = _validator()

    result = validator.validate("FastAPI uses Pydantic for request body validation.", context)

    assert result.claims[0].status == ClaimStatus.MISSING_CITATION
    assert result.valid_citation_rate == 0.0


def test_invalid_citation_when_marker_points_to_a_chunk_not_in_context():
    # Context only has citation [1] -- [7] is a deliberately corrupted
    # reference, the exact failure mode the exit criterion names.
    context = _context({1: "FastAPI uses Pydantic for request body validation."})
    validator = _validator()

    result = validator.validate("FastAPI uses Pydantic for request body validation. [7]", context)

    assert result.claims[0].status == ClaimStatus.INVALID_CITATION
    assert result.valid_citation_rate == 0.0


def test_high_overlap_passes_tier_one_without_calling_entailment_fn():
    context = _context({1: "FastAPI uses Pydantic models for request body validation."})
    calls = []

    def entailment_fn(claim, chunk_text):
        calls.append((claim, chunk_text))
        return True

    validator = _validator(entailment_fn=entailment_fn, threshold=0.4)
    result = validator.validate("FastAPI uses Pydantic models for request body validation. [1]", context)

    assert result.claims[0].status == ClaimStatus.SUPPORTED_LEXICAL
    assert calls == []
    assert result.tier2_calls == 0
    assert result.valid_citation_rate == 1.0


def test_low_overlap_falls_through_to_entailment_tier():
    context = _context({1: "Dependency injection in FastAPI is handled by the Depends function."})
    calls = []

    def entailment_fn(claim, chunk_text):
        calls.append((claim, chunk_text))
        return True

    validator = _validator(entailment_fn=entailment_fn, threshold=0.4)
    # Deliberately low lexical overlap with the chunk above, but a real paraphrase.
    result = validator.validate("You inject dependencies with a special function. [1]", context)

    assert result.claims[0].status == ClaimStatus.SUPPORTED_ENTAILMENT
    assert len(calls) == 1
    assert result.tier2_calls == 1


def test_entailment_fn_returning_false_marks_claim_unsupported():
    context = _context({1: "Dependency injection in FastAPI is handled by the Depends function."})
    validator = _validator(entailment_fn=lambda claim, chunk_text: False, threshold=0.4)

    result = validator.validate("The sky is blue on Tuesdays in Paris. [1]", context)

    assert result.claims[0].status == ClaimStatus.UNSUPPORTED
    assert result.valid_citation_rate == 0.0


def test_valid_citation_rate_excludes_exempt_claims_from_denominator():
    context = _context({1: "FastAPI uses Pydantic models for request body validation."})
    validator = _validator(entailment_fn=lambda c, t: True)

    answer = "Here's the answer. FastAPI uses Pydantic models for request body validation. [1]"
    result = validator.validate(answer, context)

    statuses = [c.status for c in result.claims]
    assert ClaimStatus.EXEMPT in statuses
    assert result.valid_citation_rate == 1.0  # the one real claim is supported; exempt sentence doesn't count


def test_multiple_citation_ids_on_one_sentence_checked_against_combined_text():
    context = _context(
        {
            1: "FastAPI validates request bodies using Pydantic models.",
            2: "Pydantic models raise a 422 error on validation failure in FastAPI.",
        }
    )
    calls = []

    def entailment_fn(claim, chunk_text):
        calls.append(chunk_text)
        return True

    validator = _validator(entailment_fn=entailment_fn, threshold=0.9)  # force tier 2
    result = validator.validate(
        "FastAPI validates request bodies using Pydantic models and raises a 422 error on failure. [1][2]", context
    )

    assert result.claims[0].citation_ids == [1, 2]
    assert len(calls) == 1
    assert context.chunks[0].chunk.text in calls[0]
    assert context.chunks[1].chunk.text in calls[0]


def test_no_claims_requiring_citation_gives_rate_of_one():
    context = _context({1: "some context"})
    validator = _validator()

    result = validator.validate("I don't know.", context)

    assert result.valid_citation_rate == 1.0


def test_fenced_code_block_does_not_get_glued_to_surrounding_prose():
    # Real bug observed in the Phase 9 full-pipeline smoke test (design
    # record 09 §6): a colon-terminated lead-in with no following period
    # before a fenced code block used to glue the lead-in, the whole code
    # block, and the next real sentence into one giant "sentence" -- diluting
    # that sentence's lexical overlap with irrelevant code tokens.
    context = _context({1: "FastAPI uses Pydantic models for request body validation."})
    validator = _validator(threshold=0.4)

    answer = (
        "Here's an example:\n"
        "```python\n"
        "print('hello')\n"
        "```\n"
        "FastAPI validates request bodies with Pydantic models. [1]"
    )
    result = validator.validate(answer, context)

    sentences = [c.sentence for c in result.claims]
    assert "Here's an example:" in sentences
    assert "FastAPI validates request bodies with Pydantic models. [1]" in sentences
    assert not any("```" in s for s in sentences)
    # The code-block-glued blob must not appear as a single claim.
    assert not any("print(" in s for s in sentences)

    cited_claim = next(c for c in result.claims if c.citation_ids == [1])
    assert cited_claim.status == ClaimStatus.SUPPORTED_LEXICAL


# Open Question #18 (design record 15k): list items and paragraphs are claims of
# their own. Before, a bulleted answer became a few very long "claims" (up to
# 1,718 chars in 15h) that passed or failed as one unit on word overlap.

_PYDANTIC_DOC = "FastAPI uses Pydantic models for request body validation."


def _no_entailment(claim: str, chunk_text: str) -> bool:
    return False


def test_each_bullet_is_its_own_claim_so_an_unsupported_bullet_is_not_masked():
    answer = (
        "Details from the context:\n"
        "- FastAPI validates request bodies with Pydantic models [1].\n"
        "- FastAPI automatically encrypts every outgoing database connection [1]."
    )
    result = _validator(entailment_fn=_no_entailment).validate(answer, _context({1: _PYDANTIC_DOC}))

    statuses = {c.sentence: c.status for c in result.claims}
    assert statuses["FastAPI validates request bodies with Pydantic models [1]."] == ClaimStatus.SUPPORTED_LEXICAL
    assert statuses["FastAPI automatically encrypts every outgoing database connection [1]."] == ClaimStatus.UNSUPPORTED


def test_numbered_list_items_are_separate_claims():
    answer = "1. FastAPI validates request bodies with Pydantic models [1]\n2. Pydantic models validate request bodies in FastAPI [1]"
    result = _validator(entailment_fn=_no_entailment).validate(answer, _context({1: _PYDANTIC_DOC}))

    assert [c.sentence for c in result.claims] == [
        "FastAPI validates request bodies with Pydantic models [1]",
        "Pydantic models validate request bodies in FastAPI [1]",
    ]


def test_paragraph_break_ends_a_claim_even_without_a_period():
    answer = "FastAPI validates request bodies with Pydantic models [1]\n\nYou should also pin your Pydantic version"
    result = _validator(entailment_fn=_no_entailment).validate(answer, _context({1: _PYDANTIC_DOC}))

    assert [c.status for c in result.claims] == [ClaimStatus.SUPPORTED_LEXICAL, ClaimStatus.MISSING_CITATION]


def test_hard_wrapped_sentence_stays_one_claim():
    answer = "FastAPI validates request bodies\nwith Pydantic models [1]."
    result = _validator(entailment_fn=_no_entailment).validate(answer, _context({1: _PYDANTIC_DOC}))

    [claim] = result.claims
    assert claim.status == ClaimStatus.SUPPORTED_LEXICAL
