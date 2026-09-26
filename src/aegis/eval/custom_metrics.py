"""Custom metrics RAGAS doesn't provide (plan.md Phase 11, step 4):
citation validity aggregation, escalation appropriateness, and code-block
static syntax validity. All three are deterministic and code-based (zero
LLM cost), reusing components already built and tested elsewhere in this
project (Phase 9's `CitationValidator`, Phase 10's escalation decision,
Phase 2's `split_code_segments`) rather than duplicating their logic.
"""

from __future__ import annotations

import ast
import re

from pydantic import BaseModel

from aegis.chunking.segments import split_code_segments
from aegis.guardrails.citations import ClaimStatus, ValidationResult

# --- Citation validity aggregation ---


class CitationValiditySummary(BaseModel):
    n_answers: int
    mean_valid_citation_rate: float
    status_counts: dict[ClaimStatus, int]


def citation_validity_summary(results: list[ValidationResult]) -> CitationValiditySummary:
    if not results:
        return CitationValiditySummary(n_answers=0, mean_valid_citation_rate=0.0, status_counts={})

    status_counts: dict[ClaimStatus, int] = {}
    for result in results:
        for claim in result.claims:
            status_counts[claim.status] = status_counts.get(claim.status, 0) + 1

    mean_rate = sum(r.valid_citation_rate for r in results) / len(results)
    return CitationValiditySummary(n_answers=len(results), mean_valid_citation_rate=mean_rate, status_counts=status_counts)


# --- Escalation appropriateness ---
# Same TP/FP/FN/TN convention as Phase 10's real confusion matrix
# (experiments/10a_tau_sweep.py, docs/results/confusion-matrix.md):
# "escalate" is the positive class. TP = correctly escalated on a real
# negative (should-refuse) query; FP = over-cautious escalation on a real
# positive; FN = the Air Canada quadrant (should have escalated, didn't);
# TN = correctly answered directly.


class EscalationConfusion(BaseModel):
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    deflection_rate: float


def escalation_appropriateness(escalated: list[bool], is_negative: list[bool]) -> EscalationConfusion:
    if len(escalated) != len(is_negative):
        raise ValueError(f"escalated and is_negative must have the same length, got {len(escalated)} and {len(is_negative)}")

    tp = sum(1 for e, n in zip(escalated, is_negative, strict=True) if e and n)
    fp = sum(1 for e, n in zip(escalated, is_negative, strict=True) if e and not n)
    fn = sum(1 for e, n in zip(escalated, is_negative, strict=True) if not e and n)
    tn = sum(1 for e, n in zip(escalated, is_negative, strict=True) if not e and not n)

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    deflection_rate = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return EscalationConfusion(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, deflection_rate=deflection_rate)


# --- Code-block static syntax validity ---
# "Is the code in the answer runnable?" (plan.md Phase 11 step 4) --
# `ast.parse`, not sandboxed execution (real security-exposure/engineering-
# time tradeoff locked in with the user last session; Phase 15 doesn't
# exist yet to harden a real execution sandbox). Catches syntax errors, not
# runtime/logic errors -- a real, disclosed limitation, not the full claim.
#
# Only Python-tagged or untagged fences are checked. Measured on the real
# corpus (data/processed/documents.jsonl): untagged fences are the single
# largest group (7,906), python-family tags the next largest (~3,495
# combined across python/Python/py/python3); everything else (bash, json,
# html, yaml, js, ...) is an explicit, unambiguous signal to skip rather
# than guess. Untagged is included, not skipped, because FastAPI answers
# routinely omit the language tag on Python snippets -- excluding it would
# make the metric blind to most real code blocks, not just non-Python ones.
_PYTHON_TAGS = {"", "python", "py", "python3"}
_OPEN_FENCE_RE = re.compile(r"^```([a-zA-Z0-9_+-]*)\n", re.MULTILINE)


class CodeBlockSyntaxResult(BaseModel):
    n_checked: int
    n_valid: int
    n_skipped_non_python: int
    invalid_blocks: list[str]

    @property
    def valid_rate(self) -> float:
        if self.n_checked == 0:
            return 1.0  # vacuous truth -- same convention as ValidationResult.valid_citation_rate
        return self.n_valid / self.n_checked


def _language_and_body(fence_content: str) -> tuple[str, str]:
    match = _OPEN_FENCE_RE.match(fence_content)
    language = match.group(1).strip().lower() if match else ""
    body = fence_content[match.end() :] if match else fence_content
    # Strip a well-formed closing fence line if present (real corpus data
    # can leave it unclosed -- split_code_segments already documents this).
    body = re.sub(r"\n```[^\n]*\Z", "", body)
    return language, body


def code_block_syntax_validity(answer_text: str) -> CodeBlockSyntaxResult:
    n_checked = 0
    n_valid = 0
    n_skipped = 0
    invalid_blocks: list[str] = []

    for segment in split_code_segments(answer_text):
        if segment.kind != "code":
            continue
        language, body = _language_and_body(segment.content)
        if language not in _PYTHON_TAGS:
            n_skipped += 1
            continue
        n_checked += 1
        try:
            ast.parse(body)
            n_valid += 1
        except SyntaxError:
            invalid_blocks.append(body)

    return CodeBlockSyntaxResult(n_checked=n_checked, n_valid=n_valid, n_skipped_non_python=n_skipped, invalid_blocks=invalid_blocks)
