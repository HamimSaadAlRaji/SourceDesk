"""Stratification for the golden evaluation set (plan.md Phase 4 step 3):
how-to, error/debugging, conceptual, version-specific, multi-hop, ambiguous.

A silent bug here corrupts every per-stratum number reported from Phase 6
onward — Phase 6, 10, and 11 all read the stratum a query was assigned at
harvest time, not something re-derived later. Pure keyword/pattern heuristic
in the same style as `aegis.ingest.product_area`, not an LLM call: cheap,
deterministic, and inspectable — the same build-vs-import call already made
for that component, for the same reason (CLAUDE.md §2).

Priority order (first match wins), most specific signal first: a query that
regresses across a version AND throws an exception is stratified as
version-specific, not error/debugging — the more actionable label for
Phase 6, which wants to know whether the system handles version drift
specifically, not just whether it can answer "I got an error".
"""

from __future__ import annotations

import re

from aegis.ingest.product_area import matched_areas, strip_code_and_boilerplate
from aegis.types import Stratum

_VERSION_NUMBER_RE = re.compile(r"\b\d+\.\d+(?:\.\d+)?\b")
_VERSION_CUES = (
    "upgrad",
    "downgrad",
    "regression",
    "changelog",
    "release note",
    "breaking change",
    "after updating",
    "since version",
    "only affects",
)

_ERROR_CUES = (
    "traceback",
    "error",
    "exception",
    "crash",
    "fails",
    "raises",
    "bug",
    "not working",
    "doesn't work",
    "does not work",
    "broken",
    "stack trace",
)

_HOW_TO_CUES = (
    "how do i",
    "how do you",
    "how to ",
    "how can i",
    "is it possible to",
    "what's the best way",
    "what is the best way",
    "recommended way",
    "want to implement",
)

_CONCEPTUAL_CUES = (
    "why does",
    "why is",
    "why do",
    "what is the difference",
    "what's the difference",
    "what does",
    "understand",
    "confused about",
    "clarif",
)


def classify_stratum(title: str, body: str, resolution: str) -> Stratum:
    query_text = strip_code_and_boilerplate(body)
    full_text = f"{title}\n{query_text}\n{strip_code_and_boilerplate(resolution)}".lower()

    if _VERSION_NUMBER_RE.search(full_text) and any(cue in full_text for cue in _VERSION_CUES):
        return Stratum.VERSION_SPECIFIC
    if any(cue in full_text for cue in _ERROR_CUES):
        return Stratum.ERROR_DEBUGGING
    if len(matched_areas(title, query_text)) >= 2:
        return Stratum.MULTI_HOP
    if any(cue in full_text for cue in _HOW_TO_CUES):
        return Stratum.HOW_TO
    if any(cue in full_text for cue in _CONCEPTUAL_CUES):
        return Stratum.CONCEPTUAL
    return Stratum.AMBIGUOUS
