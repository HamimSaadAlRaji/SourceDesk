"""Split markdown into text/code segments, fenced code blocks kept atomic.

Phase 0's baseline severed a "(deprecated)" heading from its body with a raw
500-char slice; the equivalent failure here is a fixed- or sentence-boundary
split landing inside a fenced code block, handing the reranker/generator half
a function. Every chunker that claims code-block integrity (heading-aware,
semantic) does so by chunking these segments, never raw text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# The closing fence only needs to *start* a line with "```" — not be the
# line's sole content. Two real corpus patterns break a stricter match:
# CRLF line endings (issue #15000: a closing line is literally "```\r\n",
# and a pattern requiring only trailing spaces/tabs before end-of-line
# doesn't tolerate the "\r"), and a closing fence sharing its line with
# trailing HTML (issue #9780: "```</div>", no newline in between). Either
# one, unhandled, falls through to the "unclosed fence" branch and silently
# swallows every heading and paragraph after it into one giant code segment
# — see the design record.
_FENCE_RE = re.compile(r"^```[^\n]*\n.*?(?:\n```[^\n]*|\Z)", re.MULTILINE | re.DOTALL)


@dataclass(frozen=True)
class Segment:
    kind: Literal["text", "code"]
    content: str


def split_code_segments(text: str) -> list[Segment]:
    """Alternating text/code segments in original order. A fence missing its
    closing ``` (seen in real corpus data — a truncated pasted log) is
    treated as code running to the end of the text rather than silently
    reinterpreted as prose."""
    segments: list[Segment] = []
    cursor = 0
    for match in _FENCE_RE.finditer(text):
        if match.start() > cursor:
            before = text[cursor : match.start()]
            if before.strip():
                segments.append(Segment("text", before))
        segments.append(Segment("code", match.group(0)))
        cursor = match.end()

    if cursor < len(text):
        tail = text[cursor:]
        if tail.strip():
            segments.append(Segment("text", tail))

    return segments
