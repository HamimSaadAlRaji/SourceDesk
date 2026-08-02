"""Heading-aware chunking — split on Markdown heading boundaries, keep
fenced code blocks atomic and attached to the prose that introduces them.

The single highest-leverage chunking decision for this corpus (plan.md
Phase 2): Phase 0's baseline severed a "(deprecated)" heading from the body
explaining it with a raw 500-char slice. This strategy is built specifically
so that failure mode is structurally impossible — a chunk's `heading_path`
always names the section it came from, and a fenced code block is never cut,
even when it alone exceeds `max_tokens` (see `docs/explainers/
02-chunking-strategies.md` for the two real corpus cases that hit that
exception: `release-notes.md` and issue #2266's pasted build log).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from aegis.chunking.base import make_embedding_text
from aegis.chunking.merge import greedy_merge
from aegis.chunking.recursive import recursive_split_pieces
from aegis.chunking.segments import Segment, split_code_segments
from aegis.chunking.tokens import count_tokens
from aegis.ingest.hashing import content_hash
from aegis.types import Chunk, Document

# Heading lines are only ever matched inside *text* segments (never inside a
# fenced code block), so a Python `# comment` can't be mistaken for a
# Markdown heading.
_HEADING_LINE_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)


@dataclass
class _Section:
    heading_path: list[str]
    segments: list[Segment] = field(default_factory=list)


def _build_sections(text: str) -> list[_Section]:
    heading_stack: list[tuple[int, str]] = []
    sections: list[_Section] = []
    current = _Section(heading_path=[])

    def start_new_section() -> None:
        nonlocal current
        if current.segments:
            sections.append(current)
        current = _Section(heading_path=[title for _, title in heading_stack])

    for doc_segment in split_code_segments(text):
        if doc_segment.kind == "code":
            current.segments.append(doc_segment)
            continue

        pos = 0
        for match in _HEADING_LINE_RE.finditer(doc_segment.content):
            pre = doc_segment.content[pos : match.start()]
            if pre.strip():
                current.segments.append(Segment("text", pre))

            level, title = len(match.group(1)), match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            start_new_section()
            pos = match.end()

        tail = doc_segment.content[pos:]
        if tail.strip():
            current.segments.append(Segment("text", tail))

    if current.segments:
        sections.append(current)
    return sections


def _section_pieces(section: _Section, max_tokens: int) -> list[str]:
    pieces: list[str] = []
    for seg in section.segments:
        if seg.kind == "code" or count_tokens(seg.content) <= max_tokens:
            pieces.append(seg.content)
        else:
            pieces.extend(recursive_split_pieces(seg.content, max_tokens))
    return pieces


class HeadingAwareChunker:
    name = "heading_aware"

    def __init__(self, max_tokens: int, overlap_tokens: int, contextual_enrichment: bool = False):
        if overlap_tokens >= max_tokens:
            raise ValueError(f"overlap_tokens ({overlap_tokens}) must be < max_tokens ({max_tokens})")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.contextual_enrichment = contextual_enrichment

    def chunk(self, document: Document) -> list[Chunk]:
        sections = _build_sections(document.text)
        chunks: list[Chunk] = []
        index = 0

        for section in sections:
            full_text = "\n\n".join(s.content for s in section.segments).strip()
            if not full_text:
                continue
            if count_tokens(full_text) <= self.max_tokens:
                texts = [full_text]
            else:
                pieces = _section_pieces(section, self.max_tokens)
                texts = greedy_merge(pieces, self.max_tokens, self.overlap_tokens)

            for text in texts:
                embedding_text = make_embedding_text(
                    text, document, section.heading_path, self.contextual_enrichment
                )
                chunks.append(
                    Chunk(
                        id=f"{document.id}::{self.name}::{index}",
                        doc_id=document.id,
                        text=text,
                        heading_path=section.heading_path,
                        doc_version=document.doc_version,
                        doc_type=document.doc_type,
                        product_area=document.product_area,
                        content_hash=content_hash(text),
                        token_count=count_tokens(text),
                        embedding_text=embedding_text,
                    )
                )
                index += 1

        return chunks
