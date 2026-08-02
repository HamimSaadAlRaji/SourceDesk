"""Fixed-size token windows, ± overlap — the naive baseline every other
strategy in this phase has to beat. No structure awareness at all: it will
slice through a heading, a paragraph, or a fenced code block exactly like
Phase 0's 500-char baseline did, on purpose (see the Phase 0 design record
for the concrete failure this caused: a code block severed from the
"(deprecated)" heading explaining it).
"""

from __future__ import annotations

from aegis.chunking.base import make_embedding_text
from aegis.chunking.tokens import count_tokens, decode_tokens, encode_tokens, nearest_safe_boundary
from aegis.ingest.hashing import content_hash
from aegis.types import Chunk, Document


class FixedChunker:
    name = "fixed"

    def __init__(self, max_tokens: int, overlap_tokens: int, contextual_enrichment: bool = False):
        if overlap_tokens >= max_tokens:
            raise ValueError(f"overlap_tokens ({overlap_tokens}) must be < max_tokens ({max_tokens})")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.contextual_enrichment = contextual_enrichment

    def chunk(self, document: Document) -> list[Chunk]:
        tokens = encode_tokens(document.text)
        if not tokens:
            return []

        n = len(tokens)
        chunks: list[Chunk] = []
        pos = 0
        index = 0

        while pos < n:
            target_end = min(pos + self.max_tokens, n)
            end = target_end if target_end >= n else nearest_safe_boundary(tokens, target_end)
            if end <= pos:
                end = target_end

            text = decode_tokens(tokens[pos:end])
            embedding_text = make_embedding_text(text, document, [], self.contextual_enrichment)
            chunks.append(
                Chunk(
                    id=f"{document.id}::{self.name}::{index}",
                    doc_id=document.id,
                    text=text,
                    heading_path=[],
                    doc_version=document.doc_version,
                    doc_type=document.doc_type,
                    product_area=document.product_area,
                    content_hash=content_hash(text),
                    token_count=count_tokens(text),
                    embedding_text=embedding_text,
                )
            )
            index += 1

            if end >= n:
                break
            next_pos = nearest_safe_boundary(tokens, end - self.overlap_tokens)
            pos = next_pos if next_pos > pos else pos + 1

        return chunks
