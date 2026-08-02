"""Recursive character splitting: paragraph -> line -> sentence -> word,
falling back to a hard token window only when a piece has no natural
separator left at all (a pasted hash/URL/minified blob). Each level is tried
whole first; only oversized pieces recurse further, so most of a normally-
structured document splits at paragraph boundaries and never touches the
sentence or word fallbacks.
"""

from __future__ import annotations

from aegis.chunking.base import make_embedding_text
from aegis.chunking.merge import greedy_merge
from aegis.chunking.tokens import count_tokens, decode_tokens, encode_tokens, nearest_safe_boundary
from aegis.ingest.hashing import content_hash
from aegis.types import Chunk, Document

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " "]


def _hard_token_windows(text: str, max_tokens: int) -> list[str]:
    """Last-resort split for a piece with no usable separator at all —
    slices raw token windows, safe-boundary-snapped (see tokens.py) so a
    multi-byte character is never split across two pieces."""
    tokens = encode_tokens(text)
    if not tokens:
        return []
    pieces: list[str] = []
    pos, n = 0, len(tokens)
    while pos < n:
        target_end = min(pos + max_tokens, n)
        end = target_end if target_end >= n else nearest_safe_boundary(tokens, target_end)
        if end <= pos:
            end = target_end
        pieces.append(decode_tokens(tokens[pos:end]))
        pos = end
    return pieces


def _split_once(text: str, separator: str) -> list[str]:
    # Plain split, separator dropped — greedy_merge is the sole place that
    # re-inserts inter-piece spacing ("\n\n"). An earlier version re-attached
    # the original separator to each piece, which meant a merged chunk got
    # the separator *twice* (once embedded in the piece, once from
    # greedy_merge's join) — harmless-looking extra whitespace that silently
    # inflated real chunk token counts past `max_tokens`, found via a
    # measured chunk (issue #15000) at 443 tokens against a 400 budget. See
    # docs/explainers/02-chunking-strategies.md.
    return text.split(separator)


def recursive_split_pieces(text: str, max_tokens: int, separators: list[str] = _SEPARATORS) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text] if text else []

    if not separators:
        return _hard_token_windows(text, max_tokens)

    separator, *rest = separators
    pieces: list[str] = []
    for part in _split_once(text, separator):
        if not part:
            continue
        if count_tokens(part) <= max_tokens:
            pieces.append(part)
        else:
            pieces.extend(recursive_split_pieces(part, max_tokens, rest))
    return pieces


class RecursiveChunker:
    name = "recursive"

    def __init__(self, max_tokens: int, overlap_tokens: int, contextual_enrichment: bool = False):
        if overlap_tokens >= max_tokens:
            raise ValueError(f"overlap_tokens ({overlap_tokens}) must be < max_tokens ({max_tokens})")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.contextual_enrichment = contextual_enrichment

    def chunk(self, document: Document) -> list[Chunk]:
        pieces = recursive_split_pieces(document.text, self.max_tokens)
        if not pieces:
            return []

        merged = greedy_merge(pieces, self.max_tokens, self.overlap_tokens)
        chunks: list[Chunk] = []
        for index, text in enumerate(merged):
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
        return chunks
