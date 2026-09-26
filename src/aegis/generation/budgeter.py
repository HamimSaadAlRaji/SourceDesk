"""Context budgeter (plan.md Phase 9, step 2): fits retrieved chunks into a
fixed token budget for the generation prompt, and places them to fight the
"lost in the middle" effect -- long-context attention is strongest at the
start and end of the prompt and weakest in the middle (the mechanism behind
plan.md's own note: "relevance degrades in the middle of long contexts"), so
appending chunks in plain rank order buries the 2nd/3rd-best result exactly
where the model attends to it least.

Split into two decisions on purpose: `pack_context` decides *which* chunks
survive the budget (real token counts via `aegis.chunking.tokens`, not char
counts or an assumed average -- Phase 3/6 already found this corpus doesn't
behave like a clean average), `_bookend_order` decides *where* the survivors
sit in the final text. Citation numbers track rank, not display position, so
"[1]" always means "the single most relevant retrieved chunk" regardless of
where the bookend ordering physically placed its text.
"""

from __future__ import annotations

from pydantic import BaseModel

from aegis.chunking.tokens import count_tokens, decode_tokens, encode_tokens, nearest_safe_boundary
from aegis.types import Chunk, Retrieved


class PackedChunk(BaseModel):
    citation_id: int
    chunk: Chunk
    score: float
    token_count: int
    truncated: bool = False


class PackedContext(BaseModel):
    chunks: list[PackedChunk]  # display order (bookended), not rank order
    total_tokens: int
    requested_count: int
    dropped_count: int


def _truncate_to_budget(text: str, budget_tokens: int) -> str:
    tokens = encode_tokens(text)
    if len(tokens) <= budget_tokens:
        return text
    cut = nearest_safe_boundary(tokens, budget_tokens)
    return decode_tokens(tokens[:cut])


def _bookend_order(items: list[PackedChunk]) -> list[PackedChunk]:
    """rank 1 -> first position, rank 2 -> last position, rank 3 -> second
    position, rank 4 -> second-to-last, ... worst-ranked survivor ends up
    dead center."""
    n = len(items)
    ordered: list[PackedChunk | None] = [None] * n
    left, right = 0, n - 1
    for i, item in enumerate(items):
        if i % 2 == 0:
            ordered[left] = item
            left += 1
        else:
            ordered[right] = item
            right -= 1
    return ordered  # type: ignore[return-value]


def pack_context(retrieved: list[Retrieved], budget_tokens: int) -> PackedContext:
    """`retrieved` is sorted by `.rank` internally, so callers don't have to
    guarantee list order themselves. Greedy strict-prefix packing: walks
    rank order, keeps every chunk that fits whole, and stops the instant one
    doesn't -- a lower-ranked chunk never displaces or gets inserted around
    a higher-ranked one just because it happens to be smaller.

    One exception: if the *first* (highest-ranked) chunk alone exceeds the
    whole budget, it's truncated rather than dropped. Real chunks can exceed
    the nominal 400-token chunking cap -- up to 6,325 tokens on this corpus's
    oversized atomic code blocks (`docs/results/chunking-stats.md`) -- and an
    empty context because the single most relevant result didn't fit whole
    is worse than a partial version of that same result.
    """
    ranked = sorted(retrieved, key=lambda r: r.rank)

    selected: list[PackedChunk] = []
    used_tokens = 0

    for item in ranked:
        remaining = budget_tokens - used_tokens
        cost = count_tokens(item.chunk.text)

        if cost <= remaining:
            selected.append(
                PackedChunk(citation_id=len(selected) + 1, chunk=item.chunk, score=item.score, token_count=cost)
            )
            used_tokens += cost
            continue

        if not selected and remaining > 0:
            truncated_text = _truncate_to_budget(item.chunk.text, remaining)
            truncated_chunk = item.chunk.model_copy(update={"text": truncated_text})
            truncated_cost = count_tokens(truncated_text)
            selected.append(
                PackedChunk(
                    citation_id=1,
                    chunk=truncated_chunk,
                    score=item.score,
                    token_count=truncated_cost,
                    truncated=True,
                )
            )
            used_tokens += truncated_cost

        break

    return PackedContext(
        chunks=_bookend_order(selected),
        total_tokens=used_tokens,
        requested_count=len(ranked),
        dropped_count=len(ranked) - len(selected),
    )


def format_context(context: PackedContext) -> str:
    """Numbered, citation-ready context blocks in display (bookended) order.
    Each block's number is its `citation_id` (rank), independent of its
    position here -- see module docstring."""
    blocks = [f"[{pc.citation_id}] (source: {pc.chunk.id})\n{pc.chunk.text}" for pc in context.chunks]
    return "\n\n---\n\n".join(blocks)
