"""Greedy pack-with-overlap: the packing step shared by recursive,
heading-aware, and semantic chunking once each has produced a list of pieces
no larger than `max_tokens` (recursive's word-level fallback, heading-aware's
and semantic's atomic code/sentence segments).

One exception, not an error: a single piece that is itself over `max_tokens`
(a pasted build log, a huge fenced block) is emitted as its own oversized
chunk rather than force-split — see `docs/explainers/02-chunking-strategies.md`
for the two real corpus cases this fires on.

Every packing decision is verified by actually joining the candidate pieces
and re-counting tokens on the real joined string, not by summing each
piece's token count. An earlier version tracked a running sum instead — it
looked right but under-counted the real joined text by the tokens the
`"\n\n"` join separator itself costs (plus, transiently, a separate bug
where pieces already carried an embedded separator, doubling it). Both
showed up as real chunks over `max_tokens` on real corpus documents (issue
#15000, a semantic-chunking oversized-group fallback) — see the design
record. Re-encoding on every piece is cheap here because an accumulator
never exceeds `max_tokens` (~a few hundred tokens), so this isn't the
quadratic-blowup trap it would be for unbounded strings.
"""

from __future__ import annotations

from aegis.chunking.tokens import count_tokens, decode_tokens, encode_tokens

_JOIN = "\n\n"


def _overlap_prefix(chunk_text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    tokens = encode_tokens(chunk_text)
    return decode_tokens(tokens[-overlap_tokens:])


def _join_tokens(parts: list[str]) -> int:
    return count_tokens(_JOIN.join(parts).strip())


def greedy_merge(pieces: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    if overlap_tokens >= max_tokens:
        raise ValueError(f"overlap_tokens ({overlap_tokens}) must be < max_tokens ({max_tokens})")

    chunks: list[str] = []
    current: list[str] = []

    for piece in pieces:
        if count_tokens(piece) > max_tokens:
            if current:
                chunks.append(_JOIN.join(current).strip())
            chunks.append(piece)
            current = []
            continue

        if current and _join_tokens(current + [piece]) > max_tokens:
            flushed = _JOIN.join(current).strip()
            chunks.append(flushed)
            prefix = _overlap_prefix(flushed, overlap_tokens)
            # max_tokens is a hard contract; overlap is best-effort. If the
            # carried-forward tail would itself push this piece over budget,
            # drop the overlap for this one boundary rather than violate it.
            current = [prefix, piece] if prefix and _join_tokens([prefix, piece]) <= max_tokens else [piece]
        else:
            current.append(piece)

    if current:
        chunks.append(_JOIN.join(current).strip())

    return chunks
