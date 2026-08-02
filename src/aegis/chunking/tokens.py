"""Token counting for chunk-size decisions.

`tiktoken`'s `cl100k_base` is a proxy, not the tokenizer of whatever encoder
Phase 3's bake-off picks (bge-large and BGE-M3 use WordPiece/SentencePiece,
not a GPT BPE). It's used anyway because chunk *boundaries* only need a
stable, fast, dependency-light token count to enforce `max_tokens` and
report distributions — not exact parity with a specific encoder's sequence
length. Loading an actual embedding model just to count tokens would mean
every chunker pulls in ONNX runtime + model weights for a number that only
has to be consistent with itself across strategies.
"""

from __future__ import annotations

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def encode_tokens(text: str) -> list[int]:
    return _ENCODING.encode(text, disallowed_special=())


def decode_tokens(tokens: list[int]) -> str:
    return _ENCODING.decode(tokens)


def count_tokens(text: str) -> int:
    return len(encode_tokens(text))


def _is_continuation_byte(byte: int) -> bool:
    return (byte & 0xC0) == 0x80


def is_safe_boundary(tokens: list[int], index: int) -> bool:
    """True if cutting `tokens` at `index` never separates a multi-byte
    UTF-8 character's lead byte from its continuation bytes.

    `cl100k_base` is a *byte*-level BPE: a single character like "🎉" can be
    split across multiple tokens (confirmed empirically — 3 tokens, at a
    byte boundary that isn't a character boundary). Slicing the token list
    at an arbitrary index and decoding each half independently then silently
    corrupts the text at the cut: `tiktoken.decode()` uses `errors="replace"`
    and returns U+FFFD instead of raising, so a naive fixed-size chunker
    would inject garbage characters at unpredictable chunk boundaries with
    no error to catch it.
    """
    if index <= 0 or index >= len(tokens):
        return True
    first_byte = _ENCODING.decode_single_token_bytes(tokens[index])[:1]
    return not first_byte or not _is_continuation_byte(first_byte[0])


def nearest_safe_boundary(tokens: list[int], index: int) -> int:
    """Nearest safe split point at or before `index`."""
    while not is_safe_boundary(tokens, index):
        index -= 1
    return index
