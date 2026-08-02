import pytest

from aegis.chunking.merge import greedy_merge
from aegis.chunking.tokens import count_tokens


def test_empty_pieces_returns_empty():
    assert greedy_merge([], max_tokens=100, overlap_tokens=10) == []


def test_pieces_under_budget_merge_into_one_chunk():
    pieces = ["one two three", "four five six"]
    result = greedy_merge(pieces, max_tokens=100, overlap_tokens=0)

    assert len(result) == 1
    assert "one two three" in result[0]
    assert "four five six" in result[0]


def test_splits_when_over_budget():
    # Each piece is a handful of tokens; force a split with a tight budget.
    pieces = [f"word{i} " * 5 for i in range(10)]
    result = greedy_merge(pieces, max_tokens=20, overlap_tokens=0)

    assert len(result) > 1
    for chunk in result:
        assert count_tokens(chunk) <= 20


def test_overlap_carries_tail_of_previous_chunk_forward():
    pieces = [f"alpha{i} beta{i} gamma{i}" for i in range(6)]
    result = greedy_merge(pieces, max_tokens=15, overlap_tokens=5)

    assert len(result) > 1
    # The tail of chunk N should reappear at the start of chunk N+1.
    for prev, nxt in zip(result, result[1:]):
        prev_tail_words = prev.split()[-2:]
        next_head_words = nxt.split()[: len(prev_tail_words)]
        assert any(w in nxt for w in prev_tail_words), (prev_tail_words, next_head_words)


def test_single_oversized_piece_becomes_its_own_chunk_unsplit():
    huge_piece = "x " * 500  # far bigger than max_tokens
    pieces = ["small intro", huge_piece, "small outro"]
    result = greedy_merge(pieces, max_tokens=20, overlap_tokens=5)

    assert huge_piece in result
    oversized = [c for c in result if c == huge_piece]
    assert len(oversized) == 1


def test_overlap_tokens_must_be_smaller_than_max_tokens():
    with pytest.raises(ValueError):
        greedy_merge(["a"], max_tokens=10, overlap_tokens=10)


def test_max_tokens_never_violated_even_when_overlap_prefix_would_push_over():
    # Each piece alone is just under budget; naively prepending the previous
    # chunk's overlap tail to the next piece would push that chunk over
    # max_tokens. The max_tokens contract is hard; overlap is best-effort.
    piece = "word " * 18  # ~18-19 tokens, close to the 20-token budget
    pieces = [piece, piece, piece]
    result = greedy_merge(pieces, max_tokens=20, overlap_tokens=8)

    for chunk in result:
        assert count_tokens(chunk) <= 20
