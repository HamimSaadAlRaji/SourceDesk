"""TDD for the context budgeter (plan.md Phase 9, step 2). A silent bug here
either starves generation of relevant context or blows the token budget --
both corrupt every downstream faithfulness measurement, the same reasoning
CLAUDE.md gives for TDD-ing the analyzer/chunkers/fusion.
"""

from aegis.chunking.tokens import count_tokens
from aegis.generation.budgeter import format_context, pack_context
from aegis.types import Chunk, DocType, Retrieved


def _chunk(id: str, text: str) -> Chunk:
    return Chunk(
        id=id,
        doc_id=id,
        text=text,
        doc_type=DocType.DOCUMENTATION,
        content_hash="hash",
        token_count=count_tokens(text),
        embedding_text=text,
    )


def _retrieved(id: str, text: str, rank: int, score: float = 1.0) -> Retrieved:
    return Retrieved(chunk=_chunk(id, text), score=score, rank=rank)


def test_pack_context_keeps_everything_when_budget_is_generous():
    items = [_retrieved("a", "alpha bravo charlie", 1), _retrieved("b", "delta echo foxtrot", 2)]

    packed = pack_context(items, budget_tokens=1000)

    assert {pc.chunk.id for pc in packed.chunks} == {"a", "b"}
    assert packed.dropped_count == 0
    assert packed.total_tokens == count_tokens("alpha bravo charlie") + count_tokens("delta echo foxtrot")


def test_pack_context_stops_at_first_chunk_that_does_not_fit_strict_prefix():
    a_text = "alpha " * 50  # rank 1, deliberately large
    b_text = "bravo " * 50  # rank 2, same size -- won't fit after `a`
    c_text = "c"  # rank 3, tiny -- would fit in leftover space, must NOT be pulled forward

    a_cost = count_tokens(a_text)
    items = [_retrieved("a", a_text, 1), _retrieved("b", b_text, 2), _retrieved("c", c_text, 3)]

    # Budget fits `a` exactly, with a little room left over -- not enough for `b`,
    # but easily enough for the tiny `c`.
    packed = pack_context(items, budget_tokens=a_cost + 2)

    assert [pc.chunk.id for pc in packed.chunks] == ["a"]
    assert packed.dropped_count == 2


def test_pack_context_truncates_oversized_top_ranked_chunk_instead_of_dropping_it():
    huge_text = "word " * 2000  # far larger than any realistic budget
    items = [_retrieved("only", huge_text, 1)]

    packed = pack_context(items, budget_tokens=25)

    assert len(packed.chunks) == 1
    assert packed.chunks[0].truncated is True
    assert packed.chunks[0].token_count <= 25
    assert packed.dropped_count == 0
    # Truncated text is a real prefix of the original, not empty or mangled.
    assert huge_text.startswith(packed.chunks[0].chunk.text[:10])


def test_pack_context_drops_everything_when_budget_is_zero():
    items = [_retrieved("a", "alpha bravo", 1)]

    packed = pack_context(items, budget_tokens=0)

    assert packed.chunks == []
    assert packed.dropped_count == 1


def test_pack_context_sorts_by_rank_field_not_input_list_order():
    # Passed in reverse-rank order on purpose -- pack_context must not trust
    # positional order.
    items = [_retrieved("second", "bravo", 2), _retrieved("first", "alpha", 1)]

    packed = pack_context(items, budget_tokens=1000)

    citation_by_id = {pc.chunk.id: pc.citation_id for pc in packed.chunks}
    assert citation_by_id["first"] == 1
    assert citation_by_id["second"] == 2


def test_pack_context_bookend_order_puts_worst_survivor_in_the_middle():
    items = [_retrieved(str(i), f"word{i}", i) for i in range(1, 6)]  # ranks 1..5, all tiny

    packed = pack_context(items, budget_tokens=1000)

    display_order = [pc.chunk.id for pc in packed.chunks]
    # rank1 first, rank2 last, rank3 second, rank4 second-to-last, rank5 (worst) dead center.
    assert display_order == ["1", "3", "5", "4", "2"]


def test_pack_context_citation_ids_track_rank_not_display_position():
    items = [_retrieved(str(i), f"word{i}", i) for i in range(1, 6)]

    packed = pack_context(items, budget_tokens=1000)

    for pc in packed.chunks:
        assert pc.citation_id == int(pc.chunk.id)


def test_format_context_includes_citation_markers_and_source_ids():
    items = [_retrieved("doc::42", "how to use dependency injection", 1)]
    packed = pack_context(items, budget_tokens=1000)

    text = format_context(packed)

    assert "[1]" in text
    assert "doc::42" in text
    assert "how to use dependency injection" in text


def test_pack_context_empty_input():
    packed = pack_context([], budget_tokens=1000)

    assert packed.chunks == []
    assert packed.total_tokens == 0
    assert packed.dropped_count == 0
