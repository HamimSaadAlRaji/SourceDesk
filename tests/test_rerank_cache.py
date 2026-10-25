"""The content-addressed reranker score cache (design record 16i). A cross-encoder
score is a pure function of (model, query text, chunk text), so an unchanged pair
never needs recomputing -- and a changed one must never be served a stale score.
The cache is what makes the escalation gate affordable on CPU, so its failure modes
are the gate's: a stale hit would hide a reranker regression, and an unbounded
miss would turn a PR check into a two-hour job."""

from __future__ import annotations

import json

import pytest

from aegis.eval.rerank_cache import ScoreCache, TooManyUncachedPairs, make_score_fn, pair_key


def test_the_key_depends_on_model_query_and_chunk_text():
    base = pair_key("m", "q", "c")
    assert base == pair_key("m", "q", "c")
    assert len({base, pair_key("m2", "q", "c"), pair_key("m", "q2", "c"), pair_key("m", "q", "c2")}) == 4


def test_the_key_cannot_be_forged_by_moving_text_between_fields():
    # ("ab", "c") and ("a", "bc") must not collide.
    assert pair_key("m", "ab", "c") != pair_key("m", "a", "bc")


def test_a_hit_is_served_from_the_cache_and_a_miss_is_computed_and_remembered():
    calls: list[list[tuple[str, str]]] = []

    def inner(pairs):
        calls.append(list(pairs))
        return [float(len(q) + len(c)) for q, c in pairs]

    cache = ScoreCache({pair_key("m", "q1", "c1"): 0.5})
    fn = make_score_fn("m", inner, cache, max_uncached=10)
    assert fn([("q1", "c1"), ("q2", "c2")]) == [0.5, 4.0]
    assert calls == [[("q2", "c2")]]          # only the miss reached the model
    assert fn([("q2", "c2")]) == [4.0]
    assert len(calls) == 1                    # now remembered
    assert (cache.hits, cache.misses) == (2, 1)


def test_results_come_back_in_input_order_with_mixed_hits_and_misses():
    cache = ScoreCache({pair_key("m", "a", "1"): 1.0, pair_key("m", "c", "3"): 3.0})
    fn = make_score_fn("m", lambda pairs: [2.0 for _ in pairs], cache, max_uncached=10)
    assert fn([("a", "1"), ("b", "2"), ("c", "3")]) == [1.0, 2.0, 3.0]


def test_too_many_misses_raises_before_any_model_call():
    called = []
    cache = ScoreCache({})
    fn = make_score_fn("m", lambda pairs: called.append(1) or [0.0] * len(pairs), cache, max_uncached=2)
    with pytest.raises(TooManyUncachedPairs, match="3"):
        fn([("a", "1"), ("b", "2"), ("c", "3")])
    assert called == []


def test_the_miss_budget_is_cumulative_across_calls():
    cache = ScoreCache({})
    fn = make_score_fn("m", lambda pairs: [0.0] * len(pairs), cache, max_uncached=2)
    fn([("a", "1"), ("b", "2")])
    with pytest.raises(TooManyUncachedPairs):
        fn([("c", "3")])


def test_a_model_change_misses_everything():
    cache = ScoreCache({pair_key("old-model", "q", "c"): 0.9})
    fn = make_score_fn("new-model", lambda pairs: [0.1] * len(pairs), cache, max_uncached=10)
    assert fn([("q", "c")]) == [0.1]
    assert cache.misses == 1


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "scores.json"
    cache = ScoreCache({pair_key("m", "q", "c"): 0.25})
    cache.save(path)
    assert ScoreCache.load(path).scores == cache.scores
    assert json.loads(path.read_text())["scores"]  # reviewable plain JSON


def test_a_missing_cache_file_is_empty_not_an_error(tmp_path):
    assert ScoreCache.load(tmp_path / "nope.json").scores == {}


def test_misses_record_which_pairs_were_not_in_the_cache():
    # A miss on a CI runner (a different CPU reordered a near-tie) must say WHICH pair, or the
    # cache cannot be repaired without guessing.
    cache = ScoreCache({pair_key("m", "q1", "c1"): 0.5})
    fn = make_score_fn("m", lambda pairs: [0.0] * len(pairs), cache, max_uncached=10)
    fn([("q1", "c1"), ("q2", "c2")])
    assert cache.missed == [pair_key("m", "q2", "c2")]
