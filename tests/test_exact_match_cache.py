"""TDD for the exact-match response cache (plan.md Phase 12, step 1).
Deterministic, logic-heavy (normalization + key construction decide
correctness silently -- a bug here serves the wrong cached answer to a
real user, same risk class CLAUDE.md names TDD for), so covered here with
a real sqlite file per test (`tmp_path`), no fakes needed.
"""

from aegis.cache.exact_match import CachedAnswer, ExactMatchCache, normalize_query


def _answer(text: str = "answer text") -> CachedAnswer:
    return CachedAnswer(text=text, citations=["doc::a::1"], provider="groq")


# --- normalize_query ---


def test_normalize_query_lowercases():
    assert normalize_query("How Do I Use Depends?") == normalize_query("how do i use depends?")


def test_normalize_query_collapses_whitespace():
    assert normalize_query("how   do i\nuse Depends?") == normalize_query("how do i use Depends?")


def test_normalize_query_strips_leading_trailing_whitespace():
    assert normalize_query("  how do i use Depends?  ") == normalize_query("how do i use Depends?")


# --- ExactMatchCache ---


def test_set_then_get_roundtrips(tmp_path):
    cache = ExactMatchCache(tmp_path / "cache.sqlite")
    answer = _answer()

    cache.set("how do I use Depends?", corpus_version="v1", answer=answer)
    hit = cache.get("how do I use Depends?", corpus_version="v1")

    assert hit is not None
    assert hit.text == answer.text
    assert hit.citations == answer.citations
    assert hit.provider == answer.provider


def test_get_missing_query_returns_none(tmp_path):
    cache = ExactMatchCache(tmp_path / "cache.sqlite")
    assert cache.get("nothing cached for this", corpus_version="v1") is None


def test_get_is_case_and_whitespace_insensitive(tmp_path):
    cache = ExactMatchCache(tmp_path / "cache.sqlite")
    cache.set("How do I use Depends?", corpus_version="v1", answer=_answer())

    hit = cache.get("  how do i use depends?  ", corpus_version="v1")

    assert hit is not None


def test_same_query_different_corpus_version_is_a_miss(tmp_path):
    # The whole point of keying on corpus_version: a reindex must not serve
    # a response generated against the old corpus/config.
    cache = ExactMatchCache(tmp_path / "cache.sqlite")
    cache.set("how do I use Depends?", corpus_version="v1", answer=_answer())

    assert cache.get("how do I use Depends?", corpus_version="v2") is None


def test_different_queries_same_corpus_version_are_independent(tmp_path):
    cache = ExactMatchCache(tmp_path / "cache.sqlite")
    cache.set("query a", corpus_version="v1", answer=_answer(text="answer a"))
    cache.set("query b", corpus_version="v1", answer=_answer(text="answer b"))

    assert cache.get("query a", corpus_version="v1").text == "answer a"
    assert cache.get("query b", corpus_version="v1").text == "answer b"


def test_set_overwrites_an_existing_entry(tmp_path):
    cache = ExactMatchCache(tmp_path / "cache.sqlite")
    cache.set("query a", corpus_version="v1", answer=_answer(text="first"))
    cache.set("query a", corpus_version="v1", answer=_answer(text="second"))

    assert cache.get("query a", corpus_version="v1").text == "second"


def test_persists_across_instances_reopening_the_same_path(tmp_path):
    path = tmp_path / "cache.sqlite"
    ExactMatchCache(path).set("query a", corpus_version="v1", answer=_answer(text="persisted"))

    reopened = ExactMatchCache(path)

    assert reopened.get("query a", corpus_version="v1").text == "persisted"
