"""Conversation memory (Phase 14 Unit 4): a follow-up is rewritten into a
standalone query BEFORE the cache, router and retriever see it. Doing it any
later would let "what about the free tier?" hit another conversation's cache
entry, or be routed on a pronoun."""

import pytest
from test_pipeline import _FakeCache, _FixedRetriever, _chunk, _pipeline, Query, Route, _FixedRouter, _RaisingRetriever
from test_pipeline_stream import _StreamLLM

from aegis.events import DoneEvent

_HISTORY = [("user", "how do I declare a query parameter?"), ("assistant", "Use a function argument with a default [1].")]
_FOLLOWUP = "what about making it required?"
_STANDALONE = "how do I make a query parameter required in FastAPI?"


class _Rewriter:
    def __init__(self, out: str = _STANDALONE):
        self.out, self.calls = out, []

    def rewrite(self, history, query):
        self.calls.append((history, query))
        return self.out


class _SpyRetriever(_FixedRetriever):
    def __init__(self, results):
        super().__init__(results)
        self.queries = []

    def search(self, query, top_k, **kw):
        self.queries.append(query)
        return super().search(query, top_k, **kw)


class _SpyRouter(_FixedRouter):
    def __init__(self):
        super().__init__(Route.DOCS_HOWTO)
        self.queries = []

    def classify(self, query):
        self.queries.append(query)
        return super().classify(query)


def _p(rewriter, **kw):
    llm = _StreamLLM(["Use ", "Query(...) [1]."])
    return _pipeline(rewriter=rewriter, llm=llm, **kw)[0]


def test_cache_router_and_retriever_all_see_the_standalone_query():
    exact, sem = _FakeCache(), _FakeCache()
    router, retriever = _SpyRouter(), _SpyRetriever([(_chunk("c1"), 0.9), (_chunk("c2"), 0.5)])
    p = _pipeline(rewriter=_Rewriter(), llm=_StreamLLM(["Use [1]."]), router=router, retriever=retriever,
                  exact_cache=exact, semantic_cache=sem)[0]
    result = p.answer(Query(id="q", text=_FOLLOWUP), history=_HISTORY)
    assert exact.get_calls[0][0] == sem.get_calls[0][0] == _STANDALONE
    assert router.queries == [_STANDALONE] and retriever.queries == [_STANDALONE]
    assert result.query.text == _STANDALONE and result.rewritten_query == _STANDALONE
    assert "rewrite" in result.stage_latencies_ms


def test_no_history_skips_the_rewriter_entirely():
    rw = _Rewriter()
    result = _p(rw).answer(Query(id="q", text="how do I x"))
    assert rw.calls == [] and result.rewritten_query is None
    assert "rewrite" not in result.stage_latencies_ms


def test_unchanged_rewrite_is_not_reported_as_a_rewrite():
    result = _p(_Rewriter("how do I x")).answer(Query(id="q", text="how do I x"), history=_HISTORY)
    assert result.rewritten_query is None


def test_history_without_a_configured_rewriter_is_ignored():
    result = _p(None).answer(Query(id="q", text=_FOLLOWUP), history=_HISTORY)
    assert result.query.text == _FOLLOWUP and result.rewritten_query is None


def test_rewriter_receives_the_history_and_the_raw_followup():
    rw = _Rewriter()
    _p(rw).answer(Query(id="q", text=_FOLLOWUP), history=_HISTORY)
    assert rw.calls == [(_HISTORY, _FOLLOWUP)]


def test_stream_done_event_names_the_rewritten_query():
    events = list(_p(_Rewriter()).answer_stream(Query(id="q", text=_FOLLOWUP), history=_HISTORY))
    done = next(e for e in events if isinstance(e, DoneEvent))
    assert done.rewritten_query == _STANDALONE


def test_rewriter_exception_becomes_error_event_not_a_crash():
    class Boom:
        def rewrite(self, h, q):
            raise RuntimeError("x")

    events = list(_p(Boom()).answer_stream(Query(id="q", text=_FOLLOWUP), history=_HISTORY))
    assert events[-1].type == "error"
