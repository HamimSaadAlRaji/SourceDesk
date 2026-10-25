"""Phase 17: the LangGraph orchestration must be indistinguishable from the direct one.

The strongest check is differential: the same fakes through both classes, then the
whole `PipelineResult` (minus the two fields that are per-run by construction, trace id
and latencies), the span sequence, the cache writes and the streamed events must match.
The retry edge is graph-only and off by default, so it gets its own tests."""

import pytest
from test_pipeline import (
    _IDENTITY_BOUNDS,
    _RERANKER_ONLY_WEIGHTS,
    CachedAnswer,
    Query,
    Route,
    _chunk,
    _fake_encode_fn,
    _FakeCache,
    _FakeLLM,
    _FakeReranker,
    _FixedRouter,
    _pipeline,
    _RaisingRetriever,
    _RaisingRouter,
)
from test_pipeline_stream import _StreamLLM

from aegis.generation.llm.fallback import AllProvidersFailedError
from aegis.graph.pipeline import LangGraphPipeline
from aegis.guardrails.confidence import ConfidenceScorer
from aegis.pipeline import AnswerPipeline
from aegis.router.classifier import RouterUnavailableError

_Q = "how do I use dependency injection"


class _UnavailableRouter:
    def classify(self, query):
        raise RouterUnavailableError("router call failed")


class _DeadLLM:
    def generate_with_provider(self, messages, temperature, top_p, max_tokens):
        raise AllProvidersFailedError([("openai", RuntimeError("429"))])

    def stream_with_provider(self, messages, temperature, top_p, max_tokens):
        raise AllProvidersFailedError([("openai", RuntimeError("429"))])


_CACHED = CachedAnswer(text="a cached answer", citations=[], provider="fake-provider")

# name -> factory of fresh kwargs (fakes are stateful, so each backend gets its own).
_SCENARIOS = {
    "exact_cache_hit": lambda: dict(router=_RaisingRouter(), retriever=_RaisingRetriever(), exact_cache=_FakeCache(_CACHED)),
    "semantic_cache_hit": lambda: dict(router=_RaisingRouter(), retriever=_RaisingRetriever(), semantic_cache=_FakeCache(_CACHED)),
    "policy_escalation": lambda: dict(router=_FixedRouter(Route.BILLING_ACCOUNT_LEGAL), retriever=_RaisingRetriever()),
    "router_unavailable": lambda: dict(router=_UnavailableRouter(), retriever=_RaisingRetriever()),
    "low_confidence_escalation": lambda: dict(confidence=0.0, tau=0.9),
    "generated_answer": lambda: dict(confidence=1.0, tau=0.1, llm=_FakeLLM("a real generated answer [1]")),
    "generation_unavailable": lambda: dict(confidence=1.0, tau=0.1, llm=_DeadLLM()),
}


def _run(cls, scenario, **extra):
    kwargs = _SCENARIOS[scenario]()
    exact = kwargs.setdefault("exact_cache", _FakeCache())
    semantic = kwargs.setdefault("semantic_cache", _FakeCache())
    pipeline, tracer = _pipeline(cls=cls, **kwargs, **extra)
    result = pipeline.answer(Query(id="q1", text=_Q))
    return result, tracer, exact, semantic


@pytest.mark.parametrize("scenario", list(_SCENARIOS))
def test_graph_result_matches_direct_result(scenario):
    direct, d_tracer, d_exact, d_sem = _run(AnswerPipeline, scenario)
    graph, g_tracer, g_exact, g_sem = _run(LangGraphPipeline, scenario)

    per_run = {"trace_id", "stage_latencies_ms"}
    assert graph.model_dump(exclude=per_run) == direct.model_dump(exclude=per_run)
    assert set(graph.stage_latencies_ms) == set(direct.stage_latencies_ms)
    assert [s["name"] for s in g_tracer.spans] == [s["name"] for s in d_tracer.spans]
    assert g_exact.set_calls == d_exact.set_calls and g_sem.set_calls == d_sem.set_calls


def _stream(cls, scenario):
    kwargs = _SCENARIOS[scenario]()
    pipeline, _ = _pipeline(cls=cls, **kwargs)
    return [e.model_dump(exclude={"trace_id", "stage_latencies_ms"}) for e in pipeline.answer_stream(Query(id="s1", text=_Q))]


@pytest.mark.parametrize("scenario", ["exact_cache_hit", "policy_escalation", "low_confidence_escalation"])
def test_graph_streams_the_same_events_as_direct(scenario):
    assert _stream(LangGraphPipeline, scenario) == _stream(AnswerPipeline, scenario)


def test_graph_streams_tokens_then_citations_then_done_on_the_generate_path():
    def build(cls):
        pipeline, _ = _pipeline(cls=cls, confidence=1.0, tau=0.1, llm=_StreamLLM(["Use ", "Depends ", "[1] ", "for this."]))
        return [e.model_dump(exclude={"trace_id", "stage_latencies_ms"}) for e in pipeline.answer_stream(Query(id="s2", text=_Q))]

    graph = build(LangGraphPipeline)
    assert [e["type"] for e in graph] == ["token"] * 4 + ["citation", "done"]
    assert graph == build(AnswerPipeline)


class _Rewriter:
    def rewrite(self, history, query):
        return "how do I make a query parameter required in FastAPI?"


def test_graph_rewrites_followups_before_the_cache_and_reports_it():
    history = [("user", "how do I declare a query parameter?"), ("assistant", "Use a default [1].")]

    def build(cls):
        pipeline, tracer = _pipeline(cls=cls, confidence=1.0, tau=0.1, llm=_FakeLLM("answer [1]"), rewriter=_Rewriter())
        result = pipeline.answer(Query(id="h1", text="what about making it required?"), history=history)
        return result, tracer

    (g, g_tracer), (d, d_tracer) = build(LangGraphPipeline), build(AnswerPipeline)
    assert g.rewritten_query == d.rewritten_query == "how do I make a query parameter required in FastAPI?"
    assert g.model_dump(exclude={"trace_id", "stage_latencies_ms"}) == d.model_dump(exclude={"trace_id", "stage_latencies_ms"})
    assert "rewrite" in g.stage_latencies_ms
    assert [s["name"] for s in g_tracer.spans] == [s["name"] for s in d_tracer.spans]


# --- the retry edge (graph only, off by default) -------------------------------------


class _ByQueryRetriever:
    """Weak chunk for the original question, strong chunks once it is reformulated."""

    def __init__(self):
        self.queries: list[str] = []

    def search(self, query, top_k, **kwargs):
        self.queries.append(query)
        if query == "reformulated question":
            return [(_chunk("c1"), 0.9), (_chunk("c2"), 0.5)]
        return [(_chunk("weak"), 0.9), (_chunk("c2"), 0.5)]


class _Reformulator:
    def __init__(self, out="reformulated question"):
        self.out, self.calls = out, []

    def reformulate(self, query):
        self.calls.append(query)
        return self.out


def _scorer():
    return ConfidenceScorer(
        reranker=_FakeReranker({"weak": 0.0, "c1": 1.0, "c2": 0.0}),
        encode_fn=_fake_encode_fn(),
        weights=_RERANKER_ONLY_WEIGHTS,
        bounds=_IDENTITY_BOUNDS,
        rerank_pool_size=3,
    )


def _retry_pipeline(retriever, reformulator, **extra):
    pipeline, tracer = _pipeline(cls=LangGraphPipeline, retriever=retriever, llm=_FakeLLM("answer [1]"), tau=0.5, **extra)
    pipeline.confidence_scorer = _scorer()
    pipeline.reformulator = reformulator
    return pipeline, tracer


def test_retry_is_off_by_default_so_low_confidence_escalates_without_reformulating():
    retriever, reformulator = _ByQueryRetriever(), _Reformulator()
    pipeline, _ = _retry_pipeline(retriever, reformulator)

    result = pipeline.answer(Query(id="r1", text=_Q))

    assert result.escalate is True and reformulator.calls == [] and retriever.queries == [_Q]


def test_retry_reformulates_once_and_answers_when_the_second_retrieval_is_confident():
    retriever, reformulator = _ByQueryRetriever(), _Reformulator()
    pipeline, tracer = _retry_pipeline(retriever, reformulator, retry_on_low_confidence=True)

    result = pipeline.answer(Query(id="r2", text=_Q))

    assert result.escalate is False
    assert reformulator.calls == [_Q] and retriever.queries == [_Q, "reformulated question"]
    assert [r.chunk.id for r in result.retrieved][0] == "c1"
    # The user asked _Q: the answer and the cache entry are about _Q, not the retry's wording.
    assert result.query.text == _Q
    assert pipeline.exact_cache.set_calls[0][0] == _Q
    assert "reformulation" in result.stage_latencies_ms
    assert [s["name"] for s in tracer.spans].count("retrieval") == 2


def test_retry_is_capped_so_a_still_weak_second_retrieval_escalates():
    class _AlwaysWeak(_ByQueryRetriever):
        def search(self, query, top_k, **kwargs):
            self.queries.append(query)
            return [(_chunk("weak"), 0.9), (_chunk("c2"), 0.5)]

    retriever, reformulator = _AlwaysWeak(), _Reformulator()
    pipeline, _ = _retry_pipeline(retriever, reformulator, retry_on_low_confidence=True, max_retries=1)

    result = pipeline.answer(Query(id="r3", text=_Q))

    assert result.escalate is True
    assert len(reformulator.calls) == 1 and len(retriever.queries) == 2


def test_retry_never_fires_on_a_policy_escalation():
    reformulator = _Reformulator()
    pipeline, _ = _retry_pipeline(
        _RaisingRetriever(), reformulator, router=_FixedRouter(Route.BILLING_ACCOUNT_LEGAL), retry_on_low_confidence=True
    )

    result = pipeline.answer(Query(id="r4", text="refund me"))

    assert result.escalate is True and reformulator.calls == []


# --- backend selection ---------------------------------------------------------------


def test_pipeline_class_follows_the_backend_argument():
    from aegis.factory import pipeline_class

    assert pipeline_class("direct") is AnswerPipeline
    assert pipeline_class("langgraph") is LangGraphPipeline


def test_pipeline_class_defaults_to_the_configured_backend(monkeypatch):
    from aegis.config import settings
    from aegis.factory import pipeline_class

    monkeypatch.setattr(settings.orchestration, "backend", "langgraph")
    assert pipeline_class() is LangGraphPipeline


def test_pipeline_class_rejects_an_unknown_backend():
    from aegis.factory import pipeline_class

    with pytest.raises(ValueError, match="crewai"):
        pipeline_class("crewai")
