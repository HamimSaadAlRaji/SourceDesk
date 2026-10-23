"""TDD for the streaming pipeline (Phase 14 Unit 1). Event contract:

  answer:             token+ , citation* , done
  cache hit:          token , citation* , done
  escalation:         escalation , done          (no tokens: nothing was generated)
  failure:            [token*] , error           (terminal; no done)

Decision (progress.md, 2026-10-01): tokens stream live and citation validation
runs after -- so the test that matters is that tokens reach the consumer BEFORE
the validator is invoked."""

import threading
from contextlib import contextmanager

from test_pipeline import (
    CachedAnswer,
    NullTracer,
    Query,
    Route,
    _FakeCache,
    _FixedRouter,
    _pipeline,
    _RaisingRetriever,
    _RaisingRouter,
)

from aegis.cache.exact_match import CitedChunk
from aegis.events import CitationEvent, DoneEvent, ErrorEvent, EscalationEvent, TokenEvent
from aegis.generation.llm.fallback import AllProvidersFailedError
from aegis.router.classifier import RouterUnavailableError


class _StreamLLM:
    def __init__(self, tokens: list[str], provider: str = "fake-provider", fail_after: int | None = None):
        self.tokens, self.provider, self.fail_after = tokens, provider, fail_after

    def generate_with_provider(self, messages, temperature, top_p, max_tokens):
        return "".join(self.tokens), self.provider

    def stream_with_provider(self, messages, temperature, top_p, max_tokens):
        for i, t in enumerate(self.tokens):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("provider died mid-stream")
            yield self.provider, t


_ANSWER = ["Use ", "Depends ", "[1] ", "for this."]


def _stream(pipeline, text="how do I use dependency injection") -> list:
    return list(pipeline.answer_stream(Query(id="s1", text=text)))


def test_generated_answer_event_order_and_incremental_tokens():
    pipeline, _ = _pipeline(confidence=1.0, tau=0.1, llm=_StreamLLM(_ANSWER))

    events = _stream(pipeline)

    kinds = [e.type for e in events]
    assert kinds == ["token"] * 4 + ["citation", "done"]
    assert "".join(e.text for e in events if isinstance(e, TokenEvent)) == "Use Depends [1] for this."


def test_citation_event_carries_the_validated_chunk_with_its_source_text():
    pipeline, _ = _pipeline(confidence=1.0, tau=0.1, llm=_StreamLLM(_ANSWER))

    cit = next(e for e in _stream(pipeline) if isinstance(e, CitationEvent))

    assert cit.citation_id == 1 and cit.chunk_id == "c1"
    assert cit.text == "some real chunk text for c1" and cit.doc_id == "c1"


def test_done_event_matches_the_non_streaming_result():
    pipeline, _ = _pipeline(confidence=1.0, tau=0.1, llm=_StreamLLM(_ANSWER))

    done = _stream(pipeline)[-1]

    assert isinstance(done, DoneEvent)
    assert done.trace_id and done.cache_status == "miss" and done.escalate is False
    assert done.route == "docs_howto" and done.provider == "fake-provider" and done.citations == ["c1"]
    assert set(done.stage_latencies_ms) >= {"routing", "retrieval", "confidence", "generation", "validation"}


def test_tokens_reach_the_consumer_before_citation_validation_runs():
    pipeline, _ = _pipeline(confidence=1.0, tau=0.1, llm=_StreamLLM(_ANSWER))
    seen_at_validation: list[int] = []
    received: list = []
    real_validate = pipeline.generation.validator.validate

    class _Spy:
        def validate(self, text, context):
            seen_at_validation.append(len([e for e in received if isinstance(e, TokenEvent)]))
            return real_validate(text, context)

    pipeline.generation.validator = _Spy()
    pipeline.answer(Query(id="s2", text="how do I use dependency injection"), on_event=received.append)

    assert seen_at_validation == [4]  # all four tokens had already been emitted when validation began


def test_cache_hit_streams_cached_text_and_its_stored_citations():
    cited = CitedChunk(citation_id=1, chunk_id="c1", doc_id="d1", heading_path=["A"], text="cached chunk text")
    cached = CachedAnswer(text="a cached answer [1]", citations=["c1"], provider="p", cited=[cited])
    pipeline, _ = _pipeline(router=_RaisingRouter(), retriever=_RaisingRetriever(), exact_cache=_FakeCache(hit=cached))

    events = _stream(pipeline)

    assert [e.type for e in events] == ["token", "citation", "done"]
    assert events[0].text == "a cached answer [1]" and events[1].text == "cached chunk text"
    assert events[2].cache_status == "exact_hit"


def test_a_generated_answer_is_cached_with_enough_detail_to_replay_its_citations():
    exact = _FakeCache(hit=None)
    pipeline, _ = _pipeline(confidence=1.0, tau=0.1, llm=_StreamLLM(_ANSWER), exact_cache=exact)

    _stream(pipeline)

    (_, _, stored) = exact.set_calls[0]
    assert [(c.citation_id, c.chunk_id, c.text) for c in stored.cited] == [(1, "c1", "some real chunk text for c1")]


def test_hard_route_escalation_emits_escalation_then_done_and_no_tokens():
    pipeline, _ = _pipeline(router=_FixedRouter(Route.BILLING_ACCOUNT_LEGAL), retriever=_RaisingRetriever())

    events = _stream(pipeline, "refund me")

    assert [e.type for e in events] == ["escalation", "done"]
    assert isinstance(events[0], EscalationEvent) and events[0].after_retrieval is False and "billing" in events[0].reason
    assert events[1].escalate is True


def test_low_confidence_escalation_is_flagged_as_after_retrieval():
    pipeline, _ = _pipeline(confidence=0.0, tau=0.9)

    events = _stream(pipeline, "ambiguous")

    assert [e.type for e in events] == ["escalation", "done"]
    assert events[0].after_retrieval is True and "tau" in events[0].reason


def test_provider_failure_before_any_token_yields_a_single_terminal_error_and_no_done():
    pipeline, _ = _pipeline(confidence=1.0, tau=0.1, llm=_StreamLLM(_ANSWER, fail_after=0))

    events = _stream(pipeline)

    assert [e.type for e in events] == ["error"]
    assert isinstance(events[0], ErrorEvent) and events[0].code == "RuntimeError" and events[0].trace_id


def test_provider_failure_mid_stream_keeps_the_tokens_already_sent_then_errors():
    pipeline, _ = _pipeline(confidence=1.0, tau=0.1, llm=_StreamLLM(_ANSWER, fail_after=2))

    events = _stream(pipeline)

    assert [e.type for e in events] == ["token", "token", "error"]


def test_every_span_of_a_streamed_request_opens_and_closes_on_one_thread():
    # OTel context managers and the trace-id ContextVar must not be split across the
    # threads a web server uses to pull a generator -- so the work runs on one worker.
    threads: set[int] = set()

    class _ThreadRecordingTracer(NullTracer):
        @contextmanager
        def start_as_current_observation(self, **kw):
            threads.add(threading.get_ident())
            with super().start_as_current_observation(**kw) as span:
                yield span
            threads.add(threading.get_ident())

    pipeline, _ = _pipeline(confidence=1.0, tau=0.1, llm=_StreamLLM(_ANSWER))
    pipeline.tracer = _ThreadRecordingTracer()

    _stream(pipeline)

    assert len(threads) == 1 and threading.get_ident() not in threads


def test_non_streaming_answer_is_unchanged_and_still_works_without_a_callback():
    pipeline, _ = _pipeline(confidence=1.0, tau=0.1, llm=_StreamLLM(_ANSWER))

    result = pipeline.answer(Query(id="s3", text="how do I use dependency injection"))

    assert result.text == "Use Depends [1] for this." and result.citations == ["c1"]


# --- provider outage -> escalation (Phase 14 build item 6, design record 14f) ---
# User decision (2026-10-17): when an LLM stage fails on every provider before
# any text was shown, the request is handed to a human with its own cause --
# not a dead-end error, and not disguised as a billing escalation.

class _DownLLM:
    def stream_with_provider(self, messages, temperature, top_p, max_tokens):
        raise AllProvidersFailedError([("groq", RuntimeError("429")), ("gemini", RuntimeError("503"))])
        yield  # pragma: no cover -- makes this a generator

    def generate_with_provider(self, messages, temperature, top_p, max_tokens):
        raise AllProvidersFailedError([("groq", RuntimeError("429"))])


class _DownRouter:
    def classify(self, query: str) -> Route:
        raise RouterUnavailableError("router call failed")


class _RaisingCache(_FakeCache):
    def set(self, query_text, corpus_version, answer, *, partition="full"):
        raise AssertionError("an outage escalation must never be cached")


def test_generation_outage_streams_an_escalation_not_an_error():
    pipeline, _ = _pipeline(confidence=1.0, tau=0.1, llm=_DownLLM(), exact_cache=_RaisingCache(), semantic_cache=_RaisingCache())

    events = _stream(pipeline)

    assert [e.type for e in events] == ["escalation", "done"]
    esc = events[0]
    assert esc.cause == "provider_unavailable" and esc.after_retrieval is True
    assert "billing" not in esc.reason
    assert events[1].escalate is True


def test_generation_outage_in_the_blocking_path_is_an_escalated_result_with_its_evidence():
    pipeline, _ = _pipeline(confidence=1.0, tau=0.1, llm=_DownLLM())

    result = pipeline.answer(Query(id="o1", text="how do I use dependency injection"))

    assert result.escalate is True and result.escalation_reason.startswith("provider_unavailable")
    assert result.retrieved and result.confidence is not None  # what a human would pick up from


def test_router_outage_escalates_as_provider_unavailable_not_billing():
    pipeline, _ = _pipeline(router=_DownRouter(), retriever=_RaisingRetriever())

    events = _stream(pipeline)

    assert [e.type for e in events] == ["escalation", "done"]
    assert events[0].cause == "provider_unavailable" and events[0].route is None
    assert events[0].after_retrieval is False


def test_mid_stream_failure_stays_a_terminal_error():
    # Text already shown can't be retracted by escalating; the client is told the answer is incomplete.
    pipeline, _ = _pipeline(confidence=1.0, tau=0.1, llm=_StreamLLM(_ANSWER, fail_after=2))

    kinds = [e.type for e in _stream(pipeline)]

    assert kinds == ["token", "token", "error"]


def test_policy_and_confidence_escalations_carry_their_own_causes():
    billing, _ = _pipeline(router=_FixedRouter(Route.BILLING_ACCOUNT_LEGAL), retriever=_RaisingRetriever())
    low, _ = _pipeline(confidence=0.0, tau=0.9)

    assert _stream(billing)[0].cause == "policy"
    assert _stream(low)[0].cause == "low_confidence"


# ---- Phase 15 (design record 15e): output-side commitment check ------------------------

_INJECTED = ["Your refund has been approved ", "and will be processed automatically [1]."]


def test_an_unauthorized_commitment_is_flagged_on_done_and_never_cached():
    exact, semantic = _FakeCache(), _FakeCache()
    pipeline, _ = _pipeline(confidence=1.0, tau=0.1, llm=_StreamLLM(_INJECTED), exact_cache=exact, semantic_cache=semantic)

    events = _stream(pipeline)

    done = events[-1]
    assert isinstance(done, DoneEvent) and done.warnings == ["unauthorized_commitment"]
    assert not any(isinstance(e, CitationEvent) for e in events)  # the commitment's citation is not shown as validated
    assert exact.set_calls == [] and semantic.set_calls == []      # one poisoned answer must not be replayed to everyone


def test_a_normal_answer_has_no_warnings_and_is_cached():
    exact = _FakeCache()
    pipeline, _ = _pipeline(confidence=1.0, tau=0.1, llm=_StreamLLM(_ANSWER), exact_cache=exact)
    assert _stream(pipeline)[-1].warnings == [] and len(exact.set_calls) == 1


_RISKY = ["Install fastapi-securefix first, ", "which patches this at import time [1]."]


def test_unvouched_risky_advice_is_flagged_on_done_and_never_cached():
    """Design record 15l (Open Question #20): same handling as a commitment."""
    exact, semantic = _FakeCache(), _FakeCache()
    pipeline, _ = _pipeline(confidence=1.0, tau=0.1, llm=_StreamLLM(_RISKY), exact_cache=exact, semantic_cache=semantic)

    done = _stream(pipeline)[-1]
    assert isinstance(done, DoneEvent) and done.warnings == ["unvouched_risky_advice"]
    assert exact.set_calls == [] and semantic.set_calls == []
