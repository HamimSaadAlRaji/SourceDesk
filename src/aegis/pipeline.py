"""The end-to-end request pipeline (plan.md Phase 13, Unit 2): the first
place every independently-built, independently-tested stage (cache,
router, retrieval, confidence, generation, citation validation) gets
composed into one real `answer(query)` call.

Built here, in Phase 13, not Phase 14 -- Phase 13's own tracing and
audit-trail work needs a real request to attach spans to, and none existed
anywhere in this codebase before this unit (every prior phase tested its
stage in isolation, wired together only ad hoc inside experiment scripts).
Phase 14 wraps this exact object behind FastAPI/SSE rather than rebuilding
it (design record 13b SS3 -- confirmed with the user before building this).

Stage order matches the architecture diagram referenced directly in
`router/classifier.py`'s own docstring: cache -> router -> retrieval ->
confidence -> generation. Every collaborator is dependency-injected,
including the tracer, so this is unit-testable with fakes
(`tests/test_pipeline.py`) the same way `GenerationPipeline`/
`RerankedRetriever` are; the real path is exercised by
`experiments/13b_pipeline_live_trace.py`.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel

from aegis.cache.exact_match import CachedAnswer, CitedChunk
from aegis.config import settings
from aegis.events import CitationEvent, DoneEvent, ErrorEvent, EscalationEvent, PipelineEvent, TokenEvent
from aegis.observability.logging import bind_trace_id, get_logger, log_event
from aegis.generation.pipeline import GenerationPipeline
from aegis.guardrails.confidence import (
    ConfidenceResult,
    ConfidenceScorer,
    EscalationPayload,
    build_escalation_payload,
    decide_escalation,
)
from aegis.query.rewrite import History
from aegis.router.classifier import Route
from aegis.router.policy import decide as decide_route
from aegis.types import Chunk, Query, Retrieved


class CacheStatus(StrEnum):
    EXACT_HIT = "exact_hit"
    SEMANTIC_HIT = "semantic_hit"
    MISS = "miss"


class _Cache(Protocol):
    def get(self, query_text: str, corpus_version: str) -> CachedAnswer | None: ...
    def set(self, query_text: str, corpus_version: str, answer: CachedAnswer) -> None: ...


class _Router(Protocol):
    def classify(self, query: str) -> Route: ...


class _Retriever(Protocol):
    def search(self, query: str, top_k: int, **kwargs) -> list[tuple[Chunk, float]]: ...


class _Rewriter(Protocol):
    def rewrite(self, history: list[tuple[str, str]], query: str) -> str: ...


class _Span(Protocol):
    def update(self, **fields) -> None: ...


class _Tracer(Protocol):
    def start_as_current_observation(self, *, name: str, as_type: str = "span", **kwargs) -> _Span: ...
    def create_trace_id(self) -> str: ...


class ContextEntry(BaseModel):
    """One chunk exactly as shown to the LLM: its `[n]` citation number, which
    chunk that number means, and whether the token budget truncated it. Without
    this an auditor reading "...[1]" cannot tell what [1] was."""

    citation_id: int
    chunk_id: str
    content_hash: str  # lets a later audit detect the source text changed after this answer
    score: float
    token_count: int
    truncated: bool


class PipelineResult(BaseModel):
    """The audit-trail object (plan.md Phase 13 build item 5): every field
    a later reconstruction needs to answer "what evidence produced this
    answer", without re-running anything. Mirrors what each real span's
    `output` carries -- this object and the trace are meant to agree."""

    query: Query
    cache_status: CacheStatus
    escalate: bool
    text: str
    route: Route | None = None
    citations: list[str] = []
    provider: str | None = None
    prompt_version: str | None = None
    retrieved: list[Retrieved] = []
    confidence: ConfidenceResult | None = None
    escalation_reason: str | None = None
    escalation_payload: EscalationPayload | None = None
    # Real Langfuse trace id for this request -- also what Phase 14's own
    # planned `GET /trace/{id}` endpoint (plan.md) will key on, so it's a
    # first-class field of the result, not just internal OTel plumbing.
    trace_id: str | None = None
    # Wall-clock ms per stage this request actually ran (a cache hit has only
    # `cache_lookup`, an escalation has no `generation`). Measured here, not
    # read back from Langfuse, so latency attribution (Phase 13 Unit 3) has no
    # dependency on trace-ingestion lag.
    stage_latencies_ms: dict[str, float] = {}
    # What the LLM was actually shown (Phase 13 audit trail): a subset of
    # `retrieved`, in the numbering the answer's [n] citations refer to.
    context: list[ContextEntry] = []
    context_dropped_count: int = 0
    # Set only when a follow-up was resolved against history into a different
    # standalone query; `query` then holds that standalone text (what the cache,
    # router and retriever saw). The user's original is in the trace's root input.
    rewritten_query: str | None = None


# Phase 14's real human-handoff mechanism doesn't exist yet (progress.md,
# Phase 12 Unit 6) -- this is a placeholder response, not a routing/comms
# integration. Its only job today is to be the text a trace/audit-trail
# reconstruction shows for an escalated query.
EventSink = Callable[[PipelineEvent], None]
_log = get_logger("pipeline")
# What a client is told on failure. Provider exception text can carry account/org
# identifiers (a Groq 429 includes the org id), so details go to the log only.
_ERROR_MESSAGE = "The request could not be completed."
_ESCALATION_MESSAGE = "This request has been escalated to a human agent."
_WARM_UP_QUERY = "how do I declare a query parameter"


@contextmanager
def _timed(timings: dict[str, float], stage: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = (time.perf_counter() - start) * 1000.0


class AnswerPipeline:
    def __init__(
        self,
        exact_cache: _Cache,
        semantic_cache: _Cache,
        router: _Router,
        retriever: _Retriever,
        confidence_scorer: ConfidenceScorer,
        generation: GenerationPipeline,
        tracer: _Tracer,
        # No real corpus_version wiring yet -- IndexManifest.corpus_version()
        # (Phase 12 Unit 4) only has real data once a live collection has
        # actually gone through the delta-indexing path at least once, which
        # hasn't happened (progress.md's disclosed Phase 12 open item #2).
        # Wiring this to an empty manifest's hash would look real while
        # meaning nothing; kept as an explicit, honestly-placeholder string
        # like Units 1-3's caches already accept.
        corpus_version: str,
        retrieval_top_k: int = 10,
        tau: float = settings.confidence.tau,
        rewriter: _Rewriter | None = None,
    ) -> None:
        self.exact_cache = exact_cache
        self.semantic_cache = semantic_cache
        self.router = router
        self.retriever = retriever
        self.confidence_scorer = confidence_scorer
        self.generation = generation
        self.tracer = tracer
        self.corpus_version = corpus_version
        self.retrieval_top_k = retrieval_top_k
        self.tau = tau
        self.rewriter = rewriter

    def warm_up(self) -> dict[str, float]:
        """Pay lazy model-load and first-call costs at startup instead of on
        the first user request. Unit 3 measured the first request's confidence
        stage at 17-27 s (cross-encoder weights + first MPS calls) versus
        ~0.6-1.0 s warm. Runs real retrieval and real confidence scoring,
        never the router/generation LLM calls (no quota spend), never the
        caches (no fake entries), never the tracer (no fake traces)."""
        timings: dict[str, float] = {}
        with _timed(timings, "retrieval"):
            candidates = self.retriever.search(_WARM_UP_QUERY, top_k=self.retrieval_top_k)
        with _timed(timings, "confidence"):
            self.confidence_scorer.score(_WARM_UP_QUERY, candidates)
        return timings

    def answer(
        self, query: Query, on_event: EventSink | None = None, history: History | None = None
    ) -> PipelineResult:
        """Blocking entry point. With `on_event`, typed events are emitted as the
        request progresses (tokens live, citations after validation, `done` last);
        without it generation uses the original non-streaming call path."""
        return self._run(query, on_event, self.tracer.create_trace_id(), history)

    def answer_stream(self, query: Query, history: History | None = None) -> Iterator[PipelineEvent]:
        """Same request, as an iterator of events. The work runs on ONE worker thread
        and is handed over via a queue: the OTel span context managers and the
        trace-id ContextVar must open and close on the same thread, which they would
        not if a web server pulled this generator from a thread pool, one `next()`
        per call. Failures become a terminal `error` event, never an exception."""
        events: queue.Queue = queue.Queue()
        end = object()
        trace_id = self.tracer.create_trace_id()

        def work() -> None:
            try:
                self._run(query, events.put, trace_id, history)
            except Exception as exc:  # noqa: BLE001 -- anything a provider/index SDK raises
                events.put(ErrorEvent(code=type(exc).__name__, message=_ERROR_MESSAGE, trace_id=trace_id))
            finally:
                events.put(end)

        threading.Thread(target=work, name="aegis-answer", daemon=True).start()
        while (item := events.get()) is not end:
            yield item

    def _run(
        self, query: Query, on_event: EventSink | None, trace_id: str, history: History | None = None
    ) -> PipelineResult:
        with bind_trace_id(trace_id), self.tracer.start_as_current_observation(
            name="answer_query",
            as_type="span",
            trace_context={"trace_id": trace_id},
            input={"query_id": query.id, "query": query.text},
        ) as root:
            log_event(_log, logging.INFO, "pipeline.request_start", query_id=query.id)
            timings: dict[str, float] = {}
            try:
                effective = query
                if history and self.rewriter is not None:
                    with _timed(timings, "rewrite"):
                        effective = self._rewrite(query, history)
                result = self._answer(effective, timings, on_event).model_copy(
                    update={
                        "trace_id": trace_id,
                        "stage_latencies_ms": timings,
                        "rewritten_query": effective.text if effective.text != query.text else None,
                    }
                )
            except Exception as exc:
                log_event(
                    _log, logging.ERROR, "pipeline.error",
                    query_id=query.id, error_type=type(exc).__name__, detail=str(exc)[:500],
                )
                raise
            log_event(
                _log, logging.INFO, "pipeline.request_end",
                query_id=query.id, cache_status=result.cache_status.value, escalate=result.escalate,
                route=result.route.value if result.route else None, provider=result.provider,
                stage_latencies_ms={k: round(v, 1) for k, v in timings.items()},
            )
            root.update(output=result.model_dump(mode="json", exclude={"query"}))
        # After the root span has closed, so the id in `done` names a finished trace.
        if on_event is not None:
            on_event(
                DoneEvent(
                    trace_id=trace_id, cache_status=result.cache_status.value,
                    route=result.route.value if result.route else None, escalate=result.escalate,
                    rewritten_query=result.rewritten_query,
                    citations=result.citations, provider=result.provider, prompt_version=result.prompt_version,
                    stage_latencies_ms={k: round(v, 1) for k, v in timings.items()},
                )
            )
        return result

    def _answer(self, query: Query, timings: dict[str, float], emit: EventSink | None) -> PipelineResult:
        with _timed(timings, "cache_lookup"):
            cached, status = self._check_cache(query)
        if cached is not None:
            if emit is not None:
                emit(TokenEvent(text=cached.text))
                for c in cached.cited:
                    emit(CitationEvent(**c.model_dump()))
            return PipelineResult(
                query=query,
                cache_status=status,
                escalate=False,
                text=cached.text,
                citations=cached.citations,
                provider=cached.provider,
            )

        with _timed(timings, "routing"):
            route = self._route(query)
        route_outcome = decide_route(route)
        if route_outcome.escalate:
            with _timed(timings, "escalation"):
                return self._escalate(query, status, emit, route=route, reason=route_outcome.reason)

        with _timed(timings, "retrieval"):
            candidates = self._retrieve(query)
        with _timed(timings, "confidence"):
            confidence = self._score_confidence(query, candidates)
        escalation_outcome = decide_escalation(confidence.confidence, self.tau)
        if escalation_outcome.escalate:
            with _timed(timings, "escalation"):
                payload = build_escalation_payload(query.text, candidates, confidence, escalation_outcome)
                return self._escalate(
                    query,
                    status,
                    emit,
                    route=route,
                    reason=escalation_outcome.reason,
                    candidates=candidates,
                    confidence=confidence,
                    payload=payload,
                )

        return self._generate(query, status, route, candidates, confidence, timings, emit)

    def _rewrite(self, query: Query, history: History) -> Query:
        with self.tracer.start_as_current_observation(
            name="rewrite", as_type="span", input={"query": query.text, "history": history}
        ) as span:
            text = self.rewriter.rewrite(history, query.text)
            span.update(output={"rewritten_query": text, "changed": text != query.text})
            return query.model_copy(update={"text": text})

    def _check_cache(self, query: Query) -> tuple[CachedAnswer | None, CacheStatus]:
        with self.tracer.start_as_current_observation(name="cache_lookup", as_type="span") as span:
            exact = self.exact_cache.get(query.text, self.corpus_version)
            if exact is not None:
                span.update(output={"cache_status": CacheStatus.EXACT_HIT.value})
                return exact, CacheStatus.EXACT_HIT

            semantic = self.semantic_cache.get(query.text, self.corpus_version)
            if semantic is not None:
                span.update(output={"cache_status": CacheStatus.SEMANTIC_HIT.value})
                return semantic, CacheStatus.SEMANTIC_HIT

            span.update(output={"cache_status": CacheStatus.MISS.value})
            return None, CacheStatus.MISS

    def _route(self, query: Query) -> Route:
        with self.tracer.start_as_current_observation(
            name="routing", as_type="guardrail", input={"query": query.text}
        ) as span:
            route = self.router.classify(query.text)
            span.update(output={"route": route.value})
            return route

    def _retrieve(self, query: Query) -> list[tuple[Chunk, float]]:
        with self.tracer.start_as_current_observation(
            name="retrieval", as_type="retriever", input={"top_k": self.retrieval_top_k}
        ) as span:
            candidates = self.retriever.search(query.text, top_k=self.retrieval_top_k)
            span.update(
                output={
                    "chunks_returned": len(candidates),
                    "scores": [{"chunk_id": chunk.id, "score": round(score, 4)} for chunk, score in candidates],
                }
            )
            return candidates

    def _score_confidence(self, query: Query, candidates: list[tuple[Chunk, float]]) -> ConfidenceResult:
        with self.tracer.start_as_current_observation(name="confidence", as_type="evaluator") as span:
            result = self.confidence_scorer.score(query.text, candidates)
            span.update(output=result.model_dump(mode="json"))
            return result

    def _escalate(
        self,
        query: Query,
        status: CacheStatus,
        emit: EventSink | None,
        route: Route,
        reason: str,
        candidates: list[tuple[Chunk, float]] | None = None,
        confidence: ConfidenceResult | None = None,
        payload: EscalationPayload | None = None,
    ) -> PipelineResult:
        log_event(
            _log, logging.WARNING, "pipeline.escalated",
            query_id=query.id, route=route.value, reason=reason, after_retrieval=candidates is not None,
        )
        with self.tracer.start_as_current_observation(
            name="escalation", as_type="guardrail", input={"reason": reason}
        ) as span:
            span.update(output={"escalated": True, "reason": reason})
            if emit is not None:
                emit(
                    EscalationEvent(
                        reason=reason, message=_ESCALATION_MESSAGE, route=route.value,
                        after_retrieval=candidates is not None,
                    )
                )
            return PipelineResult(
                query=query,
                cache_status=status,
                escalate=True,
                text=_ESCALATION_MESSAGE,
                route=route,
                retrieved=[Retrieved(chunk=c, score=s, rank=i) for i, (c, s) in enumerate(candidates or [], start=1)],
                confidence=confidence,
                escalation_reason=reason,
                escalation_payload=payload,
            )

    def _generate(
        self,
        query: Query,
        status: CacheStatus,
        route: Route,
        candidates: list[tuple[Chunk, float]],
        confidence: ConfidenceResult,
        timings: dict[str, float],
        emit: EventSink | None,
    ) -> PipelineResult:
        retrieved = [Retrieved(chunk=c, score=s, rank=i) for i, (c, s) in enumerate(candidates, start=1)]

        with _timed(timings, "generation"):
            with self.tracer.start_as_current_observation(
                name="generation",
                as_type="generation",
                model=self.generation.llm.__class__.__name__,
                input={"prompt_version": self.generation.prompt.version},
            ) as gen_span:
                on_token = (lambda t: emit(TokenEvent(text=t))) if emit is not None else None
                draft = self.generation.generate(query, retrieved, on_token=on_token)
                context = [
                    ContextEntry(
                        citation_id=pc.citation_id, chunk_id=pc.chunk.id, content_hash=pc.chunk.content_hash, score=pc.score,
                        token_count=pc.token_count, truncated=pc.truncated,
                    )
                    for pc in draft.context.chunks
                ]
                gen_span.update(
                    input={
                        "prompt_version": self.generation.prompt.version,
                        "temperature": self.generation.temperature,
                        "top_p": self.generation.top_p,
                        "max_tokens": self.generation.max_tokens,
                        "budget_tokens": self.generation.budget_tokens,
                        "context": [e.model_dump() for e in context],
                        "context_total_tokens": draft.context.total_tokens,
                        "context_dropped_count": draft.context.dropped_count,
                    },
                    output={"text": draft.text, "provider": draft.provider},
                )

        with _timed(timings, "validation"):
            with self.tracer.start_as_current_observation(name="validation", as_type="guardrail") as val_span:
                gen_result = self.generation.validate(draft)
                val_span.update(
                    output={"citations": gen_result.citations, "claim_count": len(gen_result.validation.claims)}
                )

        validated = set(gen_result.citations)
        cited = [
            CitedChunk(
                citation_id=pc.citation_id, chunk_id=pc.chunk.id, doc_id=pc.chunk.doc_id,
                heading_path=pc.chunk.heading_path, text=pc.chunk.text,
            )
            for pc in sorted(draft.context.chunks, key=lambda pc: pc.citation_id)
            if pc.chunk.id in validated
        ]
        if emit is not None:
            for c in cited:
                emit(CitationEvent(**c.model_dump()))

        answer = CachedAnswer(
            text=gen_result.text, citations=gen_result.citations, provider=gen_result.provider, cited=cited
        )
        self.exact_cache.set(query.text, self.corpus_version, answer)
        self.semantic_cache.set(query.text, self.corpus_version, answer)

        return PipelineResult(
            query=query,
            cache_status=status,
            escalate=False,
            text=gen_result.text,
            route=route,
            citations=gen_result.citations,
            provider=gen_result.provider,
            prompt_version=self.generation.prompt.version,
            retrieved=retrieved,
            confidence=confidence,
            context=context,
            context_dropped_count=draft.context.dropped_count,
        )
