"""The request flow as a LangGraph `StateGraph` (plan.md Phase 17, design record 17b).

Only the orchestration moves. `LangGraphPipeline` subclasses `AnswerPipeline` and
replaces one seam, `_orchestrate`; every node below calls the same stage method the
direct path calls (`_check_cache`, `_route`, `_retrieve`, `_score_confidence`,
`_escalate`, `_generate`). So spans, timings, streaming events, cache scope and the
audit trail are shared code, and the port can differ only in *which stage runs next*,
which is exactly what the differential tests pin down.

Per-run objects (the timings dict, the event sink) travel in `config["configurable"]`,
not in graph state: state is what a checkpointer would persist, and a callback is not
state.

The retry edge is the one thing the direct path lacks. It is off by default, so the
default graph is behavior-identical to direct; see `OrchestrationSettings`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Protocol, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from aegis.cache.exact_match import CachedAnswer
from aegis.config import settings
from aegis.guardrails.confidence import (
    ConfidenceResult,
    EscalationOutcome,
    build_escalation_payload,
    decide_escalation,
)
from aegis.pipeline import (
    _PROVIDER_UNAVAILABLE_REASON,
    AnswerPipeline,
    CacheStatus,
    EventSink,
    PipelineResult,
    _timed,
)
from aegis.query.rewrite import History
from aegis.router.classifier import Route, RouterUnavailableError
from aegis.router.policy import decide as decide_route
from aegis.types import Chunk, Query


class _Reformulator(Protocol):
    def reformulate(self, query: str) -> str: ...


class State(TypedDict, total=False):
    query: Query  # as the user sent it
    history: History | None
    effective: Query  # after follow-up rewriting; what the cache, router and generator see
    retrieval_query: Query  # differs from `effective` only after a retry
    cached: CachedAnswer | None
    cache_status: CacheStatus
    route: Route | None
    candidates: list[tuple[Chunk, float]]
    confidence: ConfidenceResult
    verdict: EscalationOutcome
    retries: int
    result: PipelineResult


@contextmanager
def _accumulating(timings: dict[str, float], stage: str):
    """Like `_timed`, but a stage that runs twice (the retry) reports its total
    rather than only its last pass."""
    before = timings.get(stage, 0.0)
    with _timed(timings, stage):
        yield
    timings[stage] += before


def _timings(config: RunnableConfig) -> dict[str, float]:
    return config["configurable"]["timings"]


def _emit(config: RunnableConfig) -> EventSink | None:
    return config["configurable"]["emit"]


class LangGraphPipeline(AnswerPipeline):
    def __init__(
        self,
        *args: Any,
        retry_on_low_confidence: bool = settings.orchestration.retry_on_low_confidence,
        max_retries: int = settings.orchestration.max_retries,
        reformulator: _Reformulator | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.retry_on_low_confidence = retry_on_low_confidence
        self.max_retries = max_retries
        self.reformulator = reformulator
        self.graph = self._build_graph()

    # -- seam ---------------------------------------------------------------------

    def _orchestrate(
        self, query: Query, history: History | None, timings: dict[str, float], emit: EventSink | None
    ) -> tuple[PipelineResult, Query]:
        final = self.graph.invoke(
            {"query": query, "history": history, "retries": 0},
            config={"configurable": {"timings": timings, "emit": emit}},
        )
        return final["result"], final["effective"]

    # -- graph --------------------------------------------------------------------

    def _build_graph(self) -> Any:
        g = StateGraph(State)
        g.add_node("rewrite", self._n_rewrite)
        g.add_node("cache", self._n_cache)
        g.add_node("serve_cached", self._n_serve_cached)
        g.add_node("route", self._n_route)
        g.add_node("escalate_route", self._n_escalate_route)
        g.add_node("retrieve", self._n_retrieve)
        g.add_node("confidence", self._n_confidence)
        g.add_node("reformulate", self._n_reformulate)
        g.add_node("escalate_confidence", self._n_escalate_confidence)
        g.add_node("generate", self._n_generate)

        g.add_conditional_edges(START, self._after_start, {"rewrite": "rewrite", "cache": "cache"})
        g.add_edge("rewrite", "cache")
        g.add_conditional_edges("cache", self._after_cache, {"serve_cached": "serve_cached", "route": "route"})
        g.add_conditional_edges(
            "route", self._after_route, {"escalate_route": "escalate_route", "retrieve": "retrieve", "done": END}
        )
        g.add_edge("retrieve", "confidence")
        g.add_conditional_edges(
            "confidence",
            self._after_confidence,
            {"reformulate": "reformulate", "escalate_confidence": "escalate_confidence", "generate": "generate"},
        )
        g.add_edge("reformulate", "retrieve")
        for terminal in ("serve_cached", "escalate_route", "escalate_confidence", "generate"):
            g.add_edge(terminal, END)
        return g.compile()

    # -- routing functions (the conditional edges) --------------------------------

    def _after_start(self, state: State) -> str:
        return "rewrite" if state["history"] and self.rewriter is not None else "cache"

    @staticmethod
    def _after_cache(state: State) -> str:
        return "serve_cached" if state["cached"] is not None else "route"

    @staticmethod
    def _after_route(state: State) -> str:
        if state["route"] is None:  # router unavailable: `_n_route` already built the escalation
            return "done"
        return "escalate_route" if decide_route(state["route"]).escalate else "retrieve"

    def _after_confidence(self, state: State) -> str:
        if not state["verdict"].escalate:
            return "generate"
        if self.retry_on_low_confidence and state["retries"] < self.max_retries:
            return "reformulate"
        return "escalate_confidence"

    # -- nodes --------------------------------------------------------------------

    def _n_rewrite(self, state: State, config: RunnableConfig) -> State:
        with _timed(_timings(config), "rewrite"):
            effective = self._rewrite(state["query"], state["history"] or [])
        return {"effective": effective}

    def _n_cache(self, state: State, config: RunnableConfig) -> State:
        effective = state.get("effective", state["query"])
        with _timed(_timings(config), "cache_lookup"):
            cached, status = self._check_cache(effective)
        return {"effective": effective, "retrieval_query": effective, "cached": cached, "cache_status": status}

    def _n_serve_cached(self, state: State, config: RunnableConfig) -> State:
        cached = state["cached"]
        assert cached is not None
        return {"result": self._serve_cached(state["effective"], cached, state["cache_status"], _emit(config))}

    def _n_route(self, state: State, config: RunnableConfig) -> State:
        timings = _timings(config)
        effective = state["effective"]
        try:
            with _timed(timings, "routing"):
                return {"route": self._route(effective)}
        except RouterUnavailableError:
            with _timed(timings, "escalation"):
                result = self._escalate(
                    effective, state["cache_status"], _emit(config), route=None, cause="provider_unavailable",
                    reason=_PROVIDER_UNAVAILABLE_REASON.format(stage="routing"),
                )
            return {"route": None, "result": result}

    def _n_escalate_route(self, state: State, config: RunnableConfig) -> State:
        assert state["route"] is not None
        outcome = decide_route(state["route"])
        with _timed(_timings(config), "escalation"):
            result = self._escalate(
                state["effective"], state["cache_status"], _emit(config),
                route=state["route"], cause="policy", reason=outcome.reason,
            )
        return {"result": result}

    def _n_retrieve(self, state: State, config: RunnableConfig) -> State:
        with _accumulating(_timings(config), "retrieval"):
            candidates = self._retrieve(state["retrieval_query"])
        return {"candidates": candidates}

    def _n_confidence(self, state: State, config: RunnableConfig) -> State:
        with _accumulating(_timings(config), "confidence"):
            confidence = self._score_confidence(state["retrieval_query"], state["candidates"])
        return {"confidence": confidence, "verdict": decide_escalation(confidence.confidence, self.tau)}

    def _n_reformulate(self, state: State, config: RunnableConfig) -> State:
        if self.reformulator is None:
            from aegis.query.iterative import QueryReformulator

            self.reformulator = QueryReformulator()
        current = state["retrieval_query"]
        with _accumulating(_timings(config), "reformulation"):
            text = self.reformulator.reformulate(current.text)
        return {"retrieval_query": current.model_copy(update={"text": text}), "retries": state["retries"] + 1}

    def _n_escalate_confidence(self, state: State, config: RunnableConfig) -> State:
        effective, candidates, confidence = state["effective"], state["candidates"], state["confidence"]
        with _timed(_timings(config), "escalation"):
            payload = build_escalation_payload(effective.text, candidates, confidence, state["verdict"])
            result = self._escalate(
                effective, state["cache_status"], _emit(config), route=state["route"], cause="low_confidence",
                reason=state["verdict"].reason, candidates=candidates, confidence=confidence, payload=payload,
            )
        return {"result": result}

    def _n_generate(self, state: State, config: RunnableConfig) -> State:
        route = state["route"]
        assert route is not None
        result = self._generate(
            state["effective"], state["cache_status"], route, state["candidates"], state["confidence"],
            _timings(config), _emit(config),
        )
        return {"result": result}
