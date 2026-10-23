"""TDD for the end-to-end answer pipeline (Phase 13, Unit 2): the first
real composition of cache -> router -> retrieval -> confidence ->
generation. The two tests that matter most, same discipline as
`test_router_policy.py`/`test_confidence.py`: proving structurally, with a
spy that raises if invoked, that (1) a cache hit never reaches routing or
generation, and (2) an escalation (hard route or low confidence) never
reaches generation.
"""

from contextlib import contextmanager

from aegis.cache.exact_match import CachedAnswer
from aegis.chunking.tokens import count_tokens
from aegis.generation.pipeline import GenerationPipeline
from aegis.generation.prompts.v1 import TEMPLATE as V1
from aegis.guardrails.citations import CitationValidator
from aegis.guardrails.confidence import ConfidenceScorer, SignalBounds
from aegis.pipeline import AnswerPipeline, CacheStatus
from aegis.router.classifier import Route
from aegis.types import Chunk, DocType, Query


def _chunk(id: str, text: str = "") -> Chunk:
    text = text or f"some real chunk text for {id}"
    return Chunk(
        id=id,
        doc_id=id,
        text=text,
        doc_type=DocType.DOCUMENTATION,
        content_hash="hash",
        token_count=count_tokens(text),
        embedding_text=text,
    )


class NullTracer:
    """A tracer that records every span opened but sends nothing anywhere --
    the same role a fake LLM/reranker plays elsewhere in this codebase: lets
    the pipeline's real branching be tested without touching the network."""

    def __init__(self) -> None:
        self.spans: list[dict] = []
        self._next_trace_id = 0

    def create_trace_id(self) -> str:
        self._next_trace_id += 1
        return f"null-trace-{self._next_trace_id}"

    @contextmanager
    def start_as_current_observation(self, *, name: str, as_type: str = "span", **kwargs):
        record = {"name": name, "as_type": as_type, "input": kwargs.get("input"), "output": None}
        self.spans.append(record)

        class _Span:
            def update(self, **fields):
                record.update(fields)

        yield _Span()


class _FakeCache:
    def __init__(self, hit: CachedAnswer | None = None):
        self.hit = hit
        self.get_calls: list[tuple[str, str]] = []
        self.set_calls: list[tuple[str, str, CachedAnswer]] = []

    def get(self, query_text: str, corpus_version: str, *, partition: str = "full") -> CachedAnswer | None:
        self.get_calls.append((query_text, corpus_version))
        return self.hit

    def set(self, query_text: str, corpus_version: str, answer: CachedAnswer, *, partition: str = "full") -> None:
        self.set_calls.append((query_text, corpus_version, answer))


class _RaisingRouter:
    def classify(self, query: str) -> Route:
        raise AssertionError("router must not be reached on a cache hit")


class _RaisingRetriever:
    def search(self, query: str, top_k: int, **kwargs):
        raise AssertionError("retrieval must not be reached")


class _FixedRouter:
    def __init__(self, route: Route):
        self.route = route

    def classify(self, query: str) -> Route:
        return self.route


class _FixedRetriever:
    def __init__(self, results: list[tuple[Chunk, float]]):
        self.results = results

    def search(self, query: str, top_k: int, **kwargs):
        return self.results


class _FakeReranker:
    """Same fixed-lookup-table fake as tests/test_confidence.py's own
    _FakeReranker -- lets a test dial confidence to an exact known value via
    reranker_top_score, regardless of what the retriever returned."""

    def __init__(self, scores_by_id: dict[str, float]):
        self.scores_by_id = scores_by_id

    def rerank(self, query, candidates: list[tuple[Chunk, float]], top_k: int):
        scored = sorted(((c, self.scores_by_id[c.id]) for c, _ in candidates), key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


def _fake_encode_fn(dim: int = 2):
    def encode(texts: list[str]):
        import numpy as np

        return np.ones((len(texts), dim), dtype=np.float64)

    return encode


_IDENTITY_BOUNDS = {name: SignalBounds(0.0, 1.0) for name in ("reranker_top_score", "score_gap", "semantic_agreement", "distribution_shape")}
_RERANKER_ONLY_WEIGHTS = {"reranker_top_score": 1.0, "score_gap": 0.0, "semantic_agreement": 0.0, "distribution_shape": 0.0}


def _confidence_scorer(confidence_value: float) -> ConfidenceScorer:
    # reranker_top_score is the only weighted signal (IDENTITY_BOUNDS is the
    # identity map), so the reranker's own score for the top candidate *is*
    # the resulting confidence -- deterministic and exact, same convention
    # as test_confidence.py's handle() tests.
    return ConfidenceScorer(
        reranker=_FakeReranker({"c1": confidence_value, "c2": confidence_value}),
        encode_fn=_fake_encode_fn(),
        weights=_RERANKER_ONLY_WEIGHTS,
        bounds=_IDENTITY_BOUNDS,
        rerank_pool_size=3,
    )


class _RaisingLLM:
    def generate_with_provider(self, messages, temperature, top_p, max_tokens):
        raise AssertionError("generation must not be reached on escalation")


class _FakeLLM:
    def __init__(self, text: str, provider: str = "fake-provider"):
        self.text = text
        self.provider = provider

    def generate_with_provider(self, messages, temperature, top_p, max_tokens):
        return self.text, self.provider


def _generation_pipeline(llm) -> GenerationPipeline:
    validator = CitationValidator(entailment_fn=lambda claim, chunk_text: True, lexical_overlap_threshold=0.4)
    return GenerationPipeline(llm=llm, validator=validator, prompt=V1)


def _pipeline(
    router=None,
    retriever=None,
    confidence: float = 1.0,
    llm=None,
    exact_cache=None,
    semantic_cache=None,
    tau: float = 0.5,
    rewriter=None,
) -> tuple[AnswerPipeline, NullTracer]:
    tracer = NullTracer()
    pipeline = AnswerPipeline(
        exact_cache=exact_cache or _FakeCache(hit=None),
        semantic_cache=semantic_cache or _FakeCache(hit=None),
        router=router or _FixedRouter(Route.DOCS_HOWTO),
        retriever=retriever or _FixedRetriever([( _chunk("c1"), 0.9), (_chunk("c2"), 0.5)]),
        confidence_scorer=_confidence_scorer(confidence),
        generation=_generation_pipeline(llm or _RaisingLLM()),
        tracer=tracer,
        corpus_version="test-corpus-v1",
        tau=tau,
        rewriter=rewriter,
    )
    return pipeline, tracer


def test_exact_cache_hit_never_reaches_router_or_generation():
    cached = CachedAnswer(text="a cached answer", citations=["c1"], provider="fake-provider")
    exact_cache = _FakeCache(hit=cached)
    pipeline, tracer = _pipeline(router=_RaisingRouter(), retriever=_RaisingRetriever(), exact_cache=exact_cache)

    result = pipeline.answer(Query(id="q1", text="how do I use dependency injection"))

    assert result.cache_status == CacheStatus.EXACT_HIT
    assert result.text == "a cached answer"
    assert result.escalate is False


def test_semantic_cache_hit_used_only_after_exact_miss():
    cached = CachedAnswer(text="a paraphrase-matched answer", citations=[], provider="fake-provider")
    exact_cache = _FakeCache(hit=None)
    semantic_cache = _FakeCache(hit=cached)
    pipeline, tracer = _pipeline(
        router=_RaisingRouter(), retriever=_RaisingRetriever(), exact_cache=exact_cache, semantic_cache=semantic_cache
    )

    result = pipeline.answer(Query(id="q2", text="how do i inject a dependency"))

    assert result.cache_status == CacheStatus.SEMANTIC_HIT
    assert result.text == "a paraphrase-matched answer"
    assert len(exact_cache.get_calls) == 1


def test_billing_route_escalates_before_retrieval_or_generation():
    pipeline, tracer = _pipeline(router=_FixedRouter(Route.BILLING_ACCOUNT_LEGAL), retriever=_RaisingRetriever())

    result = pipeline.answer(Query(id="q3", text="please refund my invoice"))

    assert result.escalate is True
    assert result.route == Route.BILLING_ACCOUNT_LEGAL
    assert result.retrieved == []
    assert result.confidence is None


def test_low_confidence_escalates_after_retrieval_but_never_reaches_generation():
    pipeline, tracer = _pipeline(confidence=0.0, tau=0.9, llm=_RaisingLLM())

    result = pipeline.answer(Query(id="q4", text="an ambiguous multi-hop question"))

    assert result.escalate is True
    assert result.confidence is not None
    assert result.escalation_payload is not None


def test_high_confidence_generates_validates_and_writes_through_both_caches():
    exact_cache = _FakeCache(hit=None)
    semantic_cache = _FakeCache(hit=None)
    pipeline, tracer = _pipeline(
        confidence=1.0, tau=0.1, llm=_FakeLLM("a real generated answer [c1]"), exact_cache=exact_cache, semantic_cache=semantic_cache
    )

    result = pipeline.answer(Query(id="q5", text="how do I use dependency injection"))

    assert result.escalate is False
    assert result.cache_status == CacheStatus.MISS
    assert result.text == "a real generated answer [c1]"
    assert result.provider == "fake-provider"
    assert result.prompt_version == "v1"
    assert len(exact_cache.set_calls) == 1
    assert len(semantic_cache.set_calls) == 1


def test_escalated_answers_are_never_written_to_cache():
    exact_cache = _FakeCache(hit=None)
    semantic_cache = _FakeCache(hit=None)
    pipeline, tracer = _pipeline(
        router=_FixedRouter(Route.BILLING_ACCOUNT_LEGAL),
        retriever=_RaisingRetriever(),
        exact_cache=exact_cache,
        semantic_cache=semantic_cache,
    )

    pipeline.answer(Query(id="q6", text="cancel my enterprise contract"))

    assert exact_cache.set_calls == []
    assert semantic_cache.set_calls == []


def test_every_stage_opens_a_real_span_on_the_full_generate_path():
    pipeline, tracer = _pipeline(confidence=1.0, tau=0.1, llm=_FakeLLM("answer [c1]"))

    pipeline.answer(Query(id="q7", text="how do I use dependency injection"))

    names = [s["name"] for s in tracer.spans]
    assert names == ["answer_query", "cache_lookup", "routing", "retrieval", "confidence", "generation", "validation"]


def test_result_carries_a_real_trace_id():
    pipeline, tracer = _pipeline(confidence=1.0, tau=0.1, llm=_FakeLLM("answer [c1]"))

    result = pipeline.answer(Query(id="q9", text="how do I use dependency injection"))

    assert result.trace_id == "null-trace-1"


def test_hard_route_escalation_span_sequence():
    pipeline, tracer = _pipeline(router=_FixedRouter(Route.BILLING_ACCOUNT_LEGAL), retriever=_RaisingRetriever())

    pipeline.answer(Query(id="q8", text="please refund my invoice"))

    names = [s["name"] for s in tracer.spans]
    assert names == ["answer_query", "cache_lookup", "routing", "escalation"]


def test_result_carries_per_stage_latencies_on_the_full_generate_path():
    pipeline, tracer = _pipeline(confidence=1.0, tau=0.1, llm=_FakeLLM("answer [c1]"))

    result = pipeline.answer(Query(id="q10", text="how do I use dependency injection"))

    assert set(result.stage_latencies_ms) == {
        "cache_lookup", "routing", "retrieval", "confidence", "generation", "validation",
    }
    assert all(ms >= 0.0 for ms in result.stage_latencies_ms.values())


def test_cache_hit_reports_only_cache_lookup_latency():
    cached = CachedAnswer(text="cached", citations=[], provider="fake-provider")
    pipeline, tracer = _pipeline(router=_RaisingRouter(), retriever=_RaisingRetriever(), exact_cache=_FakeCache(hit=cached))

    result = pipeline.answer(Query(id="q11", text="anything"))

    assert set(result.stage_latencies_ms) == {"cache_lookup"}


def test_hard_route_escalation_reports_only_the_stages_it_ran():
    pipeline, tracer = _pipeline(router=_FixedRouter(Route.BILLING_ACCOUNT_LEGAL), retriever=_RaisingRetriever())

    result = pipeline.answer(Query(id="q12", text="refund me"))

    assert set(result.stage_latencies_ms) == {"cache_lookup", "routing", "escalation"}


def test_warm_up_exercises_retrieval_and_confidence_but_never_router_llm_or_caches():
    exact_cache = _FakeCache(hit=None)
    semantic_cache = _FakeCache(hit=None)
    pipeline, tracer = _pipeline(
        router=_RaisingRouter(), llm=_RaisingLLM(), exact_cache=exact_cache, semantic_cache=semantic_cache
    )

    timings = pipeline.warm_up()

    assert set(timings) == {"retrieval", "confidence"}
    assert all(ms >= 0.0 for ms in timings.values())
    assert exact_cache.get_calls == [] and exact_cache.set_calls == []
    assert semantic_cache.get_calls == [] and semantic_cache.set_calls == []
    assert tracer.spans == []


def _log_capture():
    import io
    import json
    import logging

    from aegis.observability.logging import JsonFormatter

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("aegis.pipeline")
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return lambda: [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_every_log_line_of_a_request_carries_the_results_trace_id():
    records = _log_capture()
    pipeline, tracer = _pipeline(confidence=1.0, tau=0.1, llm=_FakeLLM("answer [c1]"))

    result = pipeline.answer(Query(id="q13", text="how do I use dependency injection"))

    lines = records()
    assert len(lines) >= 2
    assert all(r["trace_id"] == result.trace_id for r in lines)


def test_escalation_is_logged_with_its_reason_and_path():
    records = _log_capture()
    pipeline, tracer = _pipeline(router=_FixedRouter(Route.BILLING_ACCOUNT_LEGAL), retriever=_RaisingRetriever())

    pipeline.answer(Query(id="q14", text="refund me"))

    (esc,) = [r for r in records() if r["event"] == "pipeline.escalated"]
    assert esc["level"] == "WARNING" and esc["route"] == "billing_account_legal" and "billing" in esc["reason"]


def test_request_end_logs_cache_status_and_stage_latencies():
    records = _log_capture()
    cached = CachedAnswer(text="cached", citations=[], provider="fake-provider")
    pipeline, tracer = _pipeline(router=_RaisingRouter(), retriever=_RaisingRetriever(), exact_cache=_FakeCache(hit=cached))

    pipeline.answer(Query(id="q15", text="anything"))

    (end,) = [r for r in records() if r["event"] == "pipeline.request_end"]
    assert end["cache_status"] == "exact_hit" and end["escalate"] is False
    assert set(end["stage_latencies_ms"]) == {"cache_lookup"}


def test_result_records_the_context_actually_shown_to_the_llm_with_citation_numbers():
    pipeline, tracer = _pipeline(confidence=1.0, tau=0.1, llm=_FakeLLM("answer [1]"))

    result = pipeline.answer(Query(id="q16", text="how do I use dependency injection"))

    assert [(e.citation_id, e.chunk_id) for e in result.context] == [(1, "c1"), (2, "c2")]
    assert all(e.token_count > 0 for e in result.context)
    assert result.context_dropped_count == 0


def test_generation_span_input_carries_the_prompt_version_params_and_context_map():
    pipeline, tracer = _pipeline(confidence=1.0, tau=0.1, llm=_FakeLLM("answer [1]"))

    pipeline.answer(Query(id="q17", text="how do I use dependency injection"))

    (gen,) = [s for s in tracer.spans if s["name"] == "generation"]
    assert gen["input"]["prompt_version"] == "v1"
    assert {"temperature", "top_p", "max_tokens", "budget_tokens"} <= set(gen["input"])
    assert [(c["citation_id"], c["chunk_id"]) for c in gen["input"]["context"]] == [(1, "c1"), (2, "c2")]


def test_escalated_results_have_no_context_because_nothing_was_shown_to_an_llm():
    pipeline, tracer = _pipeline(router=_FixedRouter(Route.BILLING_ACCOUNT_LEGAL), retriever=_RaisingRetriever())

    result = pipeline.answer(Query(id="q18", text="refund me"))

    assert result.context == [] and result.context_dropped_count == 0


def test_warm_up_also_loads_the_query_encoder_through_the_retriever_last():
    # Through the retriever, not the semantic cache: the semantic tier is off by
    # default (design record 14h) but dense retrieval encodes every miss.
    order: list[str] = []

    class _WarmableRetriever(_FixedRetriever):
        def search(self, query, top_k, **kwargs):
            order.append("retrieval")
            return super().search(query, top_k, **kwargs)

        def warm_up(self, text: str) -> None:
            order.append("encoder")

    retriever = _WarmableRetriever([(_chunk("c1"), 0.9), (_chunk("c2"), 0.5)])
    pipeline, tracer = _pipeline(router=_RaisingRouter(), llm=_RaisingLLM(), retriever=retriever)

    timings = pipeline.warm_up()

    assert order == ["retrieval", "encoder"]  # encoder last: absorbs the post-MPS first-CPU-pass cost
    assert list(timings) == ["retrieval", "confidence", "encoder"]


def test_semantic_tier_off_means_exact_cache_only():
    exact_cache = _FakeCache(hit=None)
    pipeline, _ = _pipeline(confidence=1.0, tau=0.1, llm=_FakeLLM("Use Depends [1]."), exact_cache=exact_cache)
    pipeline.semantic_cache = None

    result = pipeline.answer(Query(id="q-off", text="how do I make a query parameter required in FastAPI?"))

    assert result.cache_status == CacheStatus.MISS and result.escalate is False
    assert len(exact_cache.get_calls) == 1 and len(exact_cache.set_calls) == 1


def test_semantic_tier_is_off_by_default():
    from aegis.config import CacheSettings

    assert CacheSettings().semantic_cache_enabled is False


class _Provider:
    def __init__(self, name: str, model: str):
        self.name, self.model = name, model


class _FakeChainLLM(_FakeLLM):
    """A fake with a provider chain, like FallbackLLMClient."""

    def __init__(self, text: str, models: list[tuple[str, str]]):
        super().__init__(text)
        self.providers = [_Provider(n, m) for n, m in models]


def _cached_pipeline(tmp_path, llm, prompt=V1) -> AnswerPipeline:
    from aegis.cache.exact_match import ExactMatchCache

    pipeline, _ = _pipeline(confidence=1.0, tau=0.1, llm=llm, exact_cache=ExactMatchCache(tmp_path / "exact.sqlite"))
    pipeline.semantic_cache = None
    pipeline.generation.prompt = prompt
    return pipeline


def test_answer_cached_under_one_prompt_version_is_not_served_under_another(tmp_path):
    """Open Question #19 (design record 15j): the cache scope must carry the prompt
    version, or a prompt switch (v2 -> v3, a security fix) keeps replaying old
    answers without generation or validation."""
    from aegis.generation.prompts import get_prompt

    q = Query(id="q", text="how do I make a query parameter required in FastAPI?")
    first = _cached_pipeline(tmp_path, _FakeLLM("old prompt answer [1]."), prompt=get_prompt("v2")).answer(q)
    second = _cached_pipeline(tmp_path, _FakeLLM("new prompt answer [1]."), prompt=get_prompt("v3")).answer(q)

    assert first.cache_status == CacheStatus.MISS
    assert second.cache_status == CacheStatus.MISS and second.text == "new prompt answer [1]."


def test_answer_cached_under_one_generator_is_not_served_under_another(tmp_path):
    q = Query(id="q", text="how do I make a query parameter required in FastAPI?")
    _cached_pipeline(tmp_path, _FakeChainLLM("qwen answer [1].", [("groq", "qwen/qwen3.8-27b")])).answer(q)
    second = _cached_pipeline(tmp_path, _FakeChainLLM("gpt answer [1].", [("openai", "gpt-5-mini")])).answer(q)

    assert second.cache_status == CacheStatus.MISS and second.text == "gpt answer [1]."


def test_same_answer_configuration_still_hits(tmp_path):
    q = Query(id="q", text="how do I make a query parameter required in FastAPI?")
    llm = _FakeChainLLM("answer [1].", [("openai", "gpt-5-mini")])
    _cached_pipeline(tmp_path, llm).answer(q)

    assert _cached_pipeline(tmp_path, llm).answer(q).cache_status == CacheStatus.EXACT_HIT
