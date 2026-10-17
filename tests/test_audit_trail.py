"""TDD for audit-trail reconstruction (Phase 13 build item 5): given only the
observations of one trace, rebuild what evidence produced an answer.

Observations here come from a REAL `AnswerPipeline` run recorded by the test
tracer (reusing test_pipeline.py's fakes), not hand-written dicts -- the risk
this guards against is format drift: change a span's shape in pipeline.py and
reconstruction must fail in a test, not silently in production."""

import copy

from test_pipeline import (
    CachedAnswer, Query, Route, _chunk, _FakeCache, _FakeLLM, _FixedRouter, _pipeline, _RaisingRetriever,
    _RaisingRouter,
)

from aegis.observability.audit import build_audit_trail, render_markdown, resolve_evidence


def _observations(tracer) -> list[dict]:
    return [{"name": s["name"], "type": s["as_type"], "input": s["input"], "output": s["output"]} for s in tracer.spans]


def _generated_trace():
    pipeline, tracer = _pipeline(confidence=1.0, tau=0.1, llm=_FakeLLM("use Depends [1] and [2]"))
    result = pipeline.answer(Query(id="q1", text="how do I use dependency injection"))
    return _observations(tracer), result


def test_generated_answer_is_fully_reconstructed_from_spans():
    obs, result = _generated_trace()

    trail = build_audit_trail(obs)

    assert trail.query_id == "q1" and trail.query_text == "how do I use dependency injection"
    assert trail.cache_status == "miss" and trail.route == "docs_howto" and trail.escalated is False
    assert [(r.chunk_id, r.score) for r in trail.retrieved] == [("c1", 0.9), ("c2", 0.5)]
    assert trail.confidence is not None and trail.confidence["confidence"] == 1.0
    assert trail.prompt_version == "v1"
    assert [(c["citation_id"], c["chunk_id"]) for c in trail.context] == [(1, "c1"), (2, "c2")]
    assert trail.final_text == "use Depends [1] and [2]" and trail.provider == "fake-provider"
    assert trail.citations == result.citations
    assert trail.warnings == [] and trail.inconsistencies == []


def test_hard_route_escalation_trail_has_reason_and_no_evidence():
    pipeline, tracer = _pipeline(router=_FixedRouter(Route.BILLING_ACCOUNT_LEGAL), retriever=_RaisingRetriever())
    pipeline.answer(Query(id="q2", text="refund me"))

    trail = build_audit_trail(_observations(tracer))

    assert trail.escalated is True and trail.route == "billing_account_legal"
    assert "billing" in trail.escalation_reason
    assert trail.retrieved == [] and trail.context == [] and trail.confidence is None
    assert trail.warnings == [] and trail.inconsistencies == []


def test_low_confidence_escalation_trail_keeps_retrieval_and_confidence():
    pipeline, tracer = _pipeline(confidence=0.0, tau=0.9)
    pipeline.answer(Query(id="q3", text="ambiguous"))

    trail = build_audit_trail(_observations(tracer))

    assert trail.escalated is True and "tau" in trail.escalation_reason
    assert len(trail.retrieved) == 2 and trail.confidence is not None and trail.context == []
    assert trail.warnings == [] and trail.inconsistencies == []


def test_cache_hit_trail_says_so_and_expects_no_later_stages():
    cached = CachedAnswer(text="a cached answer", citations=["c1"], provider="fake-provider")
    pipeline, tracer = _pipeline(router=_RaisingRouter(), retriever=_RaisingRetriever(), exact_cache=_FakeCache(hit=cached))
    pipeline.answer(Query(id="q4", text="anything"))

    trail = build_audit_trail(_observations(tracer))

    assert trail.cache_status == "exact_hit" and trail.final_text == "a cached answer"
    assert trail.route is None and trail.warnings == []


def test_a_missing_stage_is_reported_not_silently_ignored():
    obs, _ = _generated_trace()
    obs = [o for o in obs if o["name"] != "generation"]

    trail = build_audit_trail(obs)

    assert any("generation" in w and "not found" in w for w in trail.warnings)


def test_a_trace_without_the_context_map_says_it_cannot_resolve_citation_numbers():
    obs, _ = _generated_trace()
    for o in obs:
        if o["name"] == "generation":
            o["input"] = {"prompt_version": "v1"}  # the pre-audit trace format

    trail = build_audit_trail(obs)

    assert any("context" in w and "[n]" in w for w in trail.warnings)


def test_tampered_root_output_is_flagged_as_an_inconsistency():
    obs, _ = _generated_trace()
    root = next(o for o in obs if o["name"] == "answer_query")
    root["output"] = copy.deepcopy(root["output"])
    root["output"]["text"] = "a different answer than the generation span recorded"

    trail = build_audit_trail(obs)

    assert any("text" in i for i in trail.inconsistencies)


def test_evidence_resolution_attaches_text_marks_cited_and_detects_drift():
    obs, _ = _generated_trace()
    trail = build_audit_trail(obs)
    unchanged = _chunk("c1")
    drifted = _chunk("c2").model_copy(update={"content_hash": "a-different-hash"})

    evidence = resolve_evidence(trail, {"c1": unchanged, "c2": drifted}.get)

    by_id = {e.chunk_id: e for e in evidence}
    assert by_id["c1"].status == "ok" and by_id["c1"].text == unchanged.text
    assert by_id["c2"].status == "changed"


def test_evidence_resolution_reports_a_chunk_that_no_longer_exists():
    obs, _ = _generated_trace()
    trail = build_audit_trail(obs)

    evidence = resolve_evidence(trail, {"c1": _chunk("c1")}.get)

    assert {e.chunk_id: e.status for e in evidence}["c2"] == "missing"


def test_markdown_report_shows_the_decision_path_evidence_and_answer():
    obs, _ = _generated_trace()
    trail = build_audit_trail(obs)
    evidence = resolve_evidence(trail, {"c1": _chunk("c1"), "c2": _chunk("c2")}.get)

    md = render_markdown(trail, evidence)

    for expected in ("how do I use dependency injection", "docs_howto", "[1]", "some real chunk text for c1", "use Depends", "v1"):
        assert expected in md


def test_rewritten_query_is_surfaced_from_the_root_output():
    obs = [
        {"name": "answer_query", "type": "SPAN", "input": {"query_id": "q", "query": "what about it?"},
         "output": {"cache_status": "exact_hit", "rewritten_query": "standalone"}, "latency_ms": 1.0},
        {"name": "cache_lookup", "type": "SPAN", "input": {}, "output": {"cache_status": "exact_hit"}, "latency_ms": 1.0},
    ]
    trail = build_audit_trail(obs, trace_id="t")
    assert trail.query_text == "what about it?" and trail.rewritten_query == "standalone"
    assert "standalone" in render_markdown(trail, [])
