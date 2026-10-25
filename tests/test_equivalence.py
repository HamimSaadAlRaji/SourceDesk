"""Phase 17: the machinery that lets two orchestrations be compared on real LLM output.

A bug here would not fail loudly: a replay that quietly answers from the wrong entry, or a
comparison that ignores a field, makes the two paths look identical. So both are pinned."""

import threading

import pytest
from test_pipeline import _chunk

from aegis.eval.equivalence import ReplayMissError, Tape, TapedLLM, compare_results, taped_fn
from aegis.pipeline import CacheStatus, PipelineResult
from aegis.types import Query


def test_record_calls_the_function_once_and_replay_returns_the_recorded_value():
    recorder = Tape()
    calls = []
    fn = taped_fn(recorder, "router", lambda q: calls.append(q) or f"route:{q}")

    assert fn("a") == "route:a" and fn("a") == "route:a"
    assert calls == ["a"]  # second call is a tape hit, not another paid call

    replayer = Tape(recorder.entries, replay_only=True)
    assert taped_fn(replayer, "router", lambda q: pytest.fail("inner must not run on replay"))("a") == "route:a"


def test_replay_of_an_input_never_recorded_raises_instead_of_inventing_an_answer():
    replayer = Tape({}, replay_only=True)
    with pytest.raises(ReplayMissError):
        taped_fn(replayer, "router", lambda q: "x")("never seen")


def test_the_same_arguments_to_a_different_stage_do_not_collide():
    tape = Tape()
    a = taped_fn(tape, "router", lambda q: "from router")
    b = taped_fn(tape, "entailment", lambda q: "from entailment")
    assert a("same") == "from router" and b("same") == "from entailment"


def test_unused_entries_expose_a_call_the_replaying_path_never_made():
    recorder = Tape()
    fn = taped_fn(recorder, "router", lambda q: q)
    fn("a"), fn("b")

    replayer = Tape(recorder.entries, replay_only=True)
    taped_fn(replayer, "router", lambda q: q)("a")

    assert replayer.unused == 1


def test_tape_round_trips_through_disk(tmp_path):
    recorder = Tape()
    taped_fn(recorder, "router", lambda q: ["billing", 1])("a")
    recorder.save(tmp_path / "tape.json")

    loaded = Tape.load(tmp_path / "tape.json", replay_only=True)
    assert taped_fn(loaded, "router", lambda q: pytest.fail("no"))("a") == ["billing", 1]


def test_tape_is_safe_under_concurrent_recording():
    tape = Tape()
    fn = taped_fn(tape, "router", lambda q: q.upper())
    threads = [threading.Thread(target=lambda i=i: fn(f"q{i}")) for i in range(50)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(tape.entries) == 50


def test_taped_llm_keys_on_messages_and_sampling_parameters():
    class Inner:
        def __init__(self):
            self.calls = 0

        def generate_with_provider(self, messages, temperature, top_p, max_tokens):
            self.calls += 1
            return f"answer-{self.calls}", "p"

    inner = Inner()
    llm = TapedLLM(inner, Tape())
    msgs = [{"role": "user", "content": "hi"}]

    first = llm.generate_with_provider(msgs, 0.0, 1.0, 100)
    assert llm.generate_with_provider(msgs, 0.0, 1.0, 100) == first and inner.calls == 1
    assert llm.generate_with_provider(msgs, 0.7, 1.0, 100) != first  # different temperature: a different request


def _result(**over) -> PipelineResult:
    base = dict(query=Query(id="q", text="t"), cache_status=CacheStatus.MISS, escalate=False, text="a", trace_id="t1",
                stage_latencies_ms={"routing": 1.0})
    return PipelineResult(**{**base, **over})


def test_identical_results_have_no_differences_even_when_trace_id_and_timings_differ():
    a, b = _result(), _result(trace_id="t2", stage_latencies_ms={"routing": 99.0})
    assert compare_results(a, b) == []


def test_every_field_that_matters_is_compared():
    base = _result()
    for field, value in {
        "text": "different", "escalate": True, "citations": ["x"], "provider": "other", "warnings": ["w"],
        "escalation_reason": "r", "cache_status": CacheStatus.EXACT_HIT, "prompt_version": "v9",
    }.items():
        assert compare_results(base, _result(**{field: value})) == [field]


def test_retrieved_chunk_order_is_compared():
    from aegis.types import Retrieved

    r1, r2 = Retrieved(chunk=_chunk("a"), score=0.9, rank=1), Retrieved(chunk=_chunk("b"), score=0.8, rank=2)
    assert compare_results(_result(retrieved=[r1, r2]), _result(retrieved=[r2, r1])) == ["retrieved"]


def test_a_stage_that_ran_in_only_one_path_is_a_difference():
    a = _result(stage_latencies_ms={"routing": 1.0})
    b = _result(stage_latencies_ms={"routing": 1.0, "reformulation": 2.0})
    assert compare_results(a, b) == ["stages"]


def _v3_messages(key: str, context: str = "chunk text") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": f"rules ... key: {key} ==="},
        {"role": "user", "content": f"=== BEGIN CONTEXT [1] | source: c1 | type: issue | key: {key} ===\n{context}\n"
                                    f"=== END CONTEXT [1] | key: {key} ===\n\nQuestion: q"},
    ]


class _CountingLLM:
    def __init__(self):
        self.calls = 0

    def generate_with_provider(self, messages, temperature, top_p, max_tokens):
        self.calls += 1
        return f"answer-{self.calls}", "p"


def test_taped_llm_ignores_the_per_request_v3_boundary_key():
    """Prompt v3 draws a fresh random key per request (a Phase 15 defence), so two identical
    requests never match byte for byte; without normalizing it, replay could never hit."""
    inner = _CountingLLM()
    llm = TapedLLM(inner, Tape())

    first = llm.generate_with_provider(_v3_messages("0123456789abcdef"), 0.0, 1.0, 100)
    again = llm.generate_with_provider(_v3_messages("fedcba9876543210"), 0.0, 1.0, 100)

    assert again == first and inner.calls == 1


def test_taped_llm_still_distinguishes_requests_that_differ_in_content():
    inner = _CountingLLM()
    llm = TapedLLM(inner, Tape())

    llm.generate_with_provider(_v3_messages("0123456789abcdef", "chunk A"), 0.0, 1.0, 100)
    llm.generate_with_provider(_v3_messages("fedcba9876543210", "chunk B"), 0.0, 1.0, 100)

    assert inner.calls == 2
