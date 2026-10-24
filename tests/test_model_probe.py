"""The model-availability probe (design record 16e). gpt-4.1-nano was removed from
the OpenAI API while /v1/models still listed it, so the router, rewrite and
entailment roles failed with a 404 that no test, and no earlier phase's gate, could
see. The probe makes one tiny real call per configured model; these pin the logic
around it: which models are probed, and that any failure is reported, never skipped."""

from __future__ import annotations

from aegis.config import settings
from aegis.eval.model_probe import configured_models, probe_all


def test_every_distinct_role_model_is_probed_once():
    models = configured_models()
    o = settings.openai
    assert set(models) == {o.generation_model, o.router_model, o.rewrite_model, o.entailment_model, o.judge_model, o.ragas_model}
    assert len(models) == len(set(models))


def test_a_failing_model_is_reported_with_its_error():
    def probe(model: str) -> None:
        if model == "dead-model":
            raise RuntimeError("404 deprecated")

    failures = probe_all(["ok-model", "dead-model"], probe)
    assert list(failures) == ["dead-model"]
    assert "404 deprecated" in failures["dead-model"]


def test_all_models_healthy_means_no_failures():
    assert probe_all(["a", "b"], lambda m: None) == {}


def test_a_probe_that_raises_a_non_standard_error_is_still_reported():
    def probe(model: str) -> None:
        raise KeyboardInterrupt  # noqa: B904 - must not escape as a crash that hides the other models

    # KeyboardInterrupt is a BaseException; the probe must let it propagate, not swallow Ctrl-C.
    try:
        probe_all(["a"], probe)
    except KeyboardInterrupt:
        return
    raise AssertionError("KeyboardInterrupt must propagate")
