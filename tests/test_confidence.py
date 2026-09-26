"""TDD for the confidence scorer (plan.md Phase 10, step 1) -- CLAUDE.md
names this component explicitly as one where "a silent bug corrupts every
downstream measurement." Each signal is tested against a fake reranker/
encoder with known outputs, so the arithmetic is checked exactly, not just
"runs without crashing." The real bge-reranker-v2-m3 + bge-large path is
exercised by experiments/10a_tau_sweep.py, where it's genuinely
compute-bound (design record 10 covers the real per-pair cost).
"""

import numpy as np
import pytest

from aegis.guardrails.confidence import (
    SIGNAL_NAMES,
    ConfidenceScorer,
    SignalBounds,
    build_escalation_payload,
    decide_escalation,
    handle,
)
from aegis.types import Chunk, DocType


def _chunk(id: str, text: str = "") -> Chunk:
    return Chunk(
        id=id,
        doc_id=id,
        text=text or f"text for {id}",
        doc_type=DocType.DOCUMENTATION,
        content_hash="hash",
        token_count=3,
        embedding_text=text or f"text for {id}",
    )


A, B, C = (_chunk(x) for x in ["a", "b", "c"])

IDENTITY_BOUNDS = {name: SignalBounds(0.0, 1.0) for name in SIGNAL_NAMES}
EQUAL_WEIGHTS = {name: 0.25 for name in SIGNAL_NAMES}


class _FakeReranker:
    """Scores by a fixed lookup table keyed on chunk id, and records every
    call it receives -- same spy shape as tests/test_cross_encoder.py's
    _FakeScorer."""

    def __init__(self, scores_by_id: dict[str, float]):
        self.scores_by_id = scores_by_id
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, candidates: list[tuple[Chunk, float]], top_k: int) -> list[tuple[Chunk, float]]:
        self.calls.append((query, [c.id for c, _ in candidates], top_k))
        scored = sorted(((c, self.scores_by_id[c.id]) for c, _ in candidates), key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


def _fake_encode_fn(vectors_by_text: dict[str, list[float]]):
    def encode(texts: list[str]) -> np.ndarray:
        return np.array([vectors_by_text[t] for t in texts], dtype=np.float64)

    return encode


# --- reranker_top_score / score_gap (need the cross-encoder) ---


def test_reranker_top_score_is_the_best_reranked_score():
    reranker = _FakeReranker({"a": 0.2, "b": 0.9, "c": 0.5})
    encode_fn = _fake_encode_fn({f"text for {x}": [1, 0] for x in "abc"})
    scorer = ConfidenceScorer(reranker, encode_fn, weights=EQUAL_WEIGHTS, bounds=IDENTITY_BOUNDS, rerank_pool_size=3)

    result = scorer.score("q", [(A, 0.9), (B, 0.5), (C, 0.7)])

    assert result.raw["reranker_top_score"] == pytest.approx(0.9)


def test_score_gap_is_top_minus_second_reranked_score():
    reranker = _FakeReranker({"a": 0.2, "b": 0.9, "c": 0.5})
    encode_fn = _fake_encode_fn({f"text for {x}": [1, 0] for x in "abc"})
    scorer = ConfidenceScorer(reranker, encode_fn, weights=EQUAL_WEIGHTS, bounds=IDENTITY_BOUNDS, rerank_pool_size=3)

    result = scorer.score("q", [(A, 0.9), (B, 0.5), (C, 0.7)])

    assert result.raw["score_gap"] == pytest.approx(0.9 - 0.5)  # reranked order: b (0.9), c (0.5), a (0.2)


def test_reranker_only_scores_the_configured_pool_size():
    reranker = _FakeReranker({"a": 0.9, "b": 0.5, "c": 0.1})
    encode_fn = _fake_encode_fn({f"text for {x}": [1, 0] for x in "abc"})
    scorer = ConfidenceScorer(reranker, encode_fn, weights=EQUAL_WEIGHTS, bounds=IDENTITY_BOUNDS, rerank_pool_size=2)

    scorer.score("q", [(A, 0.9), (B, 0.5), (C, 0.7)])

    assert reranker.calls[0][1] == ["a", "b"]  # only the first 2 (dense-ranked) candidates were sent to the reranker


# --- semantic_agreement (dense embeddings, no reranker) ---


def test_semantic_agreement_is_zero_for_orthogonal_embeddings():
    reranker = _FakeReranker({"a": 0.5, "b": 0.5})
    encode_fn = _fake_encode_fn({"text for a": [1, 0], "text for b": [0, 1]})
    scorer = ConfidenceScorer(reranker, encode_fn, weights=EQUAL_WEIGHTS, bounds=IDENTITY_BOUNDS, rerank_pool_size=2)

    result = scorer.score("q", [(A, 0.9), (B, 0.8)])

    assert result.raw["semantic_agreement"] == pytest.approx(0.0)


def test_semantic_agreement_is_one_for_identical_embeddings():
    reranker = _FakeReranker({"a": 0.5, "b": 0.5})
    encode_fn = _fake_encode_fn({"text for a": [1, 0], "text for b": [1, 0]})
    scorer = ConfidenceScorer(reranker, encode_fn, weights=EQUAL_WEIGHTS, bounds=IDENTITY_BOUNDS, rerank_pool_size=2)

    result = scorer.score("q", [(A, 0.9), (B, 0.8)])

    assert result.raw["semantic_agreement"] == pytest.approx(1.0)


# --- distribution_shape (dense retrieval scores, no reranker) ---


def test_distribution_shape_is_zero_for_a_flat_score_list():
    reranker = _FakeReranker({"a": 0.5, "b": 0.5, "c": 0.5})
    encode_fn = _fake_encode_fn({f"text for {x}": [1, 0] for x in "abc"})
    scorer = ConfidenceScorer(reranker, encode_fn, weights=EQUAL_WEIGHTS, bounds=IDENTITY_BOUNDS, rerank_pool_size=3)

    result = scorer.score("q", [(A, 0.7), (B, 0.7), (C, 0.7)])

    assert result.raw["distribution_shape"] == pytest.approx(0.0)


def test_distribution_shape_is_positive_when_rank_one_stands_out():
    reranker = _FakeReranker({"a": 0.9, "b": 0.5, "c": 0.5})
    encode_fn = _fake_encode_fn({f"text for {x}": [1, 0] for x in "abc"})
    scorer = ConfidenceScorer(reranker, encode_fn, weights=EQUAL_WEIGHTS, bounds=IDENTITY_BOUNDS, rerank_pool_size=3)

    result = scorer.score("q", [(A, 0.9), (B, 0.3), (C, 0.3)])

    assert result.raw["distribution_shape"] > 0


# --- normalization + weighted combination ---


def test_normalization_clips_out_of_range_values_to_0_and_1():
    reranker = _FakeReranker({"a": 5.0, "b": -5.0})
    encode_fn = _fake_encode_fn({"text for a": [1, 0], "text for b": [1, 0]})
    scorer = ConfidenceScorer(reranker, encode_fn, weights=EQUAL_WEIGHTS, bounds=IDENTITY_BOUNDS, rerank_pool_size=2)

    result = scorer.score("q", [(A, 0.9), (B, 0.5)])

    assert result.normalized["reranker_top_score"] == 1.0


def test_confidence_is_the_weighted_average_of_normalized_signals():
    reranker = _FakeReranker({"a": 1.0, "b": 0.0})
    encode_fn = _fake_encode_fn({"text for a": [1, 0], "text for b": [1, 0]})
    weights = {"reranker_top_score": 1.0, "score_gap": 0.0, "semantic_agreement": 0.0, "distribution_shape": 0.0}
    scorer = ConfidenceScorer(reranker, encode_fn, weights=weights, bounds=IDENTITY_BOUNDS, rerank_pool_size=2)

    result = scorer.score("q", [(A, 0.9), (B, 0.5)])

    assert result.confidence == pytest.approx(1.0)  # only reranker_top_score is weighted, normalized to 1.0


def test_scorer_builds_from_settings_defaults_without_explicit_weights_or_bounds():
    reranker = _FakeReranker({"a": 0.5})
    encode_fn = _fake_encode_fn({"text for a": [1, 0]})
    scorer = ConfidenceScorer(reranker, encode_fn)

    result = scorer.score("q", [(A, 0.5)])

    assert set(result.raw) == set(result.normalized) == set(SIGNAL_NAMES)
    assert 0.0 <= result.confidence <= 1.0


# --- edge cases ---


def test_empty_candidates_produce_minimum_confidence_without_calling_reranker_or_encoder():
    reranker = _FakeReranker({})
    encode_calls: list[list[str]] = []

    def encode_fn(texts: list[str]) -> np.ndarray:
        encode_calls.append(texts)
        return np.zeros((0, 2))

    scorer = ConfidenceScorer(reranker, encode_fn, weights=EQUAL_WEIGHTS, bounds=IDENTITY_BOUNDS)

    result = scorer.score("q", [])

    assert result.confidence == pytest.approx(0.0)
    assert reranker.calls == []
    assert encode_calls == []


def test_single_candidate_has_zero_gap_and_zero_agreement_but_a_real_top_score():
    reranker = _FakeReranker({"a": 0.8})
    encode_fn = _fake_encode_fn({"text for a": [1, 0]})
    scorer = ConfidenceScorer(reranker, encode_fn, weights=EQUAL_WEIGHTS, bounds=IDENTITY_BOUNDS)

    result = scorer.score("q", [(A, 0.9)])

    assert result.raw["reranker_top_score"] == pytest.approx(0.8)
    assert result.raw["score_gap"] == pytest.approx(0.0)
    assert result.raw["semantic_agreement"] == pytest.approx(0.0)
    assert result.raw["distribution_shape"] == pytest.approx(0.0)


# --- decide_escalation: pure function, same shape as router/policy.py's decide() ---


def test_confidence_below_tau_escalates():
    outcome = decide_escalation(0.3, tau=0.5)
    assert outcome.escalate is True


def test_confidence_at_or_above_tau_does_not_escalate():
    outcome = decide_escalation(0.5, tau=0.5)
    assert outcome.escalate is False


# --- handle(): enforcement point, same proof shape as
# test_router_policy.py::test_generate_fn_that_raises_on_call_proves_billing_never_invokes_it ---


def test_handle_calls_generate_fn_when_confident_and_never_touches_escalate_fn():
    reranker = _FakeReranker({"a": 0.9, "b": 0.9})
    encode_fn = _fake_encode_fn({"text for a": [1, 0], "text for b": [1, 0]})
    scorer = ConfidenceScorer(reranker, encode_fn, weights=EQUAL_WEIGHTS, bounds=IDENTITY_BOUNDS)

    def escalate_fn(query: str, payload) -> str:
        raise AssertionError("escalate_fn must not be called when confidence >= tau")

    result = handle(
        "q", [(A, 0.9), (B, 0.8)], scorer, generate_fn=lambda q: "answer", escalate_fn=escalate_fn, tau=0.1
    )

    assert result == "answer"


def test_handle_calls_escalate_fn_when_not_confident_and_never_touches_generate_fn():
    reranker = _FakeReranker({"a": 0.0, "b": 0.0})
    encode_fn = _fake_encode_fn({"text for a": [1, 0], "text for b": [0, 1]})
    scorer = ConfidenceScorer(reranker, encode_fn, weights=EQUAL_WEIGHTS, bounds=IDENTITY_BOUNDS)

    def generate_fn(query: str) -> str:
        raise AssertionError("generate_fn must not be called when confidence < tau")

    result = handle(
        "q", [(A, 0.1), (B, 0.1)], scorer, generate_fn=generate_fn, escalate_fn=lambda q, p: "escalated", tau=0.99
    )

    assert result == "escalated"


# --- build_escalation_payload ---


def test_escalation_payload_names_the_weak_signals_and_carries_top_chunks():
    reranker = _FakeReranker({"a": 0.0})
    encode_fn = _fake_encode_fn({"text for a": [1, 0]})
    scorer = ConfidenceScorer(reranker, encode_fn, weights=EQUAL_WEIGHTS, bounds=IDENTITY_BOUNDS)

    result = scorer.score("q", [(A, 0.1)])
    outcome = decide_escalation(result.confidence, tau=0.99)
    payload = build_escalation_payload("q", [(A, 0.1)], result, outcome)

    assert "reranker_top_score" in payload.reason
    assert payload.top_chunks[0]["chunk_id"] == "a"
    assert payload.query == "q"
