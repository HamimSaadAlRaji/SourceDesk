"""Confidence scorer (plan.md Phase 10, step 1): a transparent weighted
combination of four signals over a query's top retrieved chunks --
"an interpretable score you can debug beats a marginally better one you
can't" (plan.md). Feeds `decide_escalation` (step 2's tau sweep) and
`build_escalation_payload` (step 5): the fork in the architecture diagram
(plan.md §3) between "answer" and "escalate to a human."

Four signals, two cheap (already-computed dense retrieval scores and
embeddings) and two needing one small extra cross-encoder call:

  1. `reranker_top_score` -- the cross-encoder's score for the single best
     candidate. A sharp, specific relevance judgment (design record 07's
     attention-pattern mechanism) -- the same property Phase 7 found makes
     a cross-encoder lose to a bi-encoder at *reranking* against this
     golden set's deliberately broad ("biased toward inclusion") labels,
     but that's a different question from *how confident should the system
     be in this one top chunk*. Design record 10 checks directly whether
     that sharpness helps here where it hurt there.
  2. `score_gap` -- reranker_score[rank1] - reranker_score[rank2]. A single
     standout answer vs. several similarly-scored candidates.
  3. `semantic_agreement` -- mean pairwise cosine similarity among the
     dense embeddings of the top-k retrieved chunks (cosine == dot product
     for these normalized vectors, design record 03). Do the top results
     corroborate each other, or scatter across unrelated topics?
  4. `distribution_shape` -- how far rank-1's *dense* score stands out from
     the rest of the top-k pack: (score[0] - mean(rest)) / std(rest). A
     deliberately different, cheaper signal than #1/#2: the bi-encoder's
     view of the whole pool, not just the reranked top pair.

Only #1 and #2 touch the cross-encoder, and only over a small pool
(`rerank_pool_size`, default 3 -- rank 1 and rank 2 are all either signal
reads). Real per-pair cost on this machine's CPU, measured against real
corpus-length chunks: ~65-1400ms, mean ~320ms across a real probe (design
record 10 §5). Reranking all of top-10 for a 180-query dev sweep risked
crossing CLAUDE.md's 10-minute local-compute budget for signal never read;
reranking only 3 keeps it comfortably under.

Each raw signal is min-max normalized to [0, 1] using *winsorized* (1st/99th
percentile, not raw min/max) bounds measured from the real 180-query dev
sweep (`ConfidenceSettings`, filled in from `experiments/10a_tau_sweep.py`'s
output -- same "measure first, hardcode the real number" convention as
`QuerySettings.low_confidence_score_threshold`). Winsorizing matters here:
`distribution_shape`'s raw max on the real dev set was a single ~76,856
outlier (near-zero variance among one query's non-top-1 dense scores blew up
its z-score-like ratio -- ties to Open Question #6's near-duplicate-chunk
corpus property), which under plain min-max would have compressed every
other query's real range into a sliver near 0.

plan.md's step 1 starts from "a transparent weighted combination" -- tried
equal (0.25 each) first. Measured result: equal weighting scored *worse*
than `reranker_top_score` alone (0.8333 vs 0.9333 recall at precision>=0.5,
real dev sweep), because `score_gap` and `distribution_shape` are both
close to uncorrelated with the real should-escalate label and dilute the
one strong signal. Weights are reweighted by measured rank-correlation
instead (0.60/0.15/0.20/0.05) -- confirmed with the user given it's a real
deviation from plan.md's literal starting point. See design record 10 §5
for the full ablation and `docs/results/confusion-matrix.md` for the curve.

`decide_escalation`/`handle` mirror `router/policy.py`'s pure-decision +
enforcement-point split: `decide_escalation` is trivial to audit, `handle`
is the only place a confidence score turns into either a generation call or
an escalation, and it structurally cannot call both (proved the same way
router/policy.py's test does, with a spy that raises if invoked).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from pydantic import BaseModel

from aegis.config import settings
from aegis.embedding.encoder import EncodeFn
from aegis.types import Chunk

SIGNAL_NAMES = ("reranker_top_score", "score_gap", "semantic_agreement", "distribution_shape")

_EPS = 1e-6
_WEAK_SIGNAL_THRESHOLD = 0.3  # a normalized signal below this is named in the escalation payload's reason


class _Reranker(Protocol):
    def rerank(self, query: str, candidates: list[tuple[Chunk, float]], top_k: int) -> list[tuple[Chunk, float]]: ...


@dataclass(frozen=True)
class SignalBounds:
    lo: float
    hi: float

    def normalize(self, value: float) -> float:
        if self.hi <= self.lo:
            return 0.0
        return min(1.0, max(0.0, (value - self.lo) / (self.hi - self.lo)))


def _default_bounds() -> dict[str, SignalBounds]:
    c = settings.confidence
    return {
        "reranker_top_score": SignalBounds(c.reranker_top_score_lo, c.reranker_top_score_hi),
        "score_gap": SignalBounds(c.score_gap_lo, c.score_gap_hi),
        "semantic_agreement": SignalBounds(c.semantic_agreement_lo, c.semantic_agreement_hi),
        "distribution_shape": SignalBounds(c.distribution_shape_lo, c.distribution_shape_hi),
    }


def _default_weights() -> dict[str, float]:
    c = settings.confidence
    return {
        "reranker_top_score": c.weight_reranker_top_score,
        "score_gap": c.weight_score_gap,
        "semantic_agreement": c.weight_semantic_agreement,
        "distribution_shape": c.weight_distribution_shape,
    }


class ConfidenceResult(BaseModel):
    raw: dict[str, float]
    normalized: dict[str, float]
    confidence: float


def _raw_signals(
    query: str,
    candidates: list[tuple[Chunk, float]],
    reranker: _Reranker,
    encode_fn: EncodeFn,
    rerank_pool_size: int,
) -> dict[str, float]:
    if not candidates:
        return dict.fromkeys(SIGNAL_NAMES, 0.0)

    pool = candidates[:rerank_pool_size]
    reranked = reranker.rerank(query, pool, top_k=len(pool))
    reranker_top_score = reranked[0][1]
    score_gap = reranked[0][1] - reranked[1][1] if len(reranked) > 1 else 0.0

    dense_scores = np.array([score for _, score in candidates], dtype=np.float64)
    if len(dense_scores) > 1:
        rest = dense_scores[1:]
        distribution_shape = float((dense_scores[0] - rest.mean()) / (rest.std() + _EPS))
    else:
        distribution_shape = 0.0

    if len(candidates) > 1:
        embeddings = encode_fn([chunk.text for chunk, _ in candidates])
        sims = embeddings @ embeddings.T  # cosine == dot product for normalized embeddings (design record 03)
        upper = sims[np.triu_indices(sims.shape[0], k=1)]
        semantic_agreement = float(upper.mean()) if upper.size else 0.0
    else:
        semantic_agreement = 0.0

    return {
        "reranker_top_score": float(reranker_top_score),
        "score_gap": float(score_gap),
        "semantic_agreement": semantic_agreement,
        "distribution_shape": distribution_shape,
    }


class ConfidenceScorer:
    def __init__(
        self,
        reranker: _Reranker,
        encode_fn: EncodeFn,
        weights: dict[str, float] | None = None,
        bounds: dict[str, SignalBounds] | None = None,
        rerank_pool_size: int = settings.confidence.rerank_pool_size,
    ) -> None:
        self.reranker = reranker
        self.encode_fn = encode_fn
        self.weights = weights or _default_weights()
        self.bounds = bounds or _default_bounds()
        self.rerank_pool_size = rerank_pool_size

    def score(self, query: str, candidates: list[tuple[Chunk, float]]) -> ConfidenceResult:
        raw = _raw_signals(query, candidates, self.reranker, self.encode_fn, self.rerank_pool_size)
        normalized = {name: self.bounds[name].normalize(raw[name]) for name in SIGNAL_NAMES}

        weight_sum = sum(self.weights[name] for name in SIGNAL_NAMES) or 1.0
        confidence = sum(normalized[name] * self.weights[name] for name in SIGNAL_NAMES) / weight_sum

        return ConfidenceResult(raw=raw, normalized=normalized, confidence=confidence)


@dataclass(frozen=True)
class EscalationOutcome:
    escalate: bool
    confidence: float
    tau: float
    reason: str


def decide_escalation(confidence: float, tau: float = settings.confidence.tau) -> EscalationOutcome:
    if confidence < tau:
        return EscalationOutcome(
            escalate=True, confidence=confidence, tau=tau, reason=f"confidence {confidence:.4f} < tau {tau:.4f}"
        )
    return EscalationOutcome(
        escalate=False, confidence=confidence, tau=tau, reason=f"confidence {confidence:.4f} >= tau {tau:.4f}"
    )


class EscalationPayload(BaseModel):
    """plan.md Phase 10 step 5: "auto-summary of the question, what was
    retrieved, why confidence was low. Not a cold start." -- everything a
    human agent needs to pick up the query without re-running the pipeline."""

    query: str
    top_chunks: list[dict]
    signals: ConfidenceResult
    tau: float
    reason: str


def build_escalation_payload(
    query: str,
    candidates: list[tuple[Chunk, float]],
    result: ConfidenceResult,
    outcome: EscalationOutcome,
) -> EscalationPayload:
    weak = [name for name in SIGNAL_NAMES if result.normalized[name] < _WEAK_SIGNAL_THRESHOLD]
    reason = f"Confidence {result.confidence:.3f} fell below tau={outcome.tau:.3f}. " + (
        f"Weak signals: {', '.join(weak)}." if weak else "No single signal was weak; the combination fell short."
    )
    top_chunks = [
        {
            "chunk_id": chunk.id,
            "doc_id": chunk.doc_id,
            "heading_path": chunk.heading_path,
            "retrieval_score": round(score, 4),
        }
        for chunk, score in candidates[:5]
    ]
    return EscalationPayload(query=query, top_chunks=top_chunks, signals=result, tau=outcome.tau, reason=reason)


GenerateFn = Callable[[str], str]
EscalateFn = Callable[[str, EscalationPayload], str]


def handle(
    query: str,
    candidates: list[tuple[Chunk, float]],
    scorer: ConfidenceScorer,
    generate_fn: GenerateFn,
    escalate_fn: EscalateFn,
    tau: float = settings.confidence.tau,
) -> str:
    result = scorer.score(query, candidates)
    outcome = decide_escalation(result.confidence, tau)
    if outcome.escalate:
        payload = build_escalation_payload(query, candidates, result, outcome)
        return escalate_fn(query, payload)
    return generate_fn(query)
