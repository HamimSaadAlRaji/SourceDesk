"""Cross-encoder reranker (plan.md Phase 7, step 1): rescore a pool of
bi-encoder candidates by scoring `(query, chunk_text)` jointly through one
transformer, catching query-chunk interactions independent dense encoding
loses -- see docs/explainers/07-reranking-and-encoder-architectures.md for
the attention-pattern mechanism behind why this works.

Scores `chunk.text`, not `chunk.embedding_text` -- the contextual-enrichment
prefix (Phase 2) is a bi-encoder-only technique for compensating for lost
context in independent encoding; a cross-encoder already attends over the
query and passage jointly, so it needs the real source text a reader would
see, the same distinction `aegis.types.Chunk` draws for citations.

Reorder/truncate logic is pure and unit-tested with a fake `score_fn`
(tests/test_cross_encoder.py) -- the real `bge-reranker-v2-m3` path is
GPU-bound and exercised by `experiments/07_rerank_sweep.py`, the same split
`aegis.embedding.encoder.Encoder` uses for its own real model.
"""

from __future__ import annotations

from collections.abc import Callable

from aegis.config import settings
from aegis.types import Chunk

ScoreFn = Callable[[list[tuple[str, str]]], list[float]]

_MODEL_CACHE: dict[str, object] = {}


def _default_score_fn(model_name: str, device: str, batch_size: int) -> ScoreFn:
    def score(pairs: list[tuple[str, str]]) -> list[float]:
        key = f"{model_name}::{device}"
        if key not in _MODEL_CACHE:
            from sentence_transformers import CrossEncoder

            _MODEL_CACHE[key] = CrossEncoder(model_name, device=device)
        model = _MODEL_CACHE[key]
        return [float(s) for s in model.predict(pairs, batch_size=batch_size, show_progress_bar=False)]

    return score


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        score_fn: ScoreFn | None = None,
    ) -> None:
        self.model_name = model_name or settings.rerank.model_name
        self.device = device or settings.rerank.device
        self.batch_size = batch_size or settings.rerank.batch_size
        self._score_fn = score_fn or _default_score_fn(self.model_name, self.device, self.batch_size)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []

        pairs = [(query, chunk.text) for chunk, _ in candidates]
        scores = self._score_fn(pairs)

        reranked = sorted(
            ((chunk, score) for (chunk, _bi_encoder_score), score in zip(candidates, scores)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return reranked[:top_k]
