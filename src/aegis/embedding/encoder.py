"""Encoder wrapper: `sentence-transformers` + MPS + batching + on-disk cache.

Not optional (plan.md Phase 3) — this corpus gets re-embedded dozens of
times over the project's life. The cache-interaction logic (skip already-
cached texts, only compute and write back what's missing, preserve input
order including duplicates) is pure and easy to get silently wrong, so it's
dependency-injectable and unit-tested with a fake encoder (`tests/
test_encoder.py`) — the real `sentence-transformers` path is exercised by
the Phase 3 measurement scripts instead, where it's genuinely I/O/GPU-bound.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from aegis.config import settings
from aegis.embedding.cache import EmbeddingCache
from aegis.ingest.hashing import content_hash

EncodeFn = Callable[[list[str]], np.ndarray]

_MODEL_CACHE: dict[str, object] = {}


def _default_encode_fn(model_name: str, device: str, batch_size: int) -> EncodeFn:
    def encode(texts: list[str]) -> np.ndarray:
        key = f"{model_name}::{device}"
        if key not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer

            _MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
        model = _MODEL_CACHE[key]
        return np.asarray(
            model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    return encode


class Encoder:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int | None = None,
        cache: EmbeddingCache | None = None,
        encode_fn: EncodeFn | None = None,
    ):
        self.model_name = model_name
        self.device = device or settings.embedding.device
        self.batch_size = batch_size or settings.embedding.batch_size
        self.cache = cache
        self._encode_fn = encode_fn or _default_encode_fn(self.model_name, self.device, self.batch_size)

    def warm_up(self, text: str) -> None:
        """Load the model and run one forward pass, bypassing the embedding
        cache in both directions. `encode()` can't warm anything when its input
        is already cached -- which the fixed warm-up query always is after the
        first run -- so the model would otherwise load on the first real query
        (measured: 10.5 s, design record 14h)."""
        self._encode_fn([text])

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        hashes = [content_hash(t) for t in texts]
        resolved: dict[str, np.ndarray] = {}

        if self.cache is not None:
            resolved.update(self.cache.get_many(self.model_name, list(set(hashes))))

        # De-duplicate: a repeated text (within this call, or already cached)
        # is only ever sent to the model once.
        missing_hashes = [h for h in dict.fromkeys(hashes) if h not in resolved]
        if missing_hashes:
            hash_to_text = dict(zip(hashes, texts))
            missing_texts = [hash_to_text[h] for h in missing_hashes]
            vectors = self._encode_fn(missing_texts)
            new_items = {h: vectors[i] for i, h in enumerate(missing_hashes)}
            resolved.update(new_items)
            if self.cache is not None:
                self.cache.set_many(self.model_name, new_items)

        return np.stack([resolved[h] for h in hashes])
