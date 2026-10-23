"""Semantic response cache (plan.md Phase 12, step 2): embed the query,
serve the best real cache entry above a similarity threshold. The
threshold itself is real, measured data (`experiments/
12a_semantic_cache_threshold_sweep.py`, `docs/results/semantic-cache-
threshold.md`, `CacheSettings.semantic_similarity_threshold` default
0.88) -- too loose serves a confident wrong answer to a different-but-
similar question, a hallucination vector this project built itself,
per plan.md's own warning.

Same `corpus_version`-scoping and `CachedAnswer` payload as the exact-
match cache (`cache/exact_match.py`) -- this is the fallback tier the
architecture diagram (plan.md Sec 3) checks after an exact-match miss,
not a separate concern.

Brute-force cosine similarity over whatever's cached for the current
corpus_version, not an ANN index: a query cache's real working set is
orders of magnitude smaller than the 19k-chunk corpus `DenseIndex` (Phase
3) exists to search, so HNSW's approximate-search tradeoff isn't earned
here -- exact, simple, and fast enough at this scale.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from aegis.cache.exact_match import CachedAnswer, cache_scope


class _Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b)) / denom


class SemanticCache:
    def __init__(self, path: Path, encoder: _Encoder, threshold: float):
        self.encoder = encoder
        self.threshold = threshold
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS semantic_cache ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, corpus_version TEXT NOT NULL, "
                "dim INTEGER NOT NULL, vector BLOB NOT NULL, payload TEXT NOT NULL)"
            )
            self._conn.commit()

    def get(self, query_text: str, corpus_version: str, *, partition: str) -> CachedAnswer | None:
        query_vector = self.encoder.encode([query_text])[0]
        with self._lock:
            # The `corpus_version` column holds the full scope (version +
            # partition, design record 15b); rows written before that hold a
            # bare version and are simply never matched again.
            rows = self._conn.execute(
                "SELECT dim, vector, payload FROM semantic_cache WHERE corpus_version = ?",
                (cache_scope(corpus_version, partition),),
            ).fetchall()

        best_answer: CachedAnswer | None = None
        best_similarity = -1.0
        for dim, blob, payload in rows:
            candidate_vector = np.frombuffer(blob, dtype=np.float32).reshape(dim)
            similarity = _cosine_similarity(query_vector, candidate_vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_answer = CachedAnswer.model_validate_json(payload)

        if best_answer is not None and best_similarity >= self.threshold:
            return best_answer
        return None

    def set(self, query_text: str, corpus_version: str, answer: CachedAnswer, *, partition: str) -> None:
        vector = np.asarray(self.encoder.encode([query_text])[0], dtype=np.float32)
        with self._lock:
            self._conn.execute(
                "INSERT INTO semantic_cache (corpus_version, dim, vector, payload) VALUES (?, ?, ?, ?)",
                (cache_scope(corpus_version, partition), vector.shape[0], vector.tobytes(), answer.model_dump_json()),
            )
            self._conn.commit()
