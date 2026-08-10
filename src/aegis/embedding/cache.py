"""On-disk embedding cache, keyed by `(model, content_hash)`.

Not optional (plan.md Phase 3): this corpus gets re-embedded dozens of times
over the life of the project — every chunking-strategy change, every HNSW
sweep re-run, every time a script is re-run after a crash. Content-hash
keying (reusing `aegis.ingest.hashing.content_hash`) means a chunk's text
changing invalidates its cache entry automatically; nothing changing means
zero recompute.

Thread-safe (Phase 6): the eval harness parallelizes retrieval across
golden-set queries with a `ThreadPoolExecutor`, and `DenseIndex.search`
calls into this cache from whichever worker thread handles a given query
— not the thread that constructed it. A plain `sqlite3.Connection` raises
`ProgrammingError` the moment that happens (`tests/test_embedding_cache.py
::test_get_many_is_safe_from_a_worker_thread` reproduces it directly), so
the connection is opened with `check_same_thread=False` and every access
serialized through a lock — sqlite3 connections aren't safe for
*concurrent* use even once that check is disabled.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np


class EmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "model TEXT NOT NULL, content_hash TEXT NOT NULL, "
                "dim INTEGER NOT NULL, vector BLOB NOT NULL, "
                "PRIMARY KEY (model, content_hash))"
            )
            self._conn.commit()

    def get(self, model: str, content_hash: str) -> np.ndarray | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT dim, vector FROM embeddings WHERE model = ? AND content_hash = ?",
                (model, content_hash),
            ).fetchone()
        if row is None:
            return None
        dim, blob = row
        return np.frombuffer(blob, dtype=np.float32).reshape(dim)

    def set(self, model: str, content_hash: str, vector: np.ndarray) -> None:
        self.set_many(model, {content_hash: vector})

    def get_many(self, model: str, content_hashes: list[str]) -> dict[str, np.ndarray]:
        if not content_hashes:
            return {}
        placeholders = ",".join("?" for _ in content_hashes)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT content_hash, dim, vector FROM embeddings "
                f"WHERE model = ? AND content_hash IN ({placeholders})",
                (model, *content_hashes),
            ).fetchall()
        return {h: np.frombuffer(blob, dtype=np.float32).reshape(dim) for h, dim, blob in rows}

    def set_many(self, model: str, items: dict[str, np.ndarray]) -> None:
        if not items:
            return
        rows = [
            (model, content_hash, int(vector.shape[0]), np.asarray(vector, dtype=np.float32).tobytes())
            for content_hash, vector in items.items()
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (model, content_hash, dim, vector) VALUES (?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
