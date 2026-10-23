"""Exact-match response cache (plan.md Phase 12, step 1): a normalized
query hash maps to a served answer, keyed with `corpus_version` so a
reindex or config change can't silently serve a response generated against
a corpus that no longer exists -- the same "stale index, stale answer"
failure mode Open Question #7 already named for the eval harness's own
cache, now a real production-path risk instead of a dev-tool annoyance.

Stores a purpose-built `CachedAnswer`, not the full `GenerationResult`
(`generation/pipeline.py`): a cache entry only needs what's required to
*serve* a response (the text, which chunk ids it cites, which provider
produced it) -- the retrieved chunks' full text is already durably stored
in the corpus/vector store, so duplicating it into every cache entry would
bloat storage for zero benefit and couple this cache's schema to
`PackedContext`'s internal shape.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from pathlib import Path

from pydantic import BaseModel


class CitedChunk(BaseModel):
    """A validated citation with everything needed to render it, so a cache
    hit can replay `citation` stream events without re-fetching the corpus
    (the cache is already scoped to one `corpus_version`, so the text can't go
    stale relative to the entry that carries it)."""

    citation_id: int
    chunk_id: str
    doc_id: str
    heading_path: list[str] = []
    text: str


class CachedAnswer(BaseModel):
    text: str
    citations: list[str]
    provider: str | None
    # Default [] keeps entries written before this field existed readable.
    cited: list[CitedChunk] = []


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def cache_scope(corpus_version: str, partition: str) -> str:
    """What a cached answer is valid for. The partition is part of it so a hit
    can never cross the retrieval-layer partition (design record 15b): without
    it, a docs-partition caller asking the same text would be served an answer
    built from issue chunks. Shared with the semantic cache."""
    return f"{corpus_version}::partition={partition}"


def _cache_key(query_text: str, corpus_version: str, partition: str) -> str:
    normalized = normalize_query(query_text)
    return hashlib.sha256(f"{cache_scope(corpus_version, partition)}::{normalized}".encode()).hexdigest()


class ExactMatchCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS exact_match_cache ("
                "cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            self._conn.commit()

    def get(self, query_text: str, corpus_version: str, *, partition: str) -> CachedAnswer | None:
        key = _cache_key(query_text, corpus_version, partition)
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM exact_match_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return CachedAnswer.model_validate_json(row[0])

    def set(self, query_text: str, corpus_version: str, answer: CachedAnswer, *, partition: str) -> None:
        key = _cache_key(query_text, corpus_version, partition)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO exact_match_cache (cache_key, payload) VALUES (?, ?)",
                (key, answer.model_dump_json()),
            )
            self._conn.commit()
