"""A content-addressed cache of cross-encoder scores (design record 16i).

A reranker score is a pure function of (model, query text, chunk text). Scoring is
the one expensive step of the escalation check: 1.32 s per pair on an M1 CPU, 538
distinct pairs for the 180 dev queries, over the 10-minute local compute limit and far
slower on a hosted runner. So pairs are scored once (on a GPU, see
experiments/16f_kaggle_rerank_pairs.py), committed as plain JSON, and looked up by a
hash of the exact inputs.

That keeps the check honest rather than frozen. Changing confidence weights, bounds
or tau re-evaluates from cached scores for free (that is the logic under test). A
change that alters which chunks reach the reranker, or the model, produces keys that
are not in the cache: they are computed live, up to `max_uncached`, and beyond that
the gate fails loudly rather than quietly spending an hour or serving a stale score.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

ScoreFn = Callable[[list[tuple[str, str]]], list[float]]


class TooManyUncachedPairs(RuntimeError):
    """The inputs changed for more pairs than a CI run may score on CPU."""


def pair_key(model_name: str, query: str, chunk_text: str) -> str:
    # JSON-encoded so text cannot move between fields to forge a collision
    # (("ab", "c") vs ("a", "bc")).
    return hashlib.sha256(json.dumps([model_name, query, chunk_text], ensure_ascii=False).encode()).hexdigest()


class ScoreCache:
    def __init__(self, scores: dict[str, float] | None = None, meta: dict[str, str] | None = None) -> None:
        self.scores: dict[str, float] = dict(scores or {})
        self.meta: dict[str, str] = dict(meta or {})
        self.hits = 0
        self.misses = 0

    @classmethod
    def load(cls, path: Path) -> ScoreCache:
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(raw["scores"], raw.get("meta"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"meta": self.meta, "scores": self.scores}, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )


def make_score_fn(model_name: str, inner: ScoreFn, cache: ScoreCache, max_uncached: int) -> ScoreFn:
    """Wrap `inner` (the real cross-encoder) so only uncached pairs reach it. The miss
    budget is cumulative over the run and checked *before* the model is called."""

    def score(pairs: list[tuple[str, str]]) -> list[float]:
        keys = [pair_key(model_name, q, c) for q, c in pairs]
        missing: dict[str, tuple[str, str]] = {}
        for key, pair in zip(keys, pairs):
            if key not in cache.scores and key not in missing:
                missing[key] = pair
        if missing and cache.misses + len(missing) > max_uncached:
            raise TooManyUncachedPairs(
                f"{cache.misses + len(missing)} uncached reranker pairs exceed the limit of {max_uncached}: the "
                "reranker's inputs or model changed for many pairs. Refresh the cache "
                "(experiments/16e_export_rerank_pairs.py, then Kaggle) instead of scoring them on CPU in CI."
            )
        if missing:
            for key, value in zip(missing, inner(list(missing.values()))):
                cache.scores[key] = float(value)
            cache.misses += len(missing)
        cache.hits += len(pairs) - len(missing)
        return [cache.scores[key] for key in keys]

    return score
