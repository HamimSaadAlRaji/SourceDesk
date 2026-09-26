"""TDD for the semantic cache (plan.md Phase 12, step 2). The similarity-
threshold decision is the whole point of this component (a wrong threshold
is a hallucination vector, per plan.md's own framing) -- covered here with
a fake, deterministic embedder (same DI pattern as the semantic chunker's
`embed_fn`, Phase 2) so exact similarity values are controllable, not left
to a real model's incidental behavior.
"""

import numpy as np
import pytest

from aegis.cache.exact_match import CachedAnswer
from aegis.cache.semantic import SemanticCache

# A tiny fake embedding space: each text maps to a fixed 2D unit vector at a
# known angle, so cosine similarity between any two is exactly computable
# and independent of a real model's behavior.
_VECTORS = {
    "how do i use depends?": np.array([1.0, 0.0], dtype=np.float32),
    "how can i use depends in fastapi?": np.array([0.99, 0.14], dtype=np.float32),  # cos ~= 0.99
    "how do i stream a file response?": np.array([0.85, 0.53], dtype=np.float32),  # cos ~= 0.85
    "what is the weather today?": np.array([0.0, 1.0], dtype=np.float32),  # cos = 0.0
}


class _FakeEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([_VECTORS[t] for t in texts])


def _answer(text: str = "answer text") -> CachedAnswer:
    return CachedAnswer(text=text, citations=["doc::a::1"], provider="groq")


def test_get_on_empty_cache_returns_none(tmp_path):
    cache = SemanticCache(tmp_path / "cache.sqlite", encoder=_FakeEncoder(), threshold=0.88)
    assert cache.get("how do i use depends?", corpus_version="v1") is None


def test_identical_query_is_always_a_hit(tmp_path):
    cache = SemanticCache(tmp_path / "cache.sqlite", encoder=_FakeEncoder(), threshold=0.88)
    cache.set("how do i use depends?", corpus_version="v1", answer=_answer("depends answer"))

    hit = cache.get("how do i use depends?", corpus_version="v1")

    assert hit is not None
    assert hit.text == "depends answer"


def test_a_query_above_threshold_similarity_hits(tmp_path):
    cache = SemanticCache(tmp_path / "cache.sqlite", encoder=_FakeEncoder(), threshold=0.88)
    cache.set("how do i use depends?", corpus_version="v1", answer=_answer("depends answer"))

    # cos("how can i use depends in fastapi?", "how do i use depends?") ~= 0.99 >= 0.88
    hit = cache.get("how can i use depends in fastapi?", corpus_version="v1")

    assert hit is not None
    assert hit.text == "depends answer"


def test_a_query_below_threshold_similarity_misses(tmp_path):
    cache = SemanticCache(tmp_path / "cache.sqlite", encoder=_FakeEncoder(), threshold=0.88)
    cache.set("how do i use depends?", corpus_version="v1", answer=_answer("depends answer"))

    # cos("how do i stream a file response?", "how do i use depends?") ~= 0.85 < 0.88
    # -- the whole point of the threshold: a related-but-different question must NOT hit.
    assert cache.get("how do i stream a file response?", corpus_version="v1") is None


def test_a_completely_unrelated_query_misses(tmp_path):
    cache = SemanticCache(tmp_path / "cache.sqlite", encoder=_FakeEncoder(), threshold=0.88)
    cache.set("how do i use depends?", corpus_version="v1", answer=_answer("depends answer"))

    assert cache.get("what is the weather today?", corpus_version="v1") is None


def test_same_query_different_corpus_version_is_a_miss(tmp_path):
    cache = SemanticCache(tmp_path / "cache.sqlite", encoder=_FakeEncoder(), threshold=0.88)
    cache.set("how do i use depends?", corpus_version="v1", answer=_answer("depends answer"))

    assert cache.get("how do i use depends?", corpus_version="v2") is None


def test_returns_the_best_match_among_multiple_entries(tmp_path):
    cache = SemanticCache(tmp_path / "cache.sqlite", encoder=_FakeEncoder(), threshold=0.88)
    cache.set("how do i use depends?", corpus_version="v1", answer=_answer("depends answer"))
    cache.set("how can i use depends in fastapi?", corpus_version="v1", answer=_answer("closer answer"))

    # Querying with the exact second entry's text should match itself (cos=1.0),
    # not fall back to the first (cos ~= 0.99) -- best match wins.
    hit = cache.get("how can i use depends in fastapi?", corpus_version="v1")

    assert hit.text == "closer answer"


def test_persists_across_instances_reopening_the_same_path(tmp_path):
    path = tmp_path / "cache.sqlite"
    SemanticCache(path, encoder=_FakeEncoder(), threshold=0.88).set(
        "how do i use depends?", corpus_version="v1", answer=_answer("persisted")
    )

    reopened = SemanticCache(path, encoder=_FakeEncoder(), threshold=0.88)

    assert reopened.get("how do i use depends?", corpus_version="v1").text == "persisted"


def test_threshold_is_configurable_per_instance(tmp_path):
    strict_cache = SemanticCache(tmp_path / "cache.sqlite", encoder=_FakeEncoder(), threshold=0.999)
    strict_cache.set("how do i use depends?", corpus_version="v1", answer=_answer())

    # cos ~= 0.99 < 0.999 -- a stricter threshold should reject what the
    # default (0.88) test above accepted.
    assert strict_cache.get("how can i use depends in fastapi?", corpus_version="v1") is None


def test_shipped_default_threshold_does_not_serve_the_real_hardest_near_miss(tmp_path):
    # Real integration test, plan.md's own exit criterion ("test proving the
    # semantic cache doesn't serve near-misses") -- not the fake-vector unit
    # tests above, the actual shipped default (`CacheSettings.
    # semantic_similarity_threshold=0.88`) against the real hardest
    # near-miss pair this corpus produces (`docs/results/semantic-cache-
    # threshold.md`: 0.8694 real cosine similarity, bge-large, two genuinely
    # different real dev-positive golden queries).
    from aegis.config import settings
    from aegis.embedding.cache import EmbeddingCache
    from aegis.embedding.encoder import Encoder

    query_a = "Generated OpenAPI schema does not reflect pydantic dynamic models."
    query_b = "OpenAPI schema generation sometimes fails"

    cache_backing = EmbeddingCache(settings.paths.embedding_cache)
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=cache_backing, device="cpu")
    cache = SemanticCache(
        tmp_path / "cache.sqlite", encoder=encoder, threshold=settings.cache.semantic_similarity_threshold
    )

    cache.set(query_a, corpus_version="v1", answer=_answer("answer about pydantic dynamic models"))

    # A genuinely different real question, real-measured at 0.8694 similarity
    # -- below the shipped 0.88 threshold. Must NOT serve query_a's answer.
    assert cache.get(query_b, corpus_version="v1") is None
