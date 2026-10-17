import numpy as np

from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder


class _CountingFakeEncoder:
    """Deterministic fake: each unique text maps to a fixed-seed random unit
    vector, and every call is logged so tests can assert caching actually
    avoided redundant work."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        vectors = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(size=self.dim)
            vectors.append(v / np.linalg.norm(v))
        return np.array(vectors, dtype=np.float32)

    @property
    def total_texts_encoded(self) -> int:
        return sum(len(c) for c in self.calls)


def test_encode_without_cache_returns_normalized_vectors():
    fake = _CountingFakeEncoder()
    encoder = Encoder(model_name="fake-model", encode_fn=fake)

    result = encoder.encode(["hello world", "fastapi docs"])

    assert result.shape == (2, 4)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-5)
    assert fake.total_texts_encoded == 2


def test_second_call_with_same_texts_hits_cache_entirely(tmp_path):
    fake = _CountingFakeEncoder()
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    encoder = Encoder(model_name="fake-model", encode_fn=fake, cache=cache)

    first = encoder.encode(["a", "b", "c"])
    assert fake.total_texts_encoded == 3

    second = encoder.encode(["a", "b", "c"])
    assert fake.total_texts_encoded == 3  # no new calls
    np.testing.assert_array_almost_equal(first, second)


def test_partial_cache_hit_only_encodes_new_texts(tmp_path):
    fake = _CountingFakeEncoder()
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    encoder = Encoder(model_name="fake-model", encode_fn=fake, cache=cache)

    encoder.encode(["a", "b"])
    assert fake.total_texts_encoded == 2

    encoder.encode(["a", "b", "c", "d"])
    assert fake.total_texts_encoded == 4  # only "c" and "d" are new
    assert fake.calls[-1] == ["c", "d"]


def test_result_order_matches_input_order_including_duplicates(tmp_path):
    fake = _CountingFakeEncoder()
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    encoder = Encoder(model_name="fake-model", encode_fn=fake, cache=cache)

    result = encoder.encode(["x", "y", "x", "z", "x"])

    assert result.shape == (5, 4)
    np.testing.assert_array_almost_equal(result[0], result[2])
    np.testing.assert_array_almost_equal(result[0], result[4])
    # "x" only ever encoded once despite appearing 3 times
    assert fake.total_texts_encoded == 3


def test_different_models_do_not_share_cache_entries(tmp_path):
    fake = _CountingFakeEncoder()
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    encoder_a = Encoder(model_name="model-a", encode_fn=fake, cache=cache)
    encoder_b = Encoder(model_name="model-b", encode_fn=fake, cache=cache)

    encoder_a.encode(["shared text"])
    assert fake.total_texts_encoded == 1

    encoder_b.encode(["shared text"])
    assert fake.total_texts_encoded == 2  # model-b has no cache entry yet


def test_empty_input_returns_empty_array_and_makes_no_calls():
    fake = _CountingFakeEncoder()
    encoder = Encoder(model_name="fake-model", encode_fn=fake)

    result = encoder.encode([])

    assert result.shape[0] == 0
    assert fake.calls == []


def test_no_cache_means_every_call_recomputes():
    fake = _CountingFakeEncoder()
    encoder = Encoder(model_name="fake-model", encode_fn=fake, cache=None)

    encoder.encode(["a", "b"])
    encoder.encode(["a", "b"])

    assert fake.total_texts_encoded == 4


def test_warm_up_runs_the_model_even_when_the_text_is_cached(tmp_path):
    # The pipeline's warm-up query was already in the on-disk embedding cache,
    # so "warming" via encode() never loaded the model: the first real query
    # then paid ~10.5 s inside cache_lookup (design record 14h).
    fake = _CountingFakeEncoder()
    encoder = Encoder(model_name="fake-model", encode_fn=fake, cache=EmbeddingCache(tmp_path / "cache.sqlite"))
    encoder.encode(["warm me"])

    encoder.warm_up("warm me")

    assert fake.calls == [["warm me"], ["warm me"]]


def test_warm_up_writes_nothing_to_the_embedding_cache(tmp_path):
    from aegis.ingest.hashing import content_hash

    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    encoder = Encoder(model_name="fake-model", encode_fn=_CountingFakeEncoder(), cache=cache)

    encoder.warm_up("a warm-up only text")

    assert cache.get_many("fake-model", [content_hash("a warm-up only text")]) == {}
