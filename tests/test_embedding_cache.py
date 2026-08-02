import numpy as np

from aegis.embedding.cache import EmbeddingCache


def test_set_then_get_roundtrips(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    cache.set("model-a", "hash1", vec)
    result = cache.get("model-a", "hash1")

    assert result is not None
    np.testing.assert_array_almost_equal(result, vec)


def test_get_missing_key_returns_none(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    assert cache.get("model-a", "nonexistent") is None


def test_same_hash_different_models_are_independent(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    vec_a = np.array([1.0, 0.0], dtype=np.float32)
    vec_b = np.array([0.0, 1.0], dtype=np.float32)

    cache.set("model-a", "hash1", vec_a)
    cache.set("model-b", "hash1", vec_b)

    np.testing.assert_array_almost_equal(cache.get("model-a", "hash1"), vec_a)
    np.testing.assert_array_almost_equal(cache.get("model-b", "hash1"), vec_b)


def test_set_many_and_get_many_bulk_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    items = {
        "hash1": np.array([1.0, 2.0], dtype=np.float32),
        "hash2": np.array([3.0, 4.0], dtype=np.float32),
    }
    cache.set_many("model-a", items)

    result = cache.get_many("model-a", ["hash1", "hash2", "hash3"])

    assert set(result.keys()) == {"hash1", "hash2"}
    np.testing.assert_array_almost_equal(result["hash1"], items["hash1"])
    np.testing.assert_array_almost_equal(result["hash2"], items["hash2"])


def test_persists_across_cache_instances(tmp_path):
    path = tmp_path / "cache.sqlite"
    vec = np.array([5.0, 6.0, 7.0], dtype=np.float32)

    cache1 = EmbeddingCache(path)
    cache1.set("model-a", "hash1", vec)
    cache1.close()

    cache2 = EmbeddingCache(path)
    result = cache2.get("model-a", "hash1")

    assert result is not None
    np.testing.assert_array_almost_equal(result, vec)


def test_overwriting_a_key_replaces_the_vector(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    cache.set("model-a", "hash1", np.array([1.0, 1.0], dtype=np.float32))
    cache.set("model-a", "hash1", np.array([2.0, 2.0], dtype=np.float32))

    result = cache.get("model-a", "hash1")
    np.testing.assert_array_almost_equal(result, np.array([2.0, 2.0]))
