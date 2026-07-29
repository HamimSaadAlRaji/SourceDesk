from aegis.ingest.hashing import content_hash


def test_content_hash_deterministic():
    assert content_hash("hello") == content_hash("hello")


def test_content_hash_differs_for_different_text():
    assert content_hash("hello") != content_hash("world")


def test_content_hash_is_sha256_hex():
    digest = content_hash("hello")
    assert len(digest) == 64
    int(digest, 16)
