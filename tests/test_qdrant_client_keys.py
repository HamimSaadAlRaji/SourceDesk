"""Phase 15 unit 15f (design record 15f): which Qdrant key each client sends.
The served pipeline must get the read-only key, so even a compromised API
process can't write to or delete the index."""

from __future__ import annotations

import pytest

from aegis.config import settings
from aegis.embedding import vector_store


@pytest.fixture
def captured(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(vector_store, "QdrantClient", lambda **kw: calls.append(kw) or kw)
    return calls


def test_admin_client_sends_the_admin_key(monkeypatch, captured):
    monkeypatch.setattr(settings.qdrant, "api_key", "admin-k")
    monkeypatch.setattr(settings.qdrant, "read_only_api_key", "ro-k")
    vector_store.get_client()
    assert captured[-1]["api_key"] == "admin-k"


def test_read_only_client_sends_the_read_only_key(monkeypatch, captured):
    monkeypatch.setattr(settings.qdrant, "api_key", "admin-k")
    monkeypatch.setattr(settings.qdrant, "read_only_api_key", "ro-k")
    vector_store.get_client(read_only=True)
    assert captured[-1]["api_key"] == "ro-k"


def test_read_only_client_never_falls_back_to_the_admin_key(monkeypatch, captured):
    """Silently serving with write rights would undo the point of the split."""
    monkeypatch.setattr(settings.qdrant, "api_key", "admin-k")
    monkeypatch.setattr(settings.qdrant, "read_only_api_key", "")
    with pytest.raises(RuntimeError, match="read-only"):
        vector_store.get_client(read_only=True)


def test_no_keys_configured_sends_none(monkeypatch, captured):
    monkeypatch.setattr(settings.qdrant, "api_key", "")
    monkeypatch.setattr(settings.qdrant, "read_only_api_key", "")
    vector_store.get_client(read_only=True)
    vector_store.get_client()
    assert [c["api_key"] for c in captured[-2:]] == [None, None]
