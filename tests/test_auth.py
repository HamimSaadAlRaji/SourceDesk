"""TDD for caller identity (Phase 15 unit 15c, design record 15c).

A silent bug here hands one caller another's traces or votes, so the
resolution rules are pinned one by one: anonymous callers get a fresh
server-issued token, a well-formed token maps to a stable owner, a malformed
one is replaced (never trusted), a valid key maps to its grant, and an
invalid key is an error -- never a silent downgrade to anonymous.
"""

from __future__ import annotations

import pytest

from aegis.api.auth import Unauthorized, hash_key, new_session_token, resolve_principal
from aegis.config import ApiKeyGrant

KEY = "k" * 40
GRANTS = {hash_key(KEY): ApiKeyGrant(name="owner", partition="docs", can_trace=True)}


def test_no_credentials_issues_a_fresh_session_token():
    p, issued = resolve_principal(None, None, GRANTS, anonymous_partition="full")
    assert issued is not None and len(issued) >= 43
    assert p.kind == "session" and p.partition == "full" and p.can_trace is False


def test_tokens_are_random():
    assert len({new_session_token() for _ in range(100)}) == 100


def test_a_returning_session_maps_to_the_same_owner_and_is_not_reissued():
    token = new_session_token()
    a, issued_a = resolve_principal(None, token, GRANTS, anonymous_partition="full")
    b, issued_b = resolve_principal(None, token, GRANTS, anonymous_partition="full")
    assert a.owner == b.owner and issued_a is None and issued_b is None


def test_owner_id_does_not_contain_the_token():
    """The store keeps the owner id; a leaked DB must not hand out live tokens."""
    token = new_session_token()
    p, _ = resolve_principal(None, token, GRANTS, anonymous_partition="full")
    assert token not in p.owner


def test_different_sessions_are_different_owners():
    a, _ = resolve_principal(None, new_session_token(), GRANTS, anonymous_partition="full")
    b, _ = resolve_principal(None, new_session_token(), GRANTS, anonymous_partition="full")
    assert a.owner != b.owner


@pytest.mark.parametrize("bad", ["", "short", "x" * 43 + "!", "a b", "\x00" * 43, "y" * 200])
def test_malformed_cookie_is_replaced_not_trusted(bad):
    p, issued = resolve_principal(None, bad, GRANTS, anonymous_partition="full")
    assert issued is not None and issued != bad


def test_valid_key_maps_to_its_grant_and_ignores_the_cookie():
    p, issued = resolve_principal(f"Bearer {KEY}", new_session_token(), GRANTS, anonymous_partition="full")
    assert (p.kind, p.owner, p.partition, p.can_trace) == ("key", "key:owner", "docs", True)
    assert issued is None


@pytest.mark.parametrize("header", ["Bearer wrong", f"Basic {KEY}", "Bearer", f"Bearer  {KEY}x", KEY])
def test_invalid_key_is_an_error_not_a_downgrade(header):
    with pytest.raises(Unauthorized):
        resolve_principal(header, None, GRANTS, anonymous_partition="full")


def test_keys_are_configured_by_hash_only():
    assert KEY not in repr(GRANTS) and hash_key(KEY) == hash_key(KEY) and len(hash_key(KEY)) == 64
