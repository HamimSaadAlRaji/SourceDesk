"""Caller identity for the HTTP surface (Phase 15, design record 15c).

The user's decision: chat stays public (a public demo can't hide a key), every
browser gets a server-issued ownership token, and `/trace` needs an API key.

Two kinds of principal:
- **session**: anyone. The token is a random 256-bit value in an HttpOnly
  cookie; the owner id is a hash of it, so the store never holds a usable
  token. It answers one question -- "did this browser ask that?" -- which is
  all `/feedback` needs. Minting a new one costs nothing and buys nothing:
  it owns no traces, and rate limits stay keyed on the socket peer.
- **key**: `Authorization: Bearer <key>`, looked up by sha256 against
  `ApiSettings.api_keys`. A key that doesn't match is a 401, never a silent
  fall back to anonymous: a misconfigured operator should find out.

    uv run python -m aegis.api.auth new-key --name owner --partition full --trace
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import secrets
from dataclasses import dataclass
from typing import Literal

from aegis.config import ApiKeyGrant

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")  # secrets.token_urlsafe(32)


class Unauthorized(Exception):
    """Credentials were presented and are not valid."""


@dataclass(frozen=True)
class Principal:
    kind: Literal["session", "key"]
    owner: str
    partition: str
    can_trace: bool


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def _session_owner(token: str) -> str:
    return "session:" + hashlib.sha256(token.encode()).hexdigest()[:32]


def resolve_principal(
    authorization: str | None,
    session_cookie: str | None,
    grants: dict[str, ApiKeyGrant],
    anonymous_partition: str,
) -> tuple[Principal, str | None]:
    """`(principal, token_to_set)`; `token_to_set` is a new session token when
    the caller had no valid one, else None."""
    if authorization is not None:
        scheme, _, key = authorization.partition(" ")
        if scheme != "Bearer" or not key:
            raise Unauthorized("expected 'Authorization: Bearer <key>'")
        presented = hash_key(key)
        grant = next((g for h, g in grants.items() if hmac.compare_digest(h, presented)), None)
        if grant is None:
            raise Unauthorized("unknown API key")
        return Principal("key", f"key:{grant.name}", grant.partition, grant.can_trace), None

    if session_cookie is not None and _TOKEN_RE.fullmatch(session_cookie):
        return Principal("session", _session_owner(session_cookie), anonymous_partition, False), None
    token = new_session_token()
    return Principal("session", _session_owner(token), anonymous_partition, False), token


def _main() -> None:
    parser = argparse.ArgumentParser(description="Mint an API key and print its config entry.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    new = sub.add_parser("new-key")
    new.add_argument("--name", required=True)
    new.add_argument("--partition", required=True)
    new.add_argument("--trace", action="store_true", help="may call /trace")
    args = parser.parse_args()
    key = secrets.token_urlsafe(32)
    entry = {hash_key(key): {"name": args.name, "partition": args.partition, "can_trace": args.trace}}
    print(f"key (shown once, keep it): {key}")
    print(f"add to API_API_KEYS (JSON, merge with existing): {json.dumps(entry)}")


if __name__ == "__main__":
    _main()
