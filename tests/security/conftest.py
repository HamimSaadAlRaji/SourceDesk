"""Shared helpers for the Phase 15 attack suite.

Convention for every file in this directory: each test asserts the *secure*
behavior. An attack that currently succeeds is marked
`xfail(strict=True, reason="ATTACK SUCCEEDS (before): ...")`, so the suite
stays green while the vulnerability is documented, and the moment a defense
lands the test XPASSes, which strict mode turns into a failure -- forcing the
marker to be removed and the before/after to be recorded in the design record.
"""

from __future__ import annotations

import pytest

from aegis.config import settings


def attack_succeeds(reason: str):
    return pytest.mark.xfail(strict=True, reason=f"ATTACK SUCCEEDS (before): {reason}")


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.api, "interactions_db_path", tmp_path / "default.sqlite")
