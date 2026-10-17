"""`GET /trace/{id}` backend: Langfuse trace -> AuditTrail + resolved evidence
(observability/audit.py does the work; this only wires it to real I/O)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aegis.config import settings
from aegis.observability.audit import ROOT, build_audit_trail, fetch_observations, resolve_evidence
from aegis.observability.tracing import get_tracer
from aegis.types import Chunk


def make_audit_service(chunks: list[Chunk]) -> Callable[[str], dict[str, Any] | None]:
    by_id = {c.id: c for c in chunks}

    def audit(trace_id: str) -> dict[str, Any] | None:
        obs = fetch_observations(get_tracer(), trace_id, max_wait_s=settings.api.trace_max_wait_s)
        if not any(o["name"] == ROOT for o in obs):
            return None
        trail = build_audit_trail(obs, trace_id=trace_id)
        evidence = resolve_evidence(trail, by_id.get)
        return {"trail": trail.model_dump(), "evidence": [e.model_dump() for e in evidence]}

    return audit
