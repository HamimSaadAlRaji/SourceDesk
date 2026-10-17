"""Audit-trail reconstruction (plan.md Phase 13 build item 5): from ONE trace's
observations, rebuild what evidence produced an answer -- the record Air
Canada didn't have.

Three separable steps, so each is testable without a network:
  1. `fetch_observations`  -- the only I/O: pull a trace from Langfuse Cloud.
  2. `build_audit_trail`   -- pure: observations -> `AuditTrail`.
  3. `resolve_evidence`    -- pure: attach the actual chunk text, and detect
                              whether that text changed since the answer.

The reconstruction is only as good as what the pipeline recorded, so it says
so: missing stages, a missing context map ("what was [1]?"), and disagreement
between the per-stage spans and the root span's recorded result are reported
as `warnings` / `inconsistencies`, never papered over.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from aegis.types import Chunk

ROOT = "answer_query"
_ALL_STAGES = ["cache_lookup", "routing", "retrieval", "confidence", "generation", "validation"]


class RetrievedItem(BaseModel):
    chunk_id: str
    score: float


class AuditTrail(BaseModel):
    trace_id: str | None = None
    query_id: str | None = None
    query_text: str | None = None
    # Set when the request was a follow-up resolved against history: what the
    # cache, router and retriever actually saw instead of `query_text`.
    rewritten_query: str | None = None
    cache_status: str | None = None
    route: str | None = None
    escalated: bool = False
    escalation_reason: str | None = None
    retrieved: list[RetrievedItem] = []
    confidence: dict[str, Any] | None = None
    prompt_version: str | None = None
    generation_params: dict[str, Any] = {}
    context: list[dict[str, Any]] = []
    context_dropped_count: int | None = None
    context_total_tokens: int | None = None
    final_text: str | None = None
    provider: str | None = None
    citations: list[str] = []
    validation_claim_count: int | None = None
    stages_seen: list[str] = []
    stage_durations_ms: dict[str, float] = {}
    warnings: list[str] = []
    inconsistencies: list[str] = []


class EvidenceItem(BaseModel):
    citation_id: int
    chunk_id: str
    score: float | None
    cited: bool
    truncated_in_prompt: bool
    # ok: text matches what was shown; changed: chunk exists but its content
    # hash differs from the one recorded at answer time; missing: chunk gone;
    # unverified: the trace recorded no hash to compare against.
    status: Literal["ok", "changed", "missing", "unverified"]
    doc_id: str | None = None
    heading_path: list[str] = []
    text: str | None = None


def _dict(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _expected_stages(root_out: dict[str, Any]) -> list[str]:
    if root_out.get("cache_status", "miss") != "miss":
        return ["cache_lookup"]
    if root_out.get("escalate"):
        return ["cache_lookup", "routing", "escalation"] if root_out.get("confidence") is None else [
            "cache_lookup", "routing", "retrieval", "confidence", "escalation",
        ]
    return list(_ALL_STAGES)


def build_audit_trail(observations: list[dict[str, Any]], trace_id: str | None = None) -> AuditTrail:
    by_name = {o["name"]: o for o in observations}
    if ROOT not in by_name:
        raise ValueError(f"no '{ROOT}' root observation: not a pipeline trace, or ingestion is incomplete")

    root = by_name[ROOT]
    root_in, root_out = _dict(root.get("input")), _dict(root.get("output"))
    out = lambda name: _dict(by_name.get(name, {}).get("output"))  # noqa: E731
    inp = lambda name: _dict(by_name.get(name, {}).get("input"))  # noqa: E731

    trail = AuditTrail(
        trace_id=trace_id,
        query_id=root_in.get("query_id"),
        query_text=root_in.get("query"),
        rewritten_query=root_out.get("rewritten_query"),
        cache_status=out("cache_lookup").get("cache_status"),
        route=out("routing").get("route"),
        escalated=bool(out("escalation").get("escalated", False)),
        escalation_reason=out("escalation").get("reason"),
        retrieved=[RetrievedItem(chunk_id=s["chunk_id"], score=s["score"]) for s in out("retrieval").get("scores", [])],
        confidence=by_name["confidence"].get("output") if "confidence" in by_name else None,
        final_text=root_out.get("text"),
        provider=root_out.get("provider"),
        citations=list(out("validation").get("citations", [])),
        validation_claim_count=out("validation").get("claim_count"),
        stages_seen=[o["name"] for o in observations if o["name"] != ROOT],
    )

    gen_in = inp("generation")
    if gen_in:
        trail.prompt_version = gen_in.get("prompt_version")
        trail.generation_params = {k: gen_in[k] for k in ("temperature", "top_p", "max_tokens", "budget_tokens") if k in gen_in}
        trail.context = list(gen_in.get("context", []))
        trail.context_dropped_count = gen_in.get("context_dropped_count")
        trail.context_total_tokens = gen_in.get("context_total_tokens")

    for o in observations:
        latency = o.get("latency_ms")
        if o["name"] != ROOT and latency is not None:
            trail.stage_durations_ms[o["name"]] = latency

    expected = _expected_stages(root_out)
    for stage in expected:
        if stage not in by_name:
            trail.warnings.append(f"expected stage '{stage}' not found in trace (partial ingestion, or a trace from before this stage was recorded)")
    if "generation" in expected and "generation" in by_name and not trail.context:
        trail.warnings.append("generation span has no context map: cannot say which chunk each [n] in the answer refers to (trace predates context recording)")

    trail.inconsistencies = _inconsistencies(trail, root_out, out("generation"))
    return trail


def _inconsistencies(trail: AuditTrail, root_out: dict[str, Any], gen_out: dict[str, Any]) -> list[str]:
    """Per-stage spans vs the root span's recorded PipelineResult. They are
    written by the same code path, so disagreement means a partial/garbled
    trace or an edit -- worth surfacing either way."""
    found: list[str] = []

    def check(field: str, from_spans: Any, from_root: Any) -> None:
        if from_root is not None and from_spans != from_root:
            found.append(f"{field}: spans say {from_spans!r}, root result says {from_root!r}")

    check("cache_status", trail.cache_status, root_out.get("cache_status"))
    check("route", trail.route, root_out.get("route"))
    check("escalate", trail.escalated, root_out.get("escalate"))
    if gen_out:
        check("text", gen_out.get("text"), root_out.get("text"))
        check("citations", trail.citations, root_out.get("citations"))
        check("context chunk ids", [c["chunk_id"] for c in trail.context], [c["chunk_id"] for c in root_out.get("context", [])] if root_out.get("context") else None)
    root_retrieved = [r["chunk"]["id"] for r in root_out.get("retrieved", [])]
    if root_retrieved and trail.retrieved:
        check("retrieved chunk ids", [r.chunk_id for r in trail.retrieved], root_retrieved)
    return found


def resolve_evidence(trail: AuditTrail, lookup: Callable[[str], Chunk | None]) -> list[EvidenceItem]:
    cited = set(trail.citations)
    items: list[EvidenceItem] = []
    for entry in trail.context:
        chunk = lookup(entry["chunk_id"])
        recorded_hash = entry.get("content_hash")
        if chunk is None:
            status: Literal["ok", "changed", "missing", "unverified"] = "missing"
        elif recorded_hash is None:
            status = "unverified"
        else:
            status = "ok" if chunk.content_hash == recorded_hash else "changed"
        items.append(
            EvidenceItem(
                citation_id=entry["citation_id"], chunk_id=entry["chunk_id"], score=entry.get("score"),
                cited=entry["chunk_id"] in cited, truncated_in_prompt=bool(entry.get("truncated")), status=status,
                doc_id=chunk.doc_id if chunk else None, heading_path=chunk.heading_path if chunk else [],
                text=chunk.text if chunk else None,
            )
        )
    return items


def render_markdown(trail: AuditTrail, evidence: list[EvidenceItem], excerpt_chars: int = 500) -> str:
    L: list[str] = [f"# Audit trail — trace `{trail.trace_id or 'unknown'}`", ""]
    L += ["## Request", "", f"- Query (`{trail.query_id}`): {trail.query_text!r}", f"- Cache: **{trail.cache_status}**"]
    if trail.rewritten_query:
        L.insert(-1, f"- Resolved against conversation history to: {trail.rewritten_query!r}")
    if trail.route:
        L.append(f"- Route: **{trail.route}**")
    outcome = f"escalated to a human — {trail.escalation_reason}" if trail.escalated else "answered"
    L += [f"- Outcome: **{outcome}**", ""]

    if trail.retrieved:
        L += ["## Retrieved (before the token budget)", "", "| rank | chunk | score |", "|---|---|---|"]
        L += [f"| {i} | `{r.chunk_id}` | {r.score} |" for i, r in enumerate(trail.retrieved, 1)]
        L.append("")
    if trail.confidence:
        c = trail.confidence
        L += ["## Confidence", "", f"- Score: **{c.get('confidence'):.4f}**" if isinstance(c.get("confidence"), float) else f"- Score: {c.get('confidence')}"]
        L += [f"- {k}: raw {c['raw'][k]:.4f} → normalized {c['normalized'][k]:.4f}" for k in c.get("raw", {})] + [""]

    if trail.prompt_version:
        L += ["## What the model was shown", "", f"- Prompt version: `{trail.prompt_version}`; params: {trail.generation_params}"]
        L += [f"- Context: {len(trail.context)} chunks, {trail.context_total_tokens} tokens, {trail.context_dropped_count} dropped by the budget", ""]
        for e in evidence:
            flags = ["CITED" if e.cited else "not cited"] + (["truncated"] if e.truncated_in_prompt else []) + [f"source text: {e.status.upper()}"]
            L += [f"### [{e.citation_id}] `{e.chunk_id}` ({', '.join(flags)})", ""]
            if e.heading_path:
                L.append(f"_{' > '.join(e.heading_path)}_")
                L.append("")
            if e.text:
                shown = e.text[:excerpt_chars] + ("…" if len(e.text) > excerpt_chars else "")
                L += ["> " + line for line in shown.splitlines()] + [""]

    L += ["## Answer served", "", trail.final_text or "_(none recorded)_", ""]
    if trail.provider or trail.citations:
        L += [f"- Provider: {trail.provider}", f"- Validated citations: {trail.citations} ({trail.validation_claim_count} claims checked)", ""]
    if trail.stage_durations_ms:
        L += ["## Timings (span durations, ms)", ""] + [f"- {k}: {v:.0f}" for k, v in trail.stage_durations_ms.items()] + [""]
    if trail.warnings or trail.inconsistencies:
        L += ["## Integrity notes", ""] + [f"- ⚠ {w}" for w in trail.warnings] + [f"- ✗ inconsistency — {i}" for i in trail.inconsistencies] + [""]
    else:
        L += ["## Integrity notes", "", "- All expected stages present; per-stage spans agree with the recorded result.", ""]
    return "\n".join(L)


class _TraceClient(Protocol):
    api: Any


def fetch_observations(client: _TraceClient, trace_id: str, max_wait_s: float = 90.0, poll_s: float = 5.0) -> list[dict[str, Any]]:
    """Langfuse ingests asynchronously (8-14 s+ for a real trace, design record
    13b), so poll until the observation count stops growing and the root span
    is present. `input`/`output` are only returned with `fields=core,basic,io`
    and arrive as JSON strings."""
    deadline = time.monotonic() + max_wait_s
    prev = -1
    while True:
        now = datetime.now(timezone.utc)
        resp = client.api.observations.get_many(
            trace_id=trace_id, fields="core,basic,io", limit=100,
            from_start_time=now - timedelta(days=30), to_start_time=now + timedelta(minutes=5),
        )
        rows = resp.data
        if any(r.name == ROOT for r in rows) and len(rows) == prev:
            break
        prev = len(rows)
        if time.monotonic() > deadline:
            break
        time.sleep(poll_s)
    out = []
    for r in sorted(rows, key=lambda r: r.start_time):
        def parse(v: Any) -> Any:
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    return v
            return v
        out.append({
            "name": r.name, "type": r.type, "input": parse(getattr(r, "input", None)), "output": parse(getattr(r, "output", None)),
            "latency_ms": (r.latency * 1000.0) if getattr(r, "latency", None) is not None else None,
        })
    return out
