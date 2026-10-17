"""Typed events a request emits (plan.md Phase 14 step 2): "typed events let
the UI render citations and escalation as first-class states instead of
parsing them out of text."

Contract (tests/test_pipeline_stream.py):

  answer:      token+ , citation* , done
  cache hit:   token  , citation* , done
  escalation:  escalation , done            (nothing was generated)
  failure:     token* , error               (terminal; never followed by done)

`error` can follow `token`s already sent: a provider failure after the first
token propagates by design (generation/llm/fallback.py never splices a second
provider onto a partial answer), so a client must be able to receive partial
text and then an error.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    text: str


class CitationEvent(BaseModel):
    type: Literal["citation"] = "citation"
    citation_id: int
    chunk_id: str
    doc_id: str
    heading_path: list[str] = []
    text: str


EscalationCause = Literal["policy", "low_confidence", "provider_unavailable"]


class EscalationEvent(BaseModel):
    type: Literal["escalation"] = "escalation"
    # Why a person is taking over, as a closed set the UI can explain without
    # parsing `reason` (which is free text for the trace and the human agent).
    cause: EscalationCause
    reason: str
    message: str
    route: str | None = None
    after_retrieval: bool


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    trace_id: str
    cache_status: str
    route: str | None = None
    escalate: bool
    citations: list[str] = []
    provider: str | None = None
    prompt_version: str | None = None
    stage_latencies_ms: dict[str, float] = {}
    # What the question was resolved to when it was a follow-up (None otherwise),
    # so the UI can show the user what was actually searched for.
    rewritten_query: str | None = None


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str
    trace_id: str | None = None


PipelineEvent = Annotated[
    Union[TokenEvent, CitationEvent, EscalationEvent, DoneEvent, ErrorEvent], Field(discriminator="type")
]
