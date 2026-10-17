"""HTTP surface (plan.md Phase 14 step 1). `create_app(pipeline)` takes the
pipeline as an argument so tests inject a fake and `serve.py` injects the real
one; the module itself imports no model code.

SSE over a plain `StreamingResponse`, not an SSE library: the wire format is
`event: <type>\\ndata: <json>\\n\\n`, ~3 lines of code, and the event contract is
already typed in `aegis.events`. Design record 14b has the SSE-vs-WebSocket and
no-sse-starlette reasoning.
"""

from __future__ import annotations

import logging
import math
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Iterator, Protocol

from fastapi import FastAPI, HTTPException, Path, Request
from pathlib import Path as FsPath

from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from typing import Any, Literal

from aegis.config import settings
from aegis.api.ratelimit import TokenBucketLimiter
from aegis.api.store import InteractionStore
from aegis.events import DoneEvent, ErrorEvent, EscalationEvent, PipelineEvent, TokenEvent
from aegis.observability.logging import get_logger, log_event
from aegis.types import Query

_log = get_logger("api")


class _Pipeline(Protocol):
    def warm_up(self) -> dict: ...
    def answer_stream(self, query: Query) -> Iterator[PipelineEvent]: ...


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=settings.api.max_query_chars)


class ChatRequest(BaseModel):
    query: str = Field(max_length=settings.api.max_query_chars)
    # Client-held conversation: the server stays stateless. Only this caller's
    # own rewriter input depends on it, so a forged history can at worst
    # mis-rewrite that caller's own question (design record 14d).
    history: list[Turn] = Field(default_factory=list, max_length=settings.api.max_history_messages)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


_TRACE_ID = r"^[0-9a-f]{32}$"  # what Langfuse's create_trace_id() produces


class FeedbackRequest(BaseModel):
    trace_id: str = Field(pattern=_TRACE_ID)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=settings.api.max_comment_chars)


def _history_for_rewriter(turns: list[Turn]) -> list[tuple[str, str]] | None:
    window = turns[-settings.api.history_window_messages :]
    return [(t.role, t.text[: settings.api.history_message_chars]) for t in window] or None


def sse_frame(event: PipelineEvent) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


def create_app(
    pipeline: _Pipeline | None = None,
    store: InteractionStore | None = None,
    audit: Callable[[str], dict[str, Any] | None] | None = None,
    limiter: TokenBucketLimiter | None = None,
) -> FastAPI:
    # `is None`, not `or`: the limiter defines __len__, so an empty one is falsy.
    chat_limiter = limiter if limiter is not None else TokenBucketLimiter(
        settings.api.chat_rate_capacity, settings.api.chat_rate_per_minute, settings.api.rate_limit_max_clients
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is None:
            from aegis.factory import build_pipeline  # heavy imports only when serving for real

            app.state.pipeline = build_pipeline()
        else:
            app.state.pipeline = pipeline
        warm = app.state.pipeline.warm_up()
        log_event(_log, logging.INFO, "api.warm_up", stage_latencies_ms={k: round(v, 1) for k, v in warm.items()})
        app.state.store = store or InteractionStore(settings.api.interactions_db_path)
        app.state.audit = audit  # None -> built on first /trace call, so startup doesn't load the corpus twice
        yield

    app = FastAPI(title="Aegis", lifespan=lifespan)

    static_dir = FsPath(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/chat")
    def chat(req: ChatRequest, request: Request) -> StreamingResponse:
        # After body validation (FastAPI rejects a bad body before this runs),
        # so a malformed request costs the client no tokens.
        client = request.client.host if request.client else "unknown"
        allowed, retry_after = chat_limiter.acquire(client)
        if not allowed:
            log_event(_log, logging.WARNING, "api.rate_limited", client=client, retry_after_s=retry_after)
            raise HTTPException(
                status_code=429, detail="rate limit exceeded",
                headers={"Retry-After": str(math.ceil(retry_after))},
            )
        query = Query(id=uuid.uuid4().hex, text=req.query)

        def frames() -> Iterator[str]:
            answer: list[str] = []
            try:
                for event in app.state.pipeline.answer_stream(query, history=_history_for_rewriter(req.history)):
                    if isinstance(event, TokenEvent):
                        answer.append(event.text)
                    elif isinstance(event, EscalationEvent):
                        answer.append(event.message)
                    elif isinstance(event, DoneEvent):
                        app.state.store.record_interaction(
                            event.trace_id, query.text, "".join(answer),
                            escalated=event.escalate, cache_status=event.cache_status,
                            rewritten_query=event.rewritten_query,
                        )
                    yield sse_frame(event)
            except Exception as exc:  # noqa: BLE001 -- the stream must end with a typed event, not a dropped socket
                log_event(_log, logging.ERROR, "api.stream_error", query_id=query.id, error_type=type(exc).__name__)
                yield sse_frame(ErrorEvent(code=type(exc).__name__, message="The request could not be completed."))

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/feedback")
    def feedback(req: FeedbackRequest) -> dict:
        if not app.state.store.add_feedback(req.trace_id, req.rating, req.comment):
            raise HTTPException(status_code=404, detail="unknown trace_id")
        return {"status": "recorded"}

    @app.get("/trace/{trace_id}")
    def trace(trace_id: str = Path(pattern=_TRACE_ID)) -> dict:
        if app.state.audit is None:
            from aegis.api.audit_service import make_audit_service
            from aegis.factory import load_chunks

            app.state.audit = make_audit_service(load_chunks())
        result = app.state.audit(trace_id)
        if result is None:
            raise HTTPException(status_code=404, detail="trace not found (or not yet ingested)")
        return result

    return app
