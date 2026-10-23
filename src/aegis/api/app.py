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
import threading
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path as FsPath
from typing import Any, Iterator, Literal, Mapping, Protocol

from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from aegis.api.auth import Principal, Unauthorized, resolve_principal
from aegis.api.ratelimit import TokenBucketLimiter
from aegis.api.store import InteractionStore
from aegis.config import settings
from aegis.events import DoneEvent, ErrorEvent, EscalationEvent, PipelineEvent, TokenEvent
from aegis.guardrails.input_boundary import injection_flags, redact
from aegis.observability.logging import get_logger, log_event
from aegis.types import Query

_log = get_logger("api")


class _Pipeline(Protocol):
    def warm_up(self) -> dict: ...
    def answer_stream(self, query: Query) -> Iterator[PipelineEvent]: ...


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    text: str = Field(max_length=settings.api.max_query_chars)


class ChatRequest(BaseModel):
    # Unknown fields are a 422, not silently dropped: the partition comes from
    # the server (design record 15b), and a body field like "partition" or
    # "doc_type" that looked accepted would invite exactly the wrong belief.
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(pattern=_TRACE_ID)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=settings.api.max_comment_chars)


def _history_for_rewriter(turns: list[Turn]) -> list[tuple[str, str]] | None:
    window = turns[-settings.api.history_window_messages :]
    return [(t.role, t.text[: settings.api.history_message_chars]) for t in window] or None


def sse_frame(event: PipelineEvent) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _limit(limiter: TokenBucketLimiter, request: Request, endpoint: str) -> None:
    client = _client_ip(request)
    allowed, retry_after = limiter.acquire(client)
    if not allowed:
        log_event(_log, logging.WARNING, "api.rate_limited", endpoint=endpoint, client=client, retry_after_s=retry_after)
        raise HTTPException(status_code=429, detail="rate limit exceeded",
                            headers={"Retry-After": str(math.ceil(retry_after))})


def create_app(
    pipeline: _Pipeline | None = None,
    store: InteractionStore | None = None,
    audit: Callable[[str], dict[str, Any] | None] | None = None,
    limiter: TokenBucketLimiter | None = None,
    pipelines: Mapping[str, _Pipeline] | None = None,
    feedback_limiter: TokenBucketLimiter | None = None,
) -> FastAPI:
    """`pipeline` serves the anonymous partition; `pipelines` maps partition ->
    pipeline when API keys use other partitions. With neither, the real ones are
    built for every partition the config references (design record 15c)."""
    # `is None`, not `or`: the limiter defines __len__, so an empty one is falsy.
    chat_limiter = limiter if limiter is not None else TokenBucketLimiter(
        settings.api.chat_rate_capacity, settings.api.chat_rate_per_minute, settings.api.rate_limit_max_clients
    )
    fb_limiter = feedback_limiter if feedback_limiter is not None else TokenBucketLimiter(
        settings.api.feedback_rate_capacity, settings.api.feedback_rate_per_minute, settings.api.rate_limit_max_clients
    )
    trace_slots = threading.BoundedSemaphore(settings.api.trace_max_concurrent)
    needed = {settings.api.anonymous_partition} | {g.partition for g in settings.api.api_keys.values()}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipelines is not None:
            app.state.pipelines = dict(pipelines)
        elif pipeline is not None:
            app.state.pipelines = {settings.api.anonymous_partition: pipeline}
        else:
            from aegis.factory import build_pipelines  # heavy imports only when serving for real

            app.state.pipelines = build_pipelines(sorted(needed))
        missing = needed - app.state.pipelines.keys()
        if missing:  # refuse to start rather than 500 on the first keyed request
            raise RuntimeError(f"no pipeline for configured partition(s) {sorted(missing)}")
        app.state.pipeline = app.state.pipelines[settings.api.anonymous_partition]
        for name, p in app.state.pipelines.items():
            warm = p.warm_up()
            log_event(_log, logging.INFO, "api.warm_up", partition=name,
                      stage_latencies_ms={k: round(v, 1) for k, v in warm.items()})
        app.state.store = store or InteractionStore(settings.api.interactions_db_path)
        app.state.audit = audit  # None -> built on first /trace call, so startup doesn't load the corpus twice
        yield

    app = FastAPI(title="Aegis", lifespan=lifespan)

    def principal_of(request: Request) -> tuple[Principal, str | None]:
        try:
            return resolve_principal(
                request.headers.get("authorization"), request.cookies.get(settings.api.session_cookie_name),
                settings.api.api_keys, settings.api.anonymous_partition,
            )
        except Unauthorized as exc:
            log_event(_log, logging.WARNING, "api.unauthorized", client=_client_ip(request), detail=str(exc))
            raise HTTPException(status_code=401, detail="invalid credentials",
                                headers={"WWW-Authenticate": "Bearer"}) from None

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
        # so a malformed request costs the client no tokens. The limiter keys on
        # the socket peer, not the principal: a fresh session token is free to
        # mint, so keying on it would hand out unlimited buckets.
        principal, new_token = principal_of(request)
        _limit(chat_limiter, request, "chat")
        served_by = app.state.pipelines[principal.partition]
        query_id = uuid.uuid4().hex
        # Input boundary (15d): redacted before the pipeline, so no cache, store,
        # Langfuse span or LLM provider downstream ever sees the raw secret.
        text, redacted = redact(req.query)
        turns = []
        for t in req.history:
            turn_text, turn_redacted = redact(t.text)
            turns.append(t.model_copy(update={"text": turn_text}))
            for k, n in turn_redacted.items():
                redacted[k] = redacted.get(k, 0) + n
        if redacted:
            log_event(_log, logging.INFO, "input.redacted", query_id=query_id, kinds=redacted)
        if flags := injection_flags(text):  # flag only, never blocks (15d)
            log_event(_log, logging.WARNING, "input.injection_flag", query_id=query_id, matched=flags[:5])
        query = Query(id=query_id, text=text)

        def frames() -> Iterator[str]:
            answer: list[str] = []
            try:
                for event in served_by.answer_stream(query, history=_history_for_rewriter(turns)):
                    if isinstance(event, TokenEvent):
                        answer.append(event.text)
                    elif isinstance(event, EscalationEvent):
                        answer.append(event.message)
                    elif isinstance(event, DoneEvent):
                        app.state.store.record_interaction(
                            event.trace_id, query.text, "".join(answer),
                            escalated=event.escalate, cache_status=event.cache_status,
                            owner=principal.owner, rewritten_query=event.rewritten_query,
                        )
                    yield sse_frame(event)
            except Exception as exc:  # noqa: BLE001 -- the stream must end with a typed event, not a dropped socket
                log_event(_log, logging.ERROR, "api.stream_error", query_id=query.id, error_type=type(exc).__name__)
                yield sse_frame(ErrorEvent(code=type(exc).__name__, message="The request could not be completed."))

        response = StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        if new_token is not None:
            # HttpOnly: page scripts (and any XSS) can't read it. SameSite=Strict:
            # other sites can't make the browser send it.
            response.set_cookie(
                settings.api.session_cookie_name, new_token, max_age=settings.api.session_cookie_max_age_s,
                httponly=True, samesite="strict", secure=settings.api.session_cookie_secure, path="/",
            )
        return response

    @app.post("/feedback")
    def feedback(req: FeedbackRequest, request: Request) -> dict:
        _limit(fb_limiter, request, "feedback")
        principal, _ = principal_of(request)
        # 404 both for "never issued" and "issued to someone else": the answer
        # must not reveal whether a trace id exists.
        comment = redact(req.comment)[0] if req.comment is not None else None
        if not app.state.store.add_feedback(req.trace_id, req.rating, comment, owner=principal.owner):
            raise HTTPException(status_code=404, detail="unknown trace_id")
        return {"status": "recorded"}

    @app.get("/trace/{trace_id}")
    def trace(request: Request, trace_id: str = Path(pattern=_TRACE_ID)) -> dict:
        # Operator-only (user decision, 15c): the audit trail holds a user's
        # query and the retrieved chunk text. Anonymous callers never reach the
        # worker-holding Langfuse poll at all.
        principal, _ = principal_of(request)
        if not principal.can_trace:
            if principal.kind == "session":
                raise HTTPException(status_code=401, detail="API key required", headers={"WWW-Authenticate": "Bearer"})
            raise HTTPException(status_code=403, detail="this key may not read traces")
        if not trace_slots.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="too many concurrent trace lookups",
                                headers={"Retry-After": "5"})
        try:
            if app.state.audit is None:
                from aegis.api.audit_service import make_audit_service
                from aegis.factory import load_chunks

                app.state.audit = make_audit_service(load_chunks())
            result = app.state.audit(trace_id)
        except Exception as exc:  # noqa: BLE001 -- Langfuse transport errors (a real ReadTimeout, 15c §5) are upstream, not ours
            log_event(_log, logging.ERROR, "api.trace_backend_error", error_type=type(exc).__name__)
            raise HTTPException(status_code=502, detail="trace backend unavailable") from None
        finally:
            trace_slots.release()
        if result is None:
            raise HTTPException(status_code=404, detail="trace not found (or not yet ingested)")
        return result

    return app
