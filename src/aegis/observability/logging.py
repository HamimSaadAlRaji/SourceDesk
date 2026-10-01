"""Structured logging (plan.md Phase 13 build item 2): one JSON object per
line, with the request's trace ID attached to every record automatically.

stdlib `logging` + a context variable rather than structlog/loguru: the whole
requirement is "JSON lines with a trace_id", and CLAUDE.md SS2 says a new
dependency needs a reason -- there isn't one here. The trace ID is the same
Langfuse trace ID `AnswerPipeline` puts on `PipelineResult`, so a log line and
its trace can be joined on it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

_trace_id: ContextVar[str | None] = ContextVar("aegis_trace_id", default=None)
_FIELDS_ATTR = "aegis_fields"


@contextmanager
def bind_trace_id(trace_id: str) -> Iterator[None]:
    token = _trace_id.set(trace_id)
    try:
        yield
    finally:
        _trace_id.reset(token)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"aegis.{name}")


def log_event(logger: logging.Logger, level: int, event: str, **fields: object) -> None:
    logger.log(level, event, extra={_FIELDS_ATTR: fields})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        trace_id = _trace_id.get()
        if trace_id is not None:
            payload["trace_id"] = trace_id
        payload.update(getattr(record, _FIELDS_ATTR, {}))
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Attach the JSON formatter to the `aegis` logger tree (idempotent)."""
    root = logging.getLogger("aegis")
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
