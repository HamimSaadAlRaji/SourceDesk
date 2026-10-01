"""Langfuse client wiring.

Thin on purpose (CLAUDE.md SS2: Langfuse is a solved problem to import, not
rebuild) -- this module's only job is handing every later stage (routing,
retrieval, rerank, confidence, generation, validation, cache) the same
singleton client, configured once from `settings.langfuse`, instead of each
call site constructing its own.
"""

from __future__ import annotations

from functools import lru_cache

from langfuse import Langfuse

from aegis.config import settings


@lru_cache(maxsize=1)
def get_tracer() -> Langfuse:
    """The process-wide Langfuse client. Cached: the SDK owns a background
    flush thread and its own OTel tracer provider -- constructing more than
    one per process would mean two independent batches/threads for the same
    destination project."""
    return Langfuse(
        public_key=settings.langfuse.public_key,
        secret_key=settings.langfuse.secret_key,
        host=settings.langfuse.base_url,
        environment=settings.langfuse.environment,
    )
