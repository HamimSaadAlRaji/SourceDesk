"""Phase 13 Unit 1 -- prove real, live Langfuse Cloud connectivity end-to-end
before any real pipeline stage gets instrumented.

Not a pytest test (CLAUDE.md SS6: I/O-bound components get integration tests,
and this makes one real network call to a third-party service) -- run
directly and read the real trace back from the Langfuse Cloud API, the same
verification discipline as every other experiments/ script in this project.

Requires `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` in `.env` (manual step,
CLAUDE.md SS10's Kaggle-handoff convention applies here too -- account
creation on a third-party service cannot happen from this session).
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone

from aegis.config import settings
from aegis.observability.tracing import get_tracer


def main() -> int:
    if not settings.langfuse.public_key or not settings.langfuse.secret_key:
        print(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set in .env -- "
            "sign up at https://cloud.langfuse.com (free tier), create a "
            "project, and copy its API keys into .env before running this."
        )
        return 1

    client = get_tracer()

    if not client.auth_check():
        print("auth_check() failed -- real API keys were provided but rejected by Langfuse Cloud.")
        return 1
    print("auth_check() passed -- credentials are valid.")

    trace_id = client.create_trace_id()
    with client.start_as_current_observation(
        trace_context={"trace_id": trace_id},
        name="phase13-smoke-test",
        as_type="span",
        input={"query": "does the qdrant client support hybrid search?"},
    ) as root:
        with client.start_as_current_observation(
            name="retrieval", as_type="retriever", input={"top_k": 10}
        ) as retrieval_span:
            retrieval_span.update(output={"chunks_returned": 10, "top_score": 0.83})
        with client.start_as_current_observation(
            name="generation",
            as_type="generation",
            model="qwen/qwen3.8-27b",
            input={"prompt_version": "v2"},
        ) as gen_span:
            gen_span.update(output={"answer": "yes, via named vectors"}, usage_details={"output": 12})
        root.update(output={"cache_status": "miss"})

    client.flush()

    trace_url = client.get_trace_url(trace_id=trace_id)
    print(f"trace_id: {trace_id}")
    print(f"trace_url: {trace_url}")

    # Langfuse Cloud ingests async server-side -- give it a moment before
    # reading back, then confirm the 2 nested observations really landed
    # (not just that the SDK accepted them locally). `client.api.trace.get`
    # is the legacy single-trace endpoint -- 410 on this org (created after
    # 2026-09-16, already migrated off Langfuse v3's legacy API). Read back
    # via the current `observations.get_many` (v2) instead, per the 410's
    # own migration pointer.
    time.sleep(8)
    now = datetime.now(timezone.utc)
    fetched = client.api.observations.get_many(
        trace_id=trace_id,
        from_start_time=now - timedelta(minutes=5),
        to_start_time=now + timedelta(minutes=1),
    )
    names = sorted(o.name for o in fetched.data)
    print(f"fetched observation names: {names}")
    if names != ["generation", "phase13-smoke-test", "retrieval"]:
        print("Real observations were fetched but did not match what was sent -- investigate before trusting tracing.")
        return 1

    print("Real trace + 2 nested observations confirmed round-trip via the Langfuse Cloud API.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
