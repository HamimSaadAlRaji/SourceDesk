"""Phase 14 build item 6, real-stack check (design record 14f): force each
provider-failure path with a deliberately invalid API key and record what the
real system does. Faked providers prove the branching in tests; this proves a
real SDK's real auth error takes the same branch.

Three cases, each a real failure from a real provider:
  A. failover   -- FallbackLLMClient([Gemini(invalid key), Groq(real key)]):
                   expect `llm.provider_failed` (gemini) + `llm.failover` (groq).
                   Gemini goes first as the broken one so the one real call
                   lands on Groq; the user's deferred judge-kappa run needs the
                   Gemini daily quota.
  B. router outage -- RouterClassifier built with an invalid Groq key:
                   expect escalation, cause=provider_unavailable, route=None,
                   no retrieval. Zero successful LLM calls.
  C. generation outage -- real router/retrieval/confidence, generation client
                   with both keys invalid: expect escalation after retrieval,
                   cause=provider_unavailable, nothing cached. One real router call.

Isolated throwaway caches, so case C can't be served from cache and leaves
nothing in the real ones. Raw output: docs/results/provider-failure-real-raw.json.

    uv run python experiments/14c_provider_failure_real.py
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path

from aegis.config import REPO_ROOT, settings
from aegis.factory import build_pipeline
from aegis.generation.llm.fallback import FallbackLLMClient
from aegis.generation.llm.gemini_provider import GeminiProvider
from aegis.generation.llm.groq_provider import GroqProvider
from aegis.observability.logging import JsonFormatter
from aegis.router.classifier import RouterClassifier
from aegis.types import Query

OUT = REPO_ROOT / "docs" / "results" / "provider-failure-real-raw.json"
INVALID_KEY = "invalid-key-for-failure-test"
# Answered (not escalated) in earlier real runs, so an escalation in case C is
# the generation outage, not low confidence.
QUERY = "how do I declare a query parameter"


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(JsonFormatter())
        self.records: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(json.loads(self.format(record)))


def _events(pipeline, text: str) -> tuple[list[dict], float]:
    start = time.perf_counter()
    events = [e.model_dump() for e in pipeline.answer_stream(Query(id=f"14c-{int(time.time())}", text=text))]
    return events, round((time.perf_counter() - start) * 1000.0, 1)


def main() -> None:
    logs = _Collect()
    root = logging.getLogger("aegis")
    root.addHandler(logs)
    root.setLevel(logging.INFO)
    root.propagate = False
    out: dict = {}

    # A. failover
    logs.records.clear()
    client = FallbackLLMClient([GeminiProvider(api_key=INVALID_KEY), GroqProvider()])
    start = time.perf_counter()
    text, provider = client.generate_with_provider(
        [{"role": "user", "content": "Reply with the single word: ok"}], temperature=0.0, top_p=1.0, max_tokens=5
    )
    out["A_failover"] = {
        "provider": provider, "text": text, "wall_ms": round((time.perf_counter() - start) * 1000.0, 1),
        "logs": [r for r in logs.records if r["event"].startswith("llm.")],
    }

    with tempfile.TemporaryDirectory() as tmp:
        pipeline = build_pipeline(cache_dir=Path(tmp))
        pipeline.warm_up()

        # B. router outage
        real_key = settings.groq.api_key
        settings.groq.api_key = INVALID_KEY
        try:
            broken_router = RouterClassifier()  # the default classify_fn reads the key at construction
        finally:
            settings.groq.api_key = real_key
        real_router, pipeline.router = pipeline.router, broken_router
        logs.records.clear()
        events, wall = _events(pipeline, QUERY)
        out["B_router_outage"] = {"events": events, "wall_ms": wall, "logs": logs.records[:]}
        pipeline.router = real_router

        # C. generation outage
        pipeline.generation.llm = FallbackLLMClient(
            [GroqProvider(api_key=INVALID_KEY), GeminiProvider(api_key=INVALID_KEY)]
        )
        logs.records.clear()
        events, wall = _events(pipeline, QUERY)
        out["C_generation_outage"] = {
            "events": events, "wall_ms": wall, "logs": logs.records[:],
            "cached_after": pipeline.exact_cache.get(QUERY, pipeline.corpus_version, partition=pipeline.partition) is not None,
        }

    OUT.write_text(json.dumps(out, indent=2, default=str))
    for case, r in out.items():
        print(f"== {case}  wall {r['wall_ms']} ms")
        for e in r.get("events", []):
            keep = {k: e.get(k) for k in ("type", "cause", "reason", "route", "after_retrieval", "escalate", "stage_latencies_ms") if k in e}
            print("  event", keep)
        if "provider" in r:
            print("  answered by", r["provider"], repr(r["text"]))
        if "cached_after" in r:
            print("  cached after:", r["cached_after"])
        for rec in r["logs"]:
            if rec["level"] in ("WARNING", "ERROR"):
                print("  log", rec["event"], {k: rec.get(k) for k in ("provider", "error_type", "status_code", "skipped", "cause", "elapsed_ms")})
    print("raw ->", OUT.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
