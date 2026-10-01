"""LLM-based query router (plan.md Phase 8, step 1): classifies every
incoming query into `docs_howto`, `known_issue`, or `billing_account_legal`
*before* any retrieval happens (see the architecture diagram in plan.md §3
-- this is the very first stage after the cache).

Few-shot + JSON-mode structured output, on a small/fast model
(`RouterSettings.model`) -- classification into 3 labels doesn't need a
70B-parameter model, and paying that latency on every query would eat into
the 3s end-to-end budget for no accuracy this task needs.

I/O-bound wrapper over the Groq API -- dependency-injected via
`classify_fn` so the label-parsing/fail-safe logic is unit-testable with a
fake (`tests/test_classifier.py`), the same treatment as `DenseIndex`/
`CrossEncoderReranker`. The real Groq call path is exercised by
`experiments/08a_router_eval.py`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from enum import StrEnum

from aegis.config import settings
from aegis.observability.logging import get_logger, log_event


_log = get_logger("router")


class Route(StrEnum):
    DOCS_HOWTO = "docs_howto"
    KNOWN_ISSUE = "known_issue"
    BILLING_ACCOUNT_LEGAL = "billing_account_legal"


ClassifyFn = Callable[[str], str]

# Few-shot examples are hand-written and deliberately distinct from any
# golden-set or negative-set query text -- reusing eval text here would
# leak the test set into the prompt itself and invalidate the Phase 8
# router-eval confusion matrix (the exact bias auto_label.py's Phase 4
# docstring warns about avoiding in the other direction).
SYSTEM_PROMPT = """You are a routing classifier for a FastAPI support system. \
Classify the user's message into exactly one route:

- "docs_howto": a general question about how to use FastAPI, or a request \
to understand a FastAPI concept or feature. Answerable from documentation.
- "known_issue": describes a specific error, exception, crash, or unexpected \
behavior encountered while using FastAPI. Best answered by finding a similar \
resolved bug report.
- "billing_account_legal": anything about payment, invoices, refunds, \
subscriptions, account access, contracts, licensing-as-a-legal-question, \
compliance, or a legal/regulatory complaint -- regardless of how technical \
the surrounding language sounds. This route always hard-escalates to a \
human, so when genuinely unsure between this and a technical route, prefer \
this one.

Examples:
User: How do I add a custom exception handler in FastAPI?
{"route": "docs_howto"}

User: What's the difference between Depends and Security in FastAPI?
{"route": "docs_howto"}

User: My app throws a RecursionError when I add a Pydantic model with a self-referencing field. Is this a known bug?
{"route": "known_issue"}

User: Uvicorn workers keep crashing with a BrokenPipeError under load, and I found a similar closed issue -- is this the same root cause?
{"route": "known_issue"}

User: I was billed for two support seats but only use one, can someone on your team adjust our invoice?
{"route": "billing_account_legal"}

User: We need a signed data processing agreement from your company before our compliance team will approve using FastAPI internally.
{"route": "billing_account_legal"}

User: Our procurement team is asking whether your commercial support license terms are negotiable -- who handles that?
{"route": "billing_account_legal"}

Respond with JSON only, in the form {"route": "<one of the three labels above>"}. No other text."""


def _default_classify_fn(model: str, temperature: float, max_tokens: int) -> ClassifyFn:
    from groq import APIError, Groq

    client = Groq(api_key=settings.groq.api_key)

    def classify(query: str) -> str:
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
            )
        except APIError as exc:
            # Fail-safe behavior is unchanged, but it must not be invisible:
            # a Groq quota outage once turned every query into a silent
            # billing_account_legal escalation with no log line anywhere.
            log_event(
                _log, logging.WARNING, "router.api_error",
                error_type=type(exc).__name__, status_code=getattr(exc, "status_code", None),
                detail=str(exc)[:300],
            )
            # e.g. Groq's JSON-mode grammar constraint hitting
            # `json_validate_failed` before a valid document closes
            # (observed on the Phase 8 router eval). Returns "" so
            # `RouterClassifier.classify`'s existing parse-failure path
            # handles it the same as any other malformed output --
            # fail safe to billing_account_legal, not a second code path.
            return ""
        return completion.choices[0].message.content or ""

    return classify


class RouterClassifier:
    def __init__(
        self,
        classify_fn: ClassifyFn | None = None,
        model: str = settings.router.model,
        temperature: float = settings.router.temperature,
        max_tokens: int = settings.router.max_tokens,
    ) -> None:
        self._classify_fn = classify_fn or _default_classify_fn(model, temperature, max_tokens)

    def classify(self, query: str) -> Route:
        raw = self._classify_fn(query)
        try:
            parsed = json.loads(raw)
            return Route(parsed["route"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            if raw:  # "" means the API-error path already logged its own event
                log_event(_log, logging.WARNING, "router.unparseable_output", raw=raw[:200])
            # Fail safe, not fail open: unparseable/out-of-vocabulary model
            # output defaults to the hard-escalate route. Worst case is an
            # unnecessary human handoff; the alternative (silently treating
            # a possibly billing/legal query as routine) is the Air Canada
            # failure mode this whole project exists to prevent.
            return Route.BILLING_ACCOUNT_LEGAL
