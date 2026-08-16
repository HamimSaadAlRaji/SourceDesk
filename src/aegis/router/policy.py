"""Routing policy (plan.md Phase 8, step 2): "billing_account_legal
escalates immediately and never reaches a generation call. A code path, not
a prompt instruction. Prompts can be talked out of things."

`decide()` is a pure function -- trivial to audit, no LLM call, no way for
a cleverly-worded query to talk it into a different outcome. `handle()` is
the enforcement point: it is the *only* place a route decision is turned
into either a generation call or an escalation, and it structurally cannot
call both. `tests/test_router_policy.py::
test_generate_fn_that_raises_on_call_proves_billing_never_invokes_it` proves
this by passing a `generate_fn` that raises if it's ever invoked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from aegis.router.classifier import Route


@dataclass(frozen=True)
class PolicyOutcome:
    route: Route
    escalate: bool
    reason: str


def decide(route: Route) -> PolicyOutcome:
    if route == Route.BILLING_ACCOUNT_LEGAL:
        return PolicyOutcome(
            route=route,
            escalate=True,
            reason="billing_account_legal always hard-escalates to a human, regardless of query phrasing",
        )
    return PolicyOutcome(
        route=route,
        escalate=False,
        reason=f"{route.value} continues to query understanding + retrieval",
    )


GenerateFn = Callable[[str], str]
EscalateFn = Callable[[str, PolicyOutcome], str]


def handle(query: str, route: Route, generate_fn: GenerateFn, escalate_fn: EscalateFn) -> str:
    outcome = decide(route)
    if outcome.escalate:
        return escalate_fn(query, outcome)
    return generate_fn(query)
