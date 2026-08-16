"""TDD for the routing policy (plan.md Phase 8, step 2): a hard-coded code
path, not a prompt instruction. The one test that matters most in this
whole phase is `test_billing_route_can_never_reach_generation` -- it proves
structurally, not by convention, that a billing/account/legal query cannot
reach the generation call.
"""

from aegis.router.classifier import Route
from aegis.router.policy import PolicyOutcome, decide, handle


def test_decide_escalates_billing_account_legal():
    outcome = decide(Route.BILLING_ACCOUNT_LEGAL)

    assert outcome.escalate is True
    assert outcome.route == Route.BILLING_ACCOUNT_LEGAL


def test_decide_does_not_escalate_docs_howto():
    outcome = decide(Route.DOCS_HOWTO)

    assert outcome.escalate is False


def test_decide_does_not_escalate_known_issue():
    outcome = decide(Route.KNOWN_ISSUE)

    assert outcome.escalate is False


def test_decide_returns_a_reason_string_for_every_route():
    for route in Route:
        outcome = decide(route)
        assert isinstance(outcome, PolicyOutcome)
        assert outcome.reason


def test_billing_route_can_never_reach_generation():
    generate_calls = []
    escalate_calls = []

    def generate_fn(query: str) -> str:
        generate_calls.append(query)
        return "a generated answer"

    def escalate_fn(query: str, outcome: PolicyOutcome) -> str:
        escalate_calls.append((query, outcome))
        return "escalated"

    result = handle("please refund my invoice", Route.BILLING_ACCOUNT_LEGAL, generate_fn, escalate_fn)

    assert generate_calls == []
    assert len(escalate_calls) == 1
    assert result == "escalated"


def test_non_billing_routes_reach_generation_not_escalation():
    generate_calls = []
    escalate_calls = []

    def generate_fn(query: str) -> str:
        generate_calls.append(query)
        return "a generated answer"

    def escalate_fn(query: str, outcome: PolicyOutcome) -> str:
        escalate_calls.append((query, outcome))
        return "escalated"

    for route in (Route.DOCS_HOWTO, Route.KNOWN_ISSUE):
        generate_calls.clear()
        escalate_calls.clear()
        result = handle("how do I use dependency injection", route, generate_fn, escalate_fn)

        assert escalate_calls == []
        assert generate_calls == ["how do I use dependency injection"]
        assert result == "a generated answer"


def test_generate_fn_that_raises_on_call_proves_billing_never_invokes_it():
    def generate_fn_that_must_never_run(query: str) -> str:
        raise AssertionError("generation must not be reachable for billing_account_legal")

    def escalate_fn(query: str, outcome: PolicyOutcome) -> str:
        return "escalated"

    result = handle("cancel my enterprise contract", Route.BILLING_ACCOUNT_LEGAL, generate_fn_that_must_never_run, escalate_fn)

    assert result == "escalated"
