"""TDD for the router classifier's parsing/fail-safe logic (plan.md Phase 8,
step 1). The live Groq call path is dependency-injected out via
`classify_fn` -- these tests never hit the network, same treatment as
`tests/test_cross_encoder.py` gives the reranker.
"""

import json

from aegis.router.classifier import Route, RouterClassifier


def _fixed(response: str):
    def classify_fn(query: str) -> str:
        return response

    return classify_fn


def test_parses_a_well_formed_docs_howto_response():
    classifier = RouterClassifier(classify_fn=_fixed(json.dumps({"route": "docs_howto"})))

    assert classifier.classify("how do I use dependency injection") == Route.DOCS_HOWTO


def test_parses_a_well_formed_known_issue_response():
    classifier = RouterClassifier(classify_fn=_fixed(json.dumps({"route": "known_issue"})))

    assert classifier.classify("I get a RecursionError with self-referencing models") == Route.KNOWN_ISSUE


def test_parses_a_well_formed_billing_account_legal_response():
    classifier = RouterClassifier(classify_fn=_fixed(json.dumps({"route": "billing_account_legal"})))

    assert classifier.classify("please refund my subscription") == Route.BILLING_ACCOUNT_LEGAL


def test_malformed_json_fails_safe_to_billing_account_legal():
    classifier = RouterClassifier(classify_fn=_fixed("not json at all"))

    assert classifier.classify("anything") == Route.BILLING_ACCOUNT_LEGAL


def test_missing_route_key_fails_safe_to_billing_account_legal():
    classifier = RouterClassifier(classify_fn=_fixed(json.dumps({"wrong_key": "docs_howto"})))

    assert classifier.classify("anything") == Route.BILLING_ACCOUNT_LEGAL


def test_out_of_vocabulary_route_fails_safe_to_billing_account_legal():
    classifier = RouterClassifier(classify_fn=_fixed(json.dumps({"route": "some_other_route"})))

    assert classifier.classify("anything") == Route.BILLING_ACCOUNT_LEGAL


def test_empty_response_fails_safe_to_billing_account_legal():
    classifier = RouterClassifier(classify_fn=_fixed(""))

    assert classifier.classify("anything") == Route.BILLING_ACCOUNT_LEGAL


def test_classify_fn_receives_the_raw_query_text():
    seen = []

    def classify_fn(query: str) -> str:
        seen.append(query)
        return json.dumps({"route": "docs_howto"})

    classifier = RouterClassifier(classify_fn=classify_fn)
    classifier.classify("what is the best way to validate query params?")

    assert seen == ["what is the best way to validate query params?"]
