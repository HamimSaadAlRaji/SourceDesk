from aegis.eval.strata import classify_stratum
from aegis.types import Stratum


def test_version_specific_detects_upgrade_regression():
    assert (
        classify_stratum(
            title="Regression after upgrading to 0.100.0",
            body="Worked fine on fastapi==0.99.1, broke after upgrading to fastapi==0.100.0.",
            resolution="Fixed in 0.100.1, see changelog.",
        )
        == Stratum.VERSION_SPECIFIC
    )


def test_error_debugging_detects_traceback():
    assert (
        classify_stratum(
            title="500 error on startup",
            body="Traceback (most recent call last):\n  File app.py, line 3\nValueError: invalid config",
            resolution="You need to set the env var before importing.",
        )
        == Stratum.ERROR_DEBUGGING
    )


def test_error_debugging_detects_exception_name_without_traceback():
    assert (
        classify_stratum(
            title="Getting a ValidationError",
            body="I keep getting a pydantic.ValidationError when I submit a form.",
            resolution="Your model field is missing a default.",
        )
        == Stratum.ERROR_DEBUGGING
    )


def test_multi_hop_detects_two_distinct_product_areas():
    assert (
        classify_stratum(
            title="Using dependency injection together with websockets",
            body="I want to use Depends() inside a websocket route, combined with a background task after close.",
            resolution="You can inject dependencies into websocket routes the same way.",
        )
        == Stratum.MULTI_HOP
    )


def test_how_to_detects_question_phrasing():
    assert (
        classify_stratum(
            title="How do I return a custom response header?",
            body="What's the best way to add a custom header to every response?",
            resolution="Use a middleware or set response.headers directly.",
        )
        == Stratum.HOW_TO
    )


def test_conceptual_detects_why_question():
    assert (
        classify_stratum(
            title="Why does FastAPI validate query params before path params?",
            body="I don't understand the order dependencies are resolved in, can someone clarify?",
            resolution="Path operation parameters are resolved in declaration order.",
        )
        == Stratum.CONCEPTUAL
    )


def test_ambiguous_fallback_when_nothing_matches():
    assert (
        classify_stratum(
            title="Question",
            body="not sure what's going on here",
            resolution="Can you provide a minimal reproducible example?",
        )
        == Stratum.AMBIGUOUS
    )


def test_version_specific_takes_priority_over_error_debugging():
    # A version regression that also throws an exception is stratified as
    # version-specific, not error/debugging -- the more actionable signal
    # for Phase 6, which wants to know if the system handles version drift,
    # not just whether it can answer "I got an error".
    assert (
        classify_stratum(
            title="TypeError after upgrading to 0.101.0",
            body="Traceback (most recent call last):\nTypeError: unexpected keyword argument\nWorked on 0.100.0.",
            resolution="This was a breaking change documented in the 0.101.0 changelog.",
        )
        == Stratum.VERSION_SPECIFIC
    )


def test_code_blocks_stripped_before_multi_hop_area_counting():
    # A repro snippet incidentally touching multiple unrelated APIs
    # shouldn't manufacture a false multi-hop label -- same boilerplate-
    # pollution lesson as product_area.py's own code-block stripping.
    body = (
        "How do I set a cookie?\n\n"
        "```python\n"
        "def f(q: str, db: Session = Depends(get_db)):\n"
        "    return {'cors': True}\n"
        "```\n"
    )
    assert classify_stratum(title="Cookie question", body=body, resolution="Use Response.set_cookie().") == Stratum.HOW_TO


def test_resolution_text_contributes_version_signal():
    # The query alone doesn't always carry the version signal -- sometimes
    # only the maintainer's resolution names the version the bug was
    # specific to ("only affects 0.95, fixed in 0.96").
    assert (
        classify_stratum(
            title="Weird behavior with nested models",
            body="Nested Pydantic models don't validate correctly in some cases.",
            resolution="This only affects 0.95.0, already fixed in 0.96.0 after an upgrade to the internal serializer.",
        )
        == Stratum.VERSION_SPECIFIC
    )
