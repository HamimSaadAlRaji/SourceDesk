"""TDD for the custom analyzer (plan.md Phase 5, step 1).

Standard tokenizers destroy exactly the tokens support queries depend on:
they lowercase-and-split `snake_case` into meaningless fragments, treat
`.` in `app.get` as a sentence boundary, and can't tell a version number
from three unrelated integers. Each rule below is tested against small
synthetic examples first, then the whole analyzer is checked against 40
literal tokens pulled from the real corpus (`data/processed/documents.jsonl`)
per the Phase 5 exit criterion — not invented examples.
"""

import pytest

from aegis.analysis.analyzer import analyze


def test_plain_words_are_lowercased_and_split_on_whitespace():
    assert analyze("Hello World") == ["hello", "world"]


def test_snake_case_retains_the_original_alongside_its_parts():
    tokens = analyze("set the response_model field")
    assert "response_model" in tokens
    assert "response" in tokens
    assert "model" in tokens


def test_snake_case_original_appears_before_its_parts():
    tokens = analyze("status_code")
    assert tokens == ["status_code", "status", "code"]


def test_dotted_path_retains_the_original_alongside_its_parts():
    tokens = analyze("call app.get(...) here")
    assert tokens == ["call", "app.get", "app", "get", "here"]


def test_deep_dotted_path_splits_every_segment():
    assert analyze("fastapi.tiangolo.com") == [
        "fastapi.tiangolo.com",
        "fastapi",
        "tiangolo",
        "com",
    ]


def test_screaming_case_constant_retains_original_alongside_parts():
    tokens = analyze("raise HTTP_401_UNAUTHORIZED")
    assert tokens == ["raise", "http_401_unauthorized", "http", "401", "unauthorized"]


def test_screaming_case_single_word_is_not_split():
    # No separator to split on -- must survive whole, not be shredded
    # into individual letters by a naive tokenizer.
    assert analyze("ECONNREFUSED") == ["econnrefused"]


def test_single_word_flag_retains_dashes_and_bare_word():
    assert analyze("run with --reload") == ["run", "with", "--reload", "reload"]


def test_multi_word_flag_retains_original_and_splits_into_words():
    assert analyze("--no-cache-dir") == ["--no-cache-dir", "no", "cache", "dir"]


def test_version_string_survives_as_one_token_undecomposed():
    # 0.100.0 must not become three meaningless integers ["0", "100", "0"].
    assert analyze("upgraded to 0.100.0") == ["upgraded", "to", "0.100.0"]


def test_short_version_string_survives_undecomposed():
    assert analyze("requires 3.12") == ["requires", "3.12"]


def test_version_at_end_of_sentence_drops_trailing_period():
    assert analyze("Now on 3.8.10.") == ["now", "on", "3.8.10"]


def test_ip_address_style_version_survives_undecomposed():
    assert analyze("bind to 127.0.0.1") == ["bind", "to", "127.0.0.1"]


def test_error_code_with_underscores_retains_original_and_parts():
    tokens = analyze("saw ERR_CERT_AUTHORITY_INVALID in the logs")
    assert "err_cert_authority_invalid" in tokens
    assert "err" in tokens
    assert "cert" in tokens
    assert "authority" in tokens
    assert "invalid" in tokens


def test_code_identifier_call_retains_original_and_parts():
    tokens = analyze("use jsonable_encoder(item)")
    assert "jsonable_encoder" in tokens
    assert "jsonable" in tokens
    assert "encoder" in tokens


def test_empty_string_produces_no_tokens():
    assert analyze("") == []


REAL_CORPUS_TOKENS = [
    # snake_case identifiers -- data/processed/documents.jsonl, top counts
    "status_code",
    "response_model",
    "item_id",
    "jsonable_encoder",
    "include_router",
    "current_user",
    "get_db",
    "hashed_password",
    # dotted paths / attribute access -- same source
    "app.get",
    "self.app",
    "app.post",
    "router.get",
    "main.py",
    "routing.py",
    "uvicorn.run",
    "openapi.json",
    # SCREAMING_CASE constants -- issue bodies in the same corpus
    "HTTP_401_UNAUTHORIZED",
    "SECRET_KEY",
    "HTTP_200_OK",
    "ACCESS_TOKEN_EXPIRE_MINUTES",
    "DATABASE_URL",
    "HTTP_404_NOT_FOUND",
    "API_KEY",
    "CONTENT_TYPE",
    # CLI flags -- uvicorn/gunicorn invocations in docs + issues
    "--reload",
    "--port",
    "--workers",
    "--host",
    "--root-path",
    "--no-cache-dir",
    "--proxy-headers",
    "--forwarded-allow-ips",
    # version strings and real error codes -- issue bodies
    "0.100.0",
    "3.12",
    "3.8.10",
    "127.0.0.1",
    "ECONNREFUSED",
    "ERR_CERT_AUTHORITY_INVALID",
    "ERR_FAILED",
    "0.78.0",
]

assert len(REAL_CORPUS_TOKENS) == 40


@pytest.mark.parametrize("token", REAL_CORPUS_TOKENS)
def test_real_corpus_token_survives_tokenization(token):
    """plan.md Phase 5 exit criterion: 'Analyzer tests green on 40 real
    corpus tokens.' Each token must appear, verbatim (case-folded), in the
    output when embedded in a realistic sentence -- i.e. tokenization must
    not silently destroy the exact identifier a support query would use."""
    sentence = f"See {token} for details in the FastAPI project."
    assert token.lower() in analyze(sentence)
