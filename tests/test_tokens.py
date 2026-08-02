from aegis.chunking.tokens import count_tokens, decode_tokens, encode_tokens, nearest_safe_boundary


def test_count_tokens_matches_encode_length():
    text = "FastAPI uses Pydantic for data validation."
    assert count_tokens(text) == len(encode_tokens(text))


def test_count_tokens_empty_string_is_zero():
    assert count_tokens("") == 0


def test_encode_decode_roundtrip():
    text = "app.include_router(users.router, prefix=\"/users\")"
    assert decode_tokens(encode_tokens(text)) == text


def test_count_tokens_grows_with_repetition():
    short = count_tokens("dependency injection")
    long = count_tokens("dependency injection " * 20)
    assert long > short


def test_nearest_safe_boundary_never_splits_multibyte_character():
    # "🎉" (U+1F389) is 4 UTF-8 bytes split across 3 cl100k_base tokens —
    # confirmed empirically: decode(tokens[:18]) and decode(tokens[18:])
    # each end/start with a U+FFFD replacement character.
    text = "Congratulations! You created your first API. 🎉 Keep going with FastAPI."
    tokens = encode_tokens(text)

    for candidate in range(len(tokens) + 1):
        safe = nearest_safe_boundary(tokens, candidate)
        left = decode_tokens(tokens[:safe])
        right = decode_tokens(tokens[safe:])
        assert "�" not in left
        assert "�" not in right
        assert left + right == text


def test_nearest_safe_boundary_is_a_no_op_on_ascii_text():
    tokens = encode_tokens("plain ascii text with no special characters")
    for candidate in range(len(tokens) + 1):
        assert nearest_safe_boundary(tokens, candidate) == candidate
