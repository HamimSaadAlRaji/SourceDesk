import pytest

from aegis.ingest.markdown import clean_markdown, extract_title, resolve_admonitions, resolve_includes


def test_resolve_includes_reads_referenced_file(tmp_path):
    src = tmp_path / "code_examples" / "example.py"
    src.parent.mkdir()
    src.write_text("print('hello')\n")

    text = "Some text\n\n{* code_examples/example.py *}\n\nMore text"
    result = resolve_includes(text, tmp_path)

    assert "```python" in result
    assert "print('hello')" in result
    assert "{*" not in result


def test_resolve_includes_strips_highlight_annotation(tmp_path):
    src = tmp_path / "ex.py"
    src.write_text("x = 1\ny = 2\n")

    result = resolve_includes("{* ex.py hl[1,2] *}", tmp_path)

    assert "x = 1" in result
    assert "y = 2" in result


def test_resolve_includes_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_includes("{* nonexistent.py *}", tmp_path)


def test_resolve_includes_handles_snippet_syntax(tmp_path):
    src = tmp_path / "item.html"
    src.write_text("<p>{{ id }}</p>\n")

    result = resolve_includes("```jinja\n{!item.html!}\n```", tmp_path)

    assert "{!" not in result
    assert "<p>{{ id }}</p>" in result


def test_resolve_includes_handles_snippet_syntax_with_marker(tmp_path):
    src = tmp_path / "code.py"
    src.write_text("x = 1\n")

    result = resolve_includes("{!> code.py!}", tmp_path)

    assert "{!" not in result
    assert "x = 1" in result


def test_resolve_includes_snippet_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_includes("{!nonexistent.py!}", tmp_path)


def test_resolve_admonitions_note():
    text = "Intro.\n\n/// note\n\nThis is a note.\n\n///\n\nOutro."
    result = resolve_admonitions(text)

    assert "/// note" not in result
    assert "Note:" in result
    assert "This is a note." in result
    assert "Outro." in result


def test_resolve_admonitions_generic_type_label():
    result = resolve_admonitions("/// warning\n\nBe careful.\n\n///")

    assert "Warning:" in result
    assert "Be careful." in result


def test_resolve_admonitions_custom_title():
    result = resolve_admonitions("/// note | Heads up\n\nBody text.\n\n///")

    assert "Heads up:" in result
    assert "Body text." in result


def test_resolve_admonitions_tab_group():
    text = (
        "//// tab | Linux\n\nRun this on Linux.\n\n////\n\n"
        "//// tab | Windows\n\nRun this on Windows.\n\n////"
    )
    result = resolve_admonitions(text)

    assert "Linux" in result and "Run this on Linux." in result
    assert "Windows" in result and "Run this on Windows." in result
    assert "////" not in result


def test_resolve_admonitions_indented_inside_list_item():
    text = (
        "1. Do the thing.\n\n"
        "    /// note\n\n"
        "    Indented note body.\n\n"
        "    ///\n\n"
        "    Continue the list item."
    )
    result = resolve_admonitions(text)

    assert "/// note" not in result
    assert "Note:" in result
    assert "Indented note body." in result
    assert "Continue the list item." in result


def test_resolve_admonitions_nested():
    text = "//// tab | Bash\n\ncontent\n\n/// tip\n\nhelpful tip\n\n///\n\n////"
    result = resolve_admonitions(text)

    assert "Tip:" in result
    assert "helpful tip" in result
    assert "Bash" in result
    assert "/" * 4 not in result


def test_extract_title_strips_anchor():
    text = "# Query Parameters { #query-parameters }\n\nBody"
    assert extract_title(text) == "Query Parameters"


def test_extract_title_no_heading_returns_none():
    assert extract_title("just some text, no heading") is None


def test_clean_markdown_strips_termy_div_but_keeps_code(tmp_path):
    text = '<div class="termy">\n\n```console\n$ echo hi\n```\n\n</div>'
    result = clean_markdown(text, tmp_path)

    assert "<div" not in result
    assert "</div>" not in result
    assert "$ echo hi" in result


def test_clean_markdown_strips_screenshot_div_and_image(tmp_path):
    text = '<div class="screenshot">\n<img src="/img/x.png">\n</div>'
    result = clean_markdown(text, tmp_path)

    assert "<div" not in result
    assert "<img" not in result
