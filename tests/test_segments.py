from aegis.chunking.segments import split_code_segments


def test_plain_text_is_one_text_segment():
    segments = split_code_segments("Just prose, no code here.")
    assert [(s.kind, s.content) for s in segments] == [("text", "Just prose, no code here.")]


def test_single_fenced_block_isolated():
    text = "Before.\n\n```python\nx = 1\n```\n\nAfter."
    segments = split_code_segments(text)

    kinds = [s.kind for s in segments]
    assert kinds == ["text", "code", "text"]
    assert segments[1].content == "```python\nx = 1\n```"
    assert "Before." in segments[0].content
    assert "After." in segments[2].content


def test_code_segment_never_split_even_if_it_contains_blank_lines():
    text = "```python\ndef f():\n\n    return 1\n```"
    segments = split_code_segments(text)

    assert len(segments) == 1
    assert segments[0].kind == "code"
    assert "def f():" in segments[0].content
    assert "return 1" in segments[0].content


def test_multiple_fenced_blocks_each_isolated():
    text = "Intro.\n\n```bash\n$ echo hi\n```\n\nMiddle.\n\n```python\ny = 2\n```\n\nEnd."
    segments = split_code_segments(text)

    code_segments = [s for s in segments if s.kind == "code"]
    assert len(code_segments) == 2
    assert "$ echo hi" in code_segments[0].content
    assert "y = 2" in code_segments[1].content


def test_leading_and_trailing_whitespace_only_segments_dropped():
    text = "\n\n```python\nx = 1\n```\n\n"
    segments = split_code_segments(text)
    assert [s.kind for s in segments] == ["code"]


def test_unclosed_fence_treated_as_code_to_end_of_text():
    text = "Before.\n\n```python\nx = 1\ny = 2"
    segments = split_code_segments(text)

    assert segments[-1].kind == "code"
    assert "y = 2" in segments[-1].content


def test_closing_fence_followed_by_trailing_html_on_same_line():
    # Real corpus case (issue #9780): "```</div>" — the closing fence and a
    # wrapper </div> share one line with no newline between them. A pattern
    # that requires the closing fence line to contain *only* backticks (plus
    # trailing whitespace) misses this and swallows everything after it.
    text = "```json\n{}\n```</div>\n\n**Resolution:** fixed in 0.99.1"
    segments = split_code_segments(text)

    kinds = [s.kind for s in segments]
    assert kinds == ["code", "text"]
    assert "Resolution" in segments[1].content


def test_crlf_line_endings_do_not_break_fence_closing():
    # Real corpus case (issue #15000, pasted from a Windows user): the
    # closing fence line is "```\r\n". A closing-fence pattern that only
    # tolerates spaces/tabs before end-of-line misses the "\r" and falls
    # through to "unclosed fence", silently swallowing every heading and
    # paragraph after it into one giant code segment.
    text = "Before.\r\n\r\n```python\r\nx = 1\r\n```\r\n\r\n## Real Heading\r\n\r\nAfter."
    segments = split_code_segments(text)

    kinds = [s.kind for s in segments]
    assert kinds == ["text", "code", "text"]
    assert "## Real Heading" in segments[2].content
    assert "After." in segments[2].content
