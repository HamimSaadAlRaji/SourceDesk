from aegis.query.rewrite import QueryRewriter


def test_no_history_returns_query_unchanged_without_calling_rewrite_fn():
    calls = []

    def rewrite_fn(history, query):
        calls.append((history, query))
        return "should never be used"

    rewriter = QueryRewriter(rewrite_fn=rewrite_fn)
    result = rewriter.rewrite([], "how do I use dependency injection?")

    assert result == "how do I use dependency injection?"
    assert calls == []


def test_history_present_calls_rewrite_fn_with_history_and_query():
    def rewrite_fn(history, query):
        return "standalone rewritten query"

    rewriter = QueryRewriter(rewrite_fn=rewrite_fn)
    history = [("user", "I'm working with query parameters"), ("assistant", "Sure, what do you need?")]
    result = rewriter.rewrite(history, "how do I handle ones with hyphens?")

    assert result == "standalone rewritten query"


def test_empty_rewrite_falls_back_to_raw_query():
    def rewrite_fn(history, query):
        return "   "

    rewriter = QueryRewriter(rewrite_fn=rewrite_fn)
    result = rewriter.rewrite([("user", "context")], "what about that?")

    assert result == "what about that?"


def test_rewrite_output_is_stripped():
    def rewrite_fn(history, query):
        return "  standalone query  \n"

    rewriter = QueryRewriter(rewrite_fn=rewrite_fn)
    result = rewriter.rewrite([("user", "context")], "and that?")

    assert result == "standalone query"
