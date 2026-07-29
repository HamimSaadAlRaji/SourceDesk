from aegis.ingest.github_issues import is_maintainer_comment, issue_to_document, select_resolution_comment

BOT_COMMENT = {
    "id": 1,
    "user": {"login": "codspeed-hq[bot]"},
    "author_association": "NONE",
    "body": "performance report",
    "created_at": "2026-01-01T00:00:00Z",
}
CONTRIBUTOR_COMMENT = {
    "id": 2,
    "user": {"login": "someuser"},
    "author_association": "CONTRIBUTOR",
    "body": "I have the same issue",
    "created_at": "2026-01-02T00:00:00Z",
}
MAINTAINER_COMMENT_EARLY = {
    "id": 3,
    "user": {"login": "tiangolo"},
    "author_association": "OWNER",
    "body": "Can you share a repro?",
    "created_at": "2026-01-03T00:00:00Z",
}
MAINTAINER_COMMENT_LATE = {
    "id": 4,
    "user": {"login": "maintainer2"},
    "author_association": "MEMBER",
    "body": "This is fixed by using `pattern=` instead of `regex=`.",
    "created_at": "2026-01-04T00:00:00Z",
}


def test_is_maintainer_comment_true_for_owner_member_collaborator():
    assert is_maintainer_comment(MAINTAINER_COMMENT_EARLY) is True
    assert is_maintainer_comment(MAINTAINER_COMMENT_LATE) is True


def test_is_maintainer_comment_false_for_bot_and_contributor():
    assert is_maintainer_comment(BOT_COMMENT) is False
    assert is_maintainer_comment(CONTRIBUTOR_COMMENT) is False


def test_select_resolution_comment_picks_latest_maintainer_comment():
    comments = [BOT_COMMENT, CONTRIBUTOR_COMMENT, MAINTAINER_COMMENT_EARLY, MAINTAINER_COMMENT_LATE]
    result = select_resolution_comment(comments)
    assert result is MAINTAINER_COMMENT_LATE


def test_select_resolution_comment_none_when_no_maintainer_comment():
    assert select_resolution_comment([BOT_COMMENT, CONTRIBUTOR_COMMENT]) is None


ISSUE_WITH_RESOLUTION = {
    "number": 1234,
    "title": "regex= doesn't work on Path params",
    "body": "Using `Path(..., regex=...)` raises a warning, what should I use instead?",
    "labels": [{"name": "question"}],
    "closed_at": "2026-01-05T00:00:00Z",
}
ISSUE_NO_BODY = {
    "number": 5678,
    "title": "Empty issue",
    "body": None,
    "labels": [],
    "closed_at": "2026-01-05T00:00:00Z",
}


def test_issue_to_document_with_resolution():
    comments = [BOT_COMMENT, MAINTAINER_COMMENT_LATE]
    doc = issue_to_document(ISSUE_WITH_RESOLUTION, comments)

    assert doc is not None
    assert doc.doc_type.value == "issue"
    assert doc.resolution_comment_id == "4"
    assert "pattern=" in doc.text
    assert doc.id == "issue::1234"


def test_issue_to_document_without_resolution_still_included():
    doc = issue_to_document(ISSUE_WITH_RESOLUTION, [CONTRIBUTOR_COMMENT])

    assert doc is not None
    assert doc.resolution_comment_id is None


def test_issue_to_document_none_for_empty_body():
    assert issue_to_document(ISSUE_NO_BODY, []) is None
