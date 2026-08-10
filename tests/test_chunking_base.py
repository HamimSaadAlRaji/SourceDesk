from aegis.chunking.base import make_embedding_text
from aegis.types import DocType, Document


def _doc(**overrides) -> Document:
    base = dict(
        id="doc::tutorial/query-params.md",
        doc_type=DocType.DOCUMENTATION,
        text="body",
        content_hash="abc123",
        title="Query Parameters",
        doc_version="0.115.0",
    )
    base.update(overrides)
    return Document(**base)


def test_enrichment_off_returns_text_unchanged():
    result = make_embedding_text("raw chunk text", _doc(), heading_path=["Tutorial", "Query Parameters"], enrichment=False)
    assert result == "raw chunk text"


def test_enrichment_on_prepends_title_headingpath_version():
    result = make_embedding_text(
        "raw chunk text", _doc(), heading_path=["Tutorial", "Query Parameters"], enrichment=True
    )
    assert result.startswith("Query Parameters > Tutorial > Query Parameters > 0.115.0")
    assert result.endswith("raw chunk text")


def test_enrichment_on_omits_missing_version_cleanly():
    doc = _doc(doc_version=None)
    result = make_embedding_text("raw chunk text", doc, heading_path=["Tutorial"], enrichment=True)

    assert " > > " not in result
    assert not result.split("\n\n")[0].endswith(">")


def test_enrichment_on_omits_empty_heading_path_cleanly():
    doc = _doc()
    result = make_embedding_text("raw chunk text", doc, heading_path=[], enrichment=True)

    assert " > > " not in result
    prefix = result.split("\n\n")[0]
    assert prefix == "Query Parameters > 0.115.0"


def test_enrichment_on_does_not_duplicate_title_when_heading_path_starts_with_it():
    # aegis.ingest.github_issues prepends the issue title as the document's
    # own H1 ("# {title}\n\n{body}"), so heading_aware's heading_path for
    # every issue chunk literally starts with document.title. Naively
    # prepending document.title *and* heading_path duplicates it verbatim
    # in every issue chunk's embedded prefix -- issues are 83% of the
    # corpus. Measured cost: Phase 6 recall@10 0.4193 -> 0.1364 with the
    # duplication present.
    doc = _doc(title="Type annotation mismatch in sub-dependencies example", doc_version=None)
    result = make_embedding_text(
        "raw chunk text",
        doc,
        heading_path=["Type annotation mismatch in sub-dependencies example", "Privileged issue"],
        enrichment=True,
    )

    prefix = result.split("\n\n")[0]
    assert prefix == "Type annotation mismatch in sub-dependencies example > Privileged issue"
    assert prefix.count("Type annotation mismatch in sub-dependencies example") == 1
