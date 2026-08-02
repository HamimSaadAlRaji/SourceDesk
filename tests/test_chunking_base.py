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
