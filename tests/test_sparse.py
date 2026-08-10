"""TDD for BM25 sparse retrieval (plan.md Phase 5, step 2).

`bm25s` itself is imported -- a mature, fast, correct BM25 implementation,
not worth reimplementing (CLAUDE.md §2). What's actually under test here is
the wiring: that the index is built and queried through *our* analyzer
(aegis.analysis.analyzer.analyze), not bm25s's own default tokenizer. A
silent regression to the default tokenizer would corrupt every retrieval
measurement downstream without a single test failing elsewhere -- exactly
the kind of bug CLAUDE.md §6 requires TDD to catch.
"""

from aegis.retrieval.sparse import SparseIndex
from aegis.types import Chunk, DocType


def _chunk(id: str, text: str, doc_id: str | None = None) -> Chunk:
    return Chunk(
        id=id,
        doc_id=doc_id if doc_id is not None else id,
        text=text,
        doc_type=DocType.ISSUE,
        content_hash="hash",
        token_count=len(text.split()),
        embedding_text=text,
    )


def test_exact_identifier_query_ranks_the_matching_chunk_first():
    # plan.md Phase 5 exit criterion: "BM25 returns sane results for
    # ECONNREFUSED-style queries where dense fails" -- a real corpus
    # example (docs/explainers/04 harvest), not an invented one.
    chunks = [
        _chunk("c1", "Restart the server if the port is already in use."),
        _chunk("c2", "Client raised ECONNREFUSED when connecting to the database."),
        _chunk("c3", "Set the response model to control the output schema."),
    ]
    index = SparseIndex(chunks)

    results = index.search("ECONNREFUSED", top_k=3)

    assert results[0][0].id == "c2"
    assert results[0][1] > 0


def test_snake_case_decomposition_matches_natural_language_query():
    # The chunk only contains "response_model" (joined); the query uses
    # natural-language spacing. This only works because the analyzer
    # emits both the whole identifier and its split parts (test_analyzer.py).
    chunks = [
        _chunk("c1", "Use response_model to filter the returned fields."),
        _chunk("c2", "Configure the database connection pool size."),
    ]
    index = SparseIndex(chunks)

    results = index.search("response model", top_k=2)

    assert results[0][0].id == "c1"
    assert results[0][1] > 0


def test_top_k_limits_result_count():
    chunks = [_chunk(f"c{i}", f"chunk number {i} about routing") for i in range(5)]
    index = SparseIndex(chunks)

    results = index.search("routing", top_k=2)

    assert len(results) == 2


def test_query_with_no_lexical_overlap_scores_zero():
    chunks = [_chunk("c1", "Set up dependency injection for the database session.")]
    index = SparseIndex(chunks)

    results = index.search("unrelated_zzz_term", top_k=1)

    assert results[0][1] == 0


def test_k1_and_b_are_configurable():
    chunks = [_chunk("c1", "status_code and response_model examples")]
    index = SparseIndex(chunks, k1=1.2, b=0.9)

    results = index.search("status_code", top_k=1)

    assert results[0][0].id == "c1"


def test_exclude_doc_id_removes_chunks_belonging_to_that_document():
    # A golden query's own source issue is a near-verbatim match of the
    # query text -- a trivial hit that ground truth deliberately excludes.
    chunks = [
        _chunk("c1", "routing setup with dependency injection", doc_id="issue::100"),
        _chunk("c2", "routing setup with dependency injection", doc_id="issue::200"),
    ]
    index = SparseIndex(chunks)

    results = index.search("routing dependency injection", top_k=5, exclude_doc_id="issue::100")

    assert [c.id for c, _ in results] == ["c2"]


def test_exclude_doc_id_still_returns_top_k_when_enough_remain():
    chunks = [_chunk(f"c{i}", "routing dependency injection", doc_id=f"issue::{i}") for i in range(5)]
    index = SparseIndex(chunks)

    results = index.search("routing dependency injection", top_k=3, exclude_doc_id="issue::0")

    assert len(results) == 3
    assert "c0" not in [c.id for c, _ in results]


def test_exclude_doc_id_none_does_not_filter_anything():
    chunks = [_chunk("c1", "routing setup", doc_id="issue::100")]
    index = SparseIndex(chunks)

    results = index.search("routing", top_k=5, exclude_doc_id=None)

    assert [c.id for c, _ in results] == ["c1"]
