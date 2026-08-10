"""Core data types shared across the pipeline.

`Chunk` carries every field the *eventual* system needs (heading path, doc
version, content hash, product area) even though Phase 0's naive baseline
only uses `id`/`doc_id`/`text`. Retrofitting metadata later means re-embedding
the whole corpus, so the schema is decided once, here.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class DocType(StrEnum):
    DOCUMENTATION = "documentation"
    ISSUE = "issue"


class Document(BaseModel):
    id: str
    doc_type: DocType
    text: str
    content_hash: str
    source_path: str | None = None
    title: str | None = None
    doc_version: str | None = None
    product_area: str | None = None

    # documentation-only
    heading_path: list[str] = Field(default_factory=list)
    nav_section: str | None = None

    # issue-only
    labels: list[str] = Field(default_factory=list)
    resolution_comment_id: str | None = None


class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    heading_path: list[str] = Field(default_factory=list)
    doc_version: str | None = None
    doc_type: DocType
    product_area: str | None = None
    content_hash: str
    token_count: int

    # What actually gets embedded. Equal to `text` unless contextual
    # enrichment is on (Phase 2), in which case it's `text` prefixed with
    # `{doc_title} > {heading_path} > {version}` — kept separate so citations
    # and UI display always show the reader the real source text, never the
    # synthetic prefix. No default: every chunker must set it explicitly
    # rather than silently falling back to an empty embedding input.
    embedding_text: str


class Query(BaseModel):
    id: str
    text: str


class Retrieved(BaseModel):
    chunk: Chunk
    score: float
    rank: int


class Answer(BaseModel):
    query: Query
    text: str
    retrieved: list[Retrieved]
    citations: list[str] = Field(default_factory=list)
    abstained: bool = False


class Stratum(StrEnum):
    """plan.md Phase 4 step 3 — the six categories the golden set is
    reported per-stratum against, so an aggregate score can't hide that
    the system is excellent at how-to and hopeless at multi-hop."""

    HOW_TO = "how_to"
    ERROR_DEBUGGING = "error_debugging"
    CONCEPTUAL = "conceptual"
    VERSION_SPECIFIC = "version_specific"
    MULTI_HOP = "multi_hop"
    AMBIGUOUS = "ambiguous"


class NegativeCategory(StrEnum):
    """plan.md Phase 4 step 4 — the four negative-set categories, each with
    its own expected system behavior."""

    OUT_OF_CORPUS = "out_of_corpus"
    BILLING_LEGAL = "billing_legal"
    UNANSWERABLE = "unanswerable"
    ADVERSARIAL_INJECTION = "adversarial_injection"


class GoldenQuery(BaseModel):
    """One row of the golden evaluation set. Positives (`is_negative=False`)
    carry a `stratum` and human-confirmed `ground_truth_chunk_ids`; negatives
    carry a `negative_category` and `expected_behavior` instead — they have
    no correct chunk to retrieve, only a correct refusal/escalation behavior.
    """

    id: str
    text: str
    is_negative: bool
    stratum: Stratum | None = None
    negative_category: NegativeCategory | None = None
    expected_behavior: str | None = None
    source_issue: str | None = None
    resolution_hint: str | None = None
    ground_truth_chunk_ids: list[str] = Field(default_factory=list)
    split: str = "dev"
