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
