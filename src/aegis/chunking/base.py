"""Shared `Chunker` interface and contextual enrichment (plan.md Phase 2,
"contextual retrieval" — usually the cheapest large win in the pipeline).

Enrichment prepends `{doc_title} > {heading_path} > {version}` to the text
that gets *embedded*, never to `Chunk.text` itself — citations and UI display
must show the reader real source text, not a synthetic breadcrumb.
"""

from __future__ import annotations

from typing import Protocol

from aegis.types import Chunk, Document


class Chunker(Protocol):
    name: str

    def chunk(self, document: Document) -> list[Chunk]: ...


def enrichment_prefix(document: Document, heading_path: list[str]) -> str:
    parts = [document.title, " > ".join(heading_path) if heading_path else None, document.doc_version]
    return " > ".join(p for p in parts if p)


def make_embedding_text(text: str, document: Document, heading_path: list[str], enrichment: bool) -> str:
    if not enrichment:
        return text
    prefix = enrichment_prefix(document, heading_path)
    return f"{prefix}\n\n{text}" if prefix else text
