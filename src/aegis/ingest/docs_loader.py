"""Walk `docs/en/docs/`, parse every page, produce `Document` objects.

I/O-bound orchestration over the tested units in `markdown.py`,
`hashing.py`, and `product_area.py` — covered by an integration check
against the real corpus (`docs/results/01-doc-verification.md`) rather than
unit tests, per CLAUDE.md's I/O-component testing guidance.
"""

from __future__ import annotations

import re
from pathlib import Path

from aegis.ingest.hashing import content_hash
from aegis.ingest.markdown import clean_markdown, extract_title
from aegis.ingest.product_area import derive_product_area_from_doc_path
from aegis.types import DocType, Document

_EXCLUDE_TOP_LEVEL_DIRS = {"reference", "css", "img", "js"}
# Not documentation: a test fixture for FastAPI's own translation-LLM prompts,
# filed under docs/ only because it needs to go through the same translation
# pipeline as real pages. Would pollute retrieval with meaningless filler text.
_EXCLUDE_FILES = {"_llm-test.md"}
_VERSION_RE = re.compile(r'__version__\s*=\s*"([^"]+)"')


def _read_fastapi_version(repo_root: Path) -> str:
    init_file = repo_root / "fastapi" / "__init__.py"
    match = _VERSION_RE.search(init_file.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"Could not find __version__ in {init_file}")
    return match.group(1)


def load_docs(repo_root: Path) -> list[Document]:
    docs_root = repo_root / "docs" / "en" / "docs"
    mkdocs_base = repo_root / "docs" / "en"
    fastapi_version = _read_fastapi_version(repo_root)

    documents: list[Document] = []
    for path in sorted(docs_root.rglob("*.md")):
        rel = path.relative_to(docs_root)
        if rel.parts[0] in _EXCLUDE_TOP_LEVEL_DIRS or rel.name in _EXCLUDE_FILES:
            continue

        raw = path.read_text(encoding="utf-8")
        cleaned = clean_markdown(raw, mkdocs_base)
        title = extract_title(cleaned)
        if title is None:
            # No H1 -> not a real content page (e.g. translation-banner.md,
            # a transclusion snippet for non-English pages).
            continue

        rel_str = rel.as_posix()
        is_root_file = len(rel.parts) == 1
        nav_section = "root" if is_root_file else rel.parts[0]
        heading_path = [title] if is_root_file else [nav_section.capitalize(), title]

        documents.append(
            Document(
                id=f"doc::{rel_str}",
                doc_type=DocType.DOCUMENTATION,
                text=cleaned,
                content_hash=content_hash(cleaned),
                source_path=rel_str,
                title=title,
                doc_version=fastapi_version,
                product_area=derive_product_area_from_doc_path(rel_str),
                heading_path=heading_path,
                nav_section=nav_section,
            )
        )
    return documents
