"""Chunk the full Phase 1 corpus and persist it — the artifact Phase 2
never produced on purpose (comparing four strategies isn't the same as
committing to one). Phase 3 needs something concrete to embed before
Phase 6's ablation grid exists to pick a winner empirically, so this uses
heading-aware as a provisional default: it has the best measured code-block
integrity and is the only strategy with real heading attribution (see
`docs/explainers/02-chunking-strategies.md`). Phase 6 re-chunks with all
four strategies and can overturn this choice with real retrieval numbers.

I/O-bound orchestration over tested units (`chunking/heading_aware.py`) —
covered by running it against the real corpus, per CLAUDE.md's
I/O-component testing guidance, same as `ingest/docs_loader.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

from aegis.chunking.factory import build_chunker
from aegis.config import settings
from aegis.types import Document

PROVISIONAL_STRATEGY = "heading_aware"


def load_documents(path: Path) -> list[Document]:
    documents = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            documents.append(Document(**json.loads(line)))
    return documents


def chunk_documents(documents: list[Document], strategy: str = PROVISIONAL_STRATEGY) -> list:
    chunker = build_chunker(strategy)
    chunks = []
    for doc in documents:
        chunks.extend(chunker.chunk(doc))
    return chunks


def main() -> None:
    documents_path = settings.paths.data_processed / "documents.jsonl"
    output_path = settings.paths.data_processed / "chunks.jsonl"

    documents = load_documents(documents_path)
    print(f"Loaded {len(documents)} documents from {documents_path}")

    chunks = chunk_documents(documents, PROVISIONAL_STRATEGY)
    print(f"Chunked into {len(chunks)} chunks using strategy={PROVISIONAL_STRATEGY!r}")

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
