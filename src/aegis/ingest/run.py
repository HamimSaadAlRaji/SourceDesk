"""Entry point for `make ingest`: docs + GitHub issues -> documents.jsonl."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from aegis.config import settings
from aegis.ingest.docs_loader import load_docs
from aegis.ingest.github_issues import run_issue_ingestion
from aegis.types import Document

REPO = "fastapi/fastapi"


def _gh_token() -> str:
    return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()


def write_documents_jsonl(documents: list[Document], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.model_dump_json())
            f.write("\n")


async def _main_async() -> None:
    repo_root = settings.paths.data_raw / "fastapi"
    docs = load_docs(repo_root)
    print(f"Loaded {len(docs)} documentation pages")

    token = _gh_token()
    issues = await run_issue_ingestion(REPO, token)
    print(f"Loaded {len(issues)} issues")

    out_path = settings.paths.data_processed / "documents.jsonl"
    write_documents_jsonl(docs + issues, out_path)
    print(f"Wrote {len(docs) + len(issues)} documents to {out_path}")


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
