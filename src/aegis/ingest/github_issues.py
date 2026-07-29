"""Closed-issue collection and resolution-comment extraction.

FastAPI's actual repo labels are process/triage labels, not topical ones
(see `product_area.py`), so `product_area` here comes from title+body text,
not labels. GitHub Issues has no native "accepted answer" marker (unlike
Discussions), so a "resolution comment" is a heuristic: the most recent
comment from someone with repo-maintainer standing
(`author_association` in OWNER/MEMBER/COLLABORATOR). Not every closed issue
has one — that count is itself a required Phase 1 measurement, since it's
the ceiling on how many issues the Phase 4 golden set can draw ground truth
from.
"""

from __future__ import annotations

import asyncio

import httpx

from aegis.ingest.hashing import content_hash
from aegis.ingest.product_area import derive_product_area_from_text
from aegis.types import DocType, Document

_MAINTAINER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
_API_ROOT = "https://api.github.com"


def is_maintainer_comment(comment: dict) -> bool:
    return comment.get("author_association") in _MAINTAINER_ASSOCIATIONS


def select_resolution_comment(comments: list[dict]) -> dict | None:
    candidates = [c for c in comments if is_maintainer_comment(c)]
    if not candidates:
        return None
    return max(candidates, key=lambda c: c["created_at"])


def issue_to_document(issue: dict, comments: list[dict]) -> Document | None:
    body = (issue.get("body") or "").strip()
    if not body:
        return None

    title = issue["title"]
    resolution = select_resolution_comment(comments)
    labels = [label["name"] for label in issue.get("labels", [])]

    text = f"# {title}\n\n{body}"
    if resolution is not None:
        author = resolution["user"]["login"]
        text += f"\n\n**Resolution (@{author}):**\n\n{resolution['body']}"

    return Document(
        id=f"issue::{issue['number']}",
        doc_type=DocType.ISSUE,
        text=text,
        content_hash=content_hash(text),
        title=title,
        product_area=derive_product_area_from_text(title, body),
        labels=labels,
        resolution_comment_id=str(resolution["id"]) if resolution else None,
    )


async def _get_json(client: httpx.AsyncClient, url: str, params: dict | None = None) -> object:
    for attempt in range(5):
        response = await client.get(url, params=params)
        if response.status_code == 403 and "rate limit" in response.text.lower():
            wait = 2**attempt
            await asyncio.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    response.raise_for_status()
    return response.json()


async def fetch_closed_issues(repo: str, token: str) -> list[dict]:
    """All closed *issues* (PRs excluded) via the paginated REST issues
    endpoint, which returns both issues and PRs mixed together."""
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    issues: list[dict] = []
    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        page = 1
        while True:
            batch = await _get_json(
                client,
                f"{_API_ROOT}/repos/{repo}/issues",
                params={"state": "closed", "per_page": 100, "page": page},
            )
            if not batch:
                break
            issues.extend(item for item in batch if "pull_request" not in item)
            page += 1
    return issues


async def fetch_comments_for_issues(
    repo: str, token: str, issue_numbers: list[int], concurrency: int = 15
) -> dict[int, list[dict]]:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[int, list[dict]] = {}

    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:

        async def _fetch(number: int) -> None:
            async with semaphore:
                results[number] = await _get_json(
                    client, f"{_API_ROOT}/repos/{repo}/issues/{number}/comments", params={"per_page": 100}
                )

        await asyncio.gather(*(_fetch(n) for n in issue_numbers))
    return results


async def run_issue_ingestion(repo: str, token: str, concurrency: int = 15) -> list[Document]:
    issues = await fetch_closed_issues(repo, token)
    numbers = [issue["number"] for issue in issues]
    comments_by_number = await fetch_comments_for_issues(repo, token, numbers, concurrency=concurrency)

    documents = []
    for issue in issues:
        doc = issue_to_document(issue, comments_by_number.get(issue["number"], []))
        if doc is not None:
            documents.append(doc)
    return documents
