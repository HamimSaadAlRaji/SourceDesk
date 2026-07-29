"""`product_area` derivation — the router (Phase 8) needs this on every doc.

FastAPI's actual GitHub labels are process/triage labels (`bug`, `p1`-`p5`,
`lang-*`, `approved-N`) — there is no topical taxonomy on issues (confirmed
via the live label list; the one label that looks topical, `dependencies`,
is dependency-version-bump tooling, not the FastAPI DI feature). So instead
of the labels-based split plan.md sketched, both docs and issues are
classified with the same keyword table: docs by file path, issues by
title+body text. One taxonomy across both doc types is what Phase 8's
router actually needs.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
# FastAPI's old issue template checklist ("I already checked if it is not
# related to FastAPI but to [Pydantic]/[Swagger UI]/[ReDoc]") appears
# verbatim on ~40% of all closed issues regardless of topic (measured) --
# it would otherwise falsely classify huge swathes of unrelated issues as
# "pydantic" or "openapi" purely from boilerplate.
_TEMPLATE_BOILERPLATE_RE = re.compile(
    r"- ?\[[xX ]\] I already checked if it is not related to FastAPI but to \[[^\]]*\]\([^)]*\)\.?"
)

# Order matters: first match wins, most specific first.
_KEYWORDS: list[tuple[str, str]] = [
    ("security", "security"),
    ("oauth2", "security"),
    ("dependencies", "dependencies"),
    ("dependency", "dependencies"),
    ("deployment", "deployment"),
    ("docker", "deployment"),
    ("cors", "cors"),
    ("middleware", "middleware"),
    ("websocket", "websockets"),
    ("background-task", "background-tasks"),
    ("background_task", "background-tasks"),
    ("sql-database", "databases"),
    ("database", "databases"),
    ("testing", "testing"),
    ("async", "async"),
    ("openapi", "openapi"),
    ("swagger", "openapi"),
    # Hyphen-adjacent, not bare "response" -- doc slugs use hyphens
    # ("custom-response.md", "response-model.md"); bare "response" is too
    # generic in free text ("no response from server"), measured on real
    # issues. Underscore/no-space variants cover prose mentions of the
    # actual FastAPI API names.
    ("-response", "responses"),
    ("response-", "responses"),
    ("response_model", "responses"),
    ("response_class", "responses"),
    ("jsonresponse", "responses"),
    ("streamingresponse", "responses"),
    ("htmlresponse", "responses"),
    ("plaintextresponse", "responses"),
    ("cookie", "request-data"),
    ("header-param", "request-data"),
    ("query-param", "request-data"),
    ("path-param", "request-data"),
    ("request-form", "request-data"),
    ("request-file", "request-data"),
    ("body", "request-data"),
    ("setting", "configuration"),
    ("environment-variable", "configuration"),
    ("graphql", "graphql"),
    ("static-file", "frontend-integration"),
    ("template", "frontend-integration"),
    ("frontend", "frontend-integration"),
    ("sub-application", "app-structure"),
    ("bigger-application", "app-structure"),
    ("pydantic", "pydantic"),
]

_TOP_LEVEL_FALLBACK = {"tutorial": "core"}


def derive_product_area_from_doc_path(rel_path: str) -> str:
    lowered = rel_path.lower()
    for keyword, area in _KEYWORDS:
        if keyword in lowered:
            return area
    top = PurePosixPath(rel_path).parts[0]
    if top.endswith(".md"):
        return "general"
    return _TOP_LEVEL_FALLBACK.get(top, top)


def derive_product_area_from_text(title: str, body: str) -> str:
    """Code blocks are stripped before matching: almost every FastAPI issue
    includes a repro snippet using `async def`, which made `async` fire on
    ~20% of all issues regardless of topic when matched against raw text —
    caught by manually reviewing a sample of `async`-classified issues."""
    body_without_code = _CODE_BLOCK_RE.sub(" ", body)
    body_without_code = _TEMPLATE_BOILERPLATE_RE.sub(" ", body_without_code)
    lowered = f"{title}\n{body_without_code}".lower()
    for keyword, area in _KEYWORDS:
        if keyword in lowered:
            return area
    return "general"
