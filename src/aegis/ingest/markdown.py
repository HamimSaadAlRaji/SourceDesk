"""Structure-preserving markdown parsing for FastAPI's MkDocs Material docs.

Phase 0's naive baseline read these files with a raw `.read_text()` and never
resolved `{* ... *}` code includes at all — the actual example code (e.g. the
`pattern=` regex-validation syntax) lived in a separate `.py` file the naive
ingestion never followed, so it wasn't fabricating from bad retrieval, it was
fabricating from a corpus that never had the answer in it. This module is the
fix: resolve includes and admonitions into plain, complete, chunkable text.
"""

from __future__ import annotations

import re
from pathlib import Path

_INCLUDE_RE = re.compile(r"\{\*\s*(?P<path>\S+?)(?:\s+[^*]*)?\*\}")
# pymdownx.snippets syntax: `{!path!}` or `{!> path!}`. Used in three files
# where the `{* *}` syntax isn't (found by manually opening parsed output —
# an automated leftover-syntax check for `{*` didn't catch this variant).
_SNIPPET_RE = re.compile(r"\{!\s*>?\s*(?P<path>[^!]+?)\s*!\}")
_FENCE_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<fence>/{3,4})[ \t]*(?P<label>[^\n]*)\n"
    r"(?P<body>(?:(?!^[ \t]*/{3,4}).)*?)\n?"
    r"^(?P=indent)(?P=fence)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
_HEADING_RE = re.compile(r"^#\s+(?P<title>.+?)\s*(?:\{\s*#\S+\s*\})?\s*$", re.MULTILINE)
_DIV_LINE_RE = re.compile(r"^<div class=\"[^\"]*\">\s*$\n?", re.MULTILINE)
_DIV_CLOSE_RE = re.compile(r"^</div>\s*$\n?", re.MULTILINE)
_IMG_LINE_RE = re.compile(r"^<img[^>]*>\s*$\n?", re.MULTILINE)

_LANG_BY_SUFFIX = {".py": "python", ".json": "json", ".sh": "bash"}


def _read_include(rel_path: str, base_dir: Path) -> str:
    target = (base_dir / rel_path).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Include not found: {rel_path} (resolved to {target})")
    return target.read_text(encoding="utf-8").rstrip("\n")


def resolve_includes(text: str, base_dir: Path) -> str:
    """Replace `{* relpath [hl[...]] *}` with the referenced file's content
    as a fenced code block, and `{!relpath!}` / `{!> relpath!}`
    (pymdownx.snippets syntax, used in a few files instead) with the raw
    content inline, since those already sit inside an existing fence in the
    source. Raises if a referenced file is missing — a silent skip here is
    exactly the Phase 0 failure mode."""
    text = _INCLUDE_RE.sub(
        lambda m: f"```{_LANG_BY_SUFFIX.get((base_dir / m.group('path')).suffix, '')}\n"
        f"{_read_include(m.group('path'), base_dir)}\n```",
        text,
    )
    text = _SNIPPET_RE.sub(lambda m: _read_include(m.group("path"), base_dir), text)
    return text


def resolve_admonitions(text: str) -> str:
    """Flatten `///type` and `////tab | Label` blocks into labeled prose.
    Handles nesting by repeatedly resolving the innermost fence (the one
    whose body contains no further fence lines) until none remain."""
    while True:
        match = _FENCE_RE.search(text)
        if match is None:
            return text
        label = match.group("label").strip()
        indent = match.group("indent")
        body = match.group("body")
        if indent:
            body = "\n".join(line.removeprefix(indent) for line in body.split("\n"))
        body = body.strip()

        if "|" in label:
            _, _, title = label.partition("|")
            title = title.strip()
        elif label.lower().startswith("tab"):
            title = label.split(None, 1)[1].strip() if len(label.split(None, 1)) > 1 else "Tab"
        else:
            title = (label.split(None, 1)[0] if label else "Note").capitalize()

        rendered = f"**{title}:**\n\n{body}"
        text = text[: match.start()] + rendered + text[match.end() :]


def strip_decorative_html(text: str) -> str:
    """Drop presentational-only wrapper markup: `<div class="...">`/`</div>`
    (unwrap, keep inner content) and standalone `<img>` lines (screenshots —
    no retrievable text value in this corpus)."""
    text = _DIV_LINE_RE.sub("", text)
    text = _DIV_CLOSE_RE.sub("", text)
    text = _IMG_LINE_RE.sub("", text)
    return text


def clean_markdown(text: str, base_dir: Path) -> str:
    """Full Phase 1 parse: resolve includes, flatten admonitions, strip
    decorative HTML. Output is plain markdown with real headings and code
    blocks — Phase 2's chunker reads this, not the raw file."""
    text = resolve_includes(text, base_dir)
    text = resolve_admonitions(text)
    text = strip_decorative_html(text)
    return text


def extract_title(text: str) -> str | None:
    """First `# Heading { #anchor }` line's title, anchor stripped."""
    match = _HEADING_RE.search(text)
    return match.group("title") if match else None
