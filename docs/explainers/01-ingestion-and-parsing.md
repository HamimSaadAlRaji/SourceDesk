# 01 — Ingestion and Parsing

**Phase:** 1 · **Date:** 2026-07-29 · **Files:** `src/aegis/ingest/markdown.py`, `docs_loader.py`, `github_issues.py`, `hashing.py`, `product_area.py`, `run.py`
**Topics from the course:** What makes production challenging (lived, this phase specifically)

## 1. What I built

A structure-preserving markdown parser that resolves FastAPI's custom MkDocs Material
syntax (`{* *}`/`{! !}` code includes, `///`/`////` admonitions and tabs) into clean,
complete markdown; a docs loader that walks `docs/en/docs/` through it (129 pages); and
a GitHub issues loader that pulls every closed issue via the authenticated REST API,
extracts a maintainer resolution comment where one exists (2,204 of 3,530), and
classifies both doc types into a shared `product_area` taxonomy. Output:
`data/processed/documents.jsonl`, 3,659 documents.

## 2. Why this approach

Phase 0's design record ended with a specific, reproduced failure: the naive baseline
fabricated an answer about path-parameter regex validation because the real answer —
`pattern=`, in `docs_src/query_params_str_validations/tutorial004_an_py310.py` — was
never in the index. Not retrieved wrong: never ingested at all, because raw
`.read_text()` doesn't follow `{* ... *}` includes. This phase's job was narrow and
concrete: make that specific failure impossible, and do the same structural diligence
for the issue corpus, which has no analogous "include" problem but has its own
structural noise (an old issue-template checklist, pasted build logs, PRs mixed into
the issues endpoint).

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
|---|---|---|
| Resolve `{* path *}` relative to each markdown file's own directory (standard filesystem semantics) | Simpler, no special-casing | Measured: the literal include text `../../docs_src/...` is identical in files at different nesting depths (`tutorial/query-params-str-validations.md`, 1 level deep, and `tutorial/security/first-steps.md`, 2 levels deep). Per-file-relative resolution would need a longer `../` prefix for the deeper file; it doesn't vary, so FastAPI's own plugin must resolve against a fixed base. Confirmed by resolving against `docs/en/` (the mkdocs.yml directory) and checking the target file exists — it does, for every file tested regardless of depth. |
| Parse `mkdocs.yml`'s `nav:` tree for `heading_path`/`nav_section` | More "correct," grounded in the real site structure | Inspected the actual tree: most nav groups are unnamed (`""`) — the site relies on each page's own H1 as the effective label (`navigation.tabs` feature). Parsing the YAML would add real complexity for accuracy `mkdocs.yml` itself doesn't provide; directory name + H1 title gets the same practical breadcrumb. |
| Derive issue `product_area` from GitHub labels, as sketched in plan.md | Labels are structured, presumably reliable | Measured: fetched the live label list — all process/triage labels (`bug`, `p1`-`p5`, `lang-*`, `approved-N`). No topical labels exist. The one label that looks topical, `dependencies`, is dependency-version-bump tooling, not the DI feature. Switched both doc types to the same keyword-based taxonomy instead — see §6 for how noisy this still turned out to be. |
| Cap issue ingestion at ~2,000–3,000, preferring resolved ones | Matches plan.md's target directly | Ingested all 3,530 real closed issues (PRs excluded) instead, tagging which have a resolution comment rather than filtering to only those. Filtering first would have thrown away the actual measurement plan.md's Measure step asks for ("count of issues with an identifiable resolution — this is the ceiling on your golden set") — capping before measuring would have made that number a choice, not a fact. |

## 4. How it works

Include resolution — two syntaxes, one fixed base directory (`docs/en/`), and a raise
(not a skip) on a missing target, because a silent skip here is the exact Phase 0
failure mode reproduced with new files:

```python
def resolve_includes(text: str, base_dir: Path) -> str:
    text = _INCLUDE_RE.sub(
        lambda m: f"```{_LANG_BY_SUFFIX.get((base_dir / m.group('path')).suffix, '')}\n"
        f"{_read_include(m.group('path'), base_dir)}\n```",
        text,
    )
    text = _SNIPPET_RE.sub(lambda m: _read_include(m.group("path"), base_dir), text)
    return text
```

Admonition flattening resolves the *innermost* fence first — a same-length-fence
backreference plus a negative lookahead that refuses to cross a nested fence — so
`////tab` blocks containing `///tip` blocks resolve correctly regardless of nesting
depth, and indentation is tracked so admonitions nested inside list items (found in
`deployment/docker.md`) don't get skipped:

```python
_FENCE_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<fence>/{3,4})[ \t]*(?P<label>[^\n]*)\n"
    r"(?P<body>(?:(?!^[ \t]*/{3,4}).)*?)\n?"
    r"^(?P=indent)(?P=fence)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
```

Resolution-comment selection has no ground truth to check against — GitHub Issues
has no "accepted answer" marker (unlike Discussions) — so it's an explicit heuristic:
the most recent comment from someone with maintainer standing on the repo:

```python
_MAINTAINER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}

def select_resolution_comment(comments: list[dict]) -> dict | None:
    candidates = [c for c in comments if c.get("author_association") in _MAINTAINER_ASSOCIATIONS]
    return max(candidates, key=lambda c: c["created_at"]) if candidates else None
```

## 5. What the numbers said

**Corpus size.** 129 documentation pages (revised down from plan.md's ≥300 estimate —
English `docs/en/docs/` is 155 files total, 131 excluding auto-generated `reference/`
stubs, 129 after also dropping two non-content files found during loading — see
Decision Log). 3,530 closed issues (PRs excluded via the `pull_request` key on the
REST issues endpoint, which returns both mixed together) — comfortably above the
2,000 floor. **2,204 of 3,530 (62.4%) have an identifiable maintainer resolution** —
this is the real ceiling on Phase 4's golden set.

**Token distribution** (chars/4 estimate): docs p50=1,436, p95=6,609, max=171,510;
issues p50=470, p95=1,444, max=48,397. Both max values are legitimate, not bugs — the
doc outlier is `release-notes.md` (686,041 chars, FastAPI's full changelog); the issue
outlier is #2266, a user who pasted an entire PyInstaller build log including a full
stack trace. Neither was a parsing error; both are real long-tail production data,
directly relevant to Phase 2's chunk-sizing decisions.

**`product_area` distribution** (issues, final): general 899, openapi 607,
dependencies 350, request-data 217, security 207, deployment 202, async 183,
middleware 174, databases 159, pydantic 140, responses 121, websockets 105,
frontend-integration 73, testing 63, cors 48. `general` (the no-match fallback) is the
largest single bucket at 25% — an honest result: a lot of real issues (vague bug
reports, environment/installation problems, feature requests) don't cleanly map to one
FastAPI feature area.

## 6. What surprised me / what to watch for

**A second include syntax slipped past every automated check.** `{* *}` was the
documented syntax; three files (`python-types.md`, `advanced/templates.md`,
`advanced/generate-clients.md`) use `{!path!}`/`{!> path!}` (pymdownx.snippets)
instead. My full-corpus stress test checked for leftover `{*`, `///`, and fence
markers and reported zero failures — clean. It was still wrong. I only found this by
opening `advanced/templates.md`'s actual parsed output and reading it, per the exit
criterion's "actually open them." This is the single best justification I've hit so
far for why that criterion exists as a hard requirement and not a nice-to-have:
automated leftover-syntax checks only catch syntax you already know to check for.

**Two rounds of a systematic classifier bug, both from boilerplate text, not real
content.** First: matching keywords against raw issue text caused `async` to fire on
~20% of all issues (695 of 3,530) regardless of topic, because nearly every issue
includes a repro snippet using `async def`. Fixed by stripping fenced code blocks
before matching. Second, larger: FastAPI's old issue-creation template includes a
checklist — *"I already checked if it is not related to FastAPI but to
[Pydantic]/[Swagger UI]/[ReDoc]"* — verbatim on **1,416 of 3,530 issues (40%)**,
which made `openapi`/`pydantic` the dominant categories purely from template
boilerplate (one sampled issue, "Feature: cron task scheduler," had zero real
connection to OpenAPI and was classified `openapi` solely because the checklist
mentions Swagger UI). Fixed by stripping that specific line pattern. A third, narrower
instance turned up during manual verification — a maintainer's canned "please open a
GitHub Discussion following all the template" reply matched the keyword `template` —
and I stopped there rather than continuing to catalog every canned maintainer
response; that's an open-ended, diminishing-returns problem for a keyword heuristic,
documented here as a known residual limitation rather than chased further. If
`product_area` proves load-bearing for Phase 8's router, it's worth replacing this
approach with something that isn't purely lexical.

**GitHub's core rate limit (5,000/hour) is a real constraint, not a hypothetical
one.** The full fetch (list + per-issue comments) uses ~3,630 requests. Running it
twice inside the same session to apply the `async`-keyword fix exhausted the limit on
the second run's 3,530th call. Recovered by reprocessing the already-fetched raw data
locally (saved before the second run) instead of re-fetching — which also turned out
to be the *more correct* way to verify the "re-running ingestion produces identical
hashes" exit criterion: `content_hash` is a deterministic pure function of its input,
verified directly, rather than comparing two live API calls, since the live issue data
is a mutable source (new comments can land between two real fetches) unlike the
frozen git-cloned docs corpus. Re-running ingestion against a *live* API twice in a
row and expecting byte-identical output is close to the wrong test.

## 7. Where this fits and what's next

`data/processed/documents.jsonl` is the input to Phase 2's chunking laboratory. Two
concrete numbers from this phase are direct inputs there: `release-notes.md`'s
686K-char size and issue #2266's 193K-char pasted log both push hard on whatever
chunk-size default gets picked, and both are worth checking by hand once chunking
strategies are compared. The 62.4% issue-resolution rate (2,204 documents) is the hard
ceiling for how many issue-derived Q&A pairs Phase 4's golden set can draw on — it
should not try to exceed what's actually resolvable. `product_area`, imperfect as
measured here, is what Phase 8's router will filter/route on; if routing accuracy
turns out to be sensitive to the ~25% residual misclassification rate documented
above, that's the trigger to revisit this with something less lexical.
