# Aegis — Progress Log

**Last updated:** 2026-07-29
**Current phase:** Phase 2 — Chunking Laboratory
**Status:** 🟢 Phase 1 complete — 129 docs + 3,530 issues ingested to `documents.jsonl`

> This file is the state of the world. It is updated at the end of every phase and
> whenever something significant lands. If you return after a gap, read this first —
> it is designed to be sufficient to reload full context on its own.

---

## Quick Status

| | |
|---|---|
| **Day** | 2 of 21 (complete) |
| **Phase** | 2 of 19 (starting) |
| **Design records written** | 2 of 20 |
| **Golden set** | not built (Phase 4) |
| **Definition of Done** | 0 of 10 criteria met |
| **Blockers** | none |

---

## Phase Board

Legend: ⬜ not started · 🟡 in progress · ✅ complete · ⏭️ cut

| # | Phase | Day | Status | Key output |
|---|---|---|---|---|
| 0 | Foundations & failing baseline | 1 | ✅ | Naive RAG that hallucinates, on purpose |
| 1 | Corpus acquisition & ingestion | 2 | ✅ | `documents.jsonl` |
| 2 | Chunking laboratory | 3 | ⬜ | 4 strategies, one interface |
| 3 | Embeddings, distance & vector store | 4 | ⬜ | ANN recall/latency curve |
| 4 | **Golden evaluation set** ⚑ | 5 | ⬜ | 180 queries + 40 negatives, frozen |
| 5 | Retrieval core (analyzer, BM25, dense, RRF) | 6 | ⬜ | Custom analyzer + fusion |
| 6 | **Retrieval eval & ablation grid** ⚑ | 7 | ⬜ | The evidence table |
| 7 | Reranking | 8 | ⬜ | Pool-size vs latency curve |
| 8 | Agentic router & query understanding | 9–10 | ⬜ | Routing confusion matrix |
| 9 | Grounded generation & citations | 11 | ⬜ | Citation validator |
| 10 | **Calibrated abstention** ⚑ | 12 | ⬜ | The escalation confusion matrix |
| 11 | **Generation eval + judge validation** ⚑ | 13 | ⬜ | RAGAS + Cohen's κ |
| 12 | Caching, delta indexing & cost | 14 | ⬜ | Cache hit rate, cost table |
| 13 | Observability & latency budget | 15 | ⬜ | p95 < 3s, per-stage breakdown |
| 14 | API, streaming & UI | 16–17 | ⬜ | Working demo w/ visible escalation |
| 15 | Security hardening | 18½ | ⬜ | Adversarial suite |
| 16 | **CI evaluation gate** ⚑ | 18½ | ⬜ | Gate that fails on regression |
| 17 | LangGraph port & framework comparison | 19 | ⬜ | Comparison writeup, both paths pass eval |
| 18 | Deployment | 20 | ⬜ | Public URL |
| 19 | Failure analysis & writeup | 21 | ⬜ | Held-out numbers, taxonomy, README |

⚑ = never cut (see `plan.md` §7)

---

## Measured Results

*Nothing goes here that wasn't produced by a command that was actually run.*

| Metric | Value | Phase | Config | Date |
|---|---|---|---|---|
| Full docs corpus size vs. context window | 363,420 tok (155 files) vs. 128K window (llama-3.3-70b-versatile) — 2.84× over | 0 | full `docs/en/docs/` tree | 2026-07-29 |
| Baseline subset size | 121,553 tok (85 tutorial/advanced docs, 486,214 chars) | 0 | `src/aegis/baseline.py` | 2026-07-29 |
| Cost: full-subset-stuffing vs. retrieval, per query (input tokens only) | $0.0717 vs. ~$0.0002 (~300× cheaper) | 0 | top_k=3, 500-char chunks, Groq $0.59/M input | 2026-07-29 |
| Confident hallucinations recorded | 3 of 5 questions, 2 independent runs | 0 | `docs/results/00-baseline-transcript.json` | 2026-07-29 |
| Documentation pages ingested | 129 (of 155 English `.md` files; 131 excl. `reference/`, 129 after excluding 2 non-content files) | 1 | `data/processed/documents.jsonl` | 2026-07-29 |
| Closed issues ingested (PRs excluded) | 3,530 | 1 | GitHub REST API, `fastapi/fastapi` | 2026-07-29 |
| Issues with identifiable maintainer resolution | 2,204 / 3,530 (62.4%) — ceiling for Phase 4 golden set | 1 | `author_association` in {OWNER,MEMBER,COLLABORATOR} | 2026-07-29 |
| Token distribution (docs / issues), chars/4 estimate | docs p50=1,436 p95=6,609 max=171,510; issues p50=470 p95=1,444 max=48,397 | 1 | `documents.jsonl` | 2026-07-29 |

### Definition of Done tracker

| # | Criterion | Target | Current | Status |
|---|---|---|---|---|
| 1 | Recall@10 + MRR lift, baseline → final | measured | — | ⬜ |
| 2 | RAGAS faithfulness | ≥ 0.90 | — | ⬜ |
| 3 | Escalation confusion matrix published | 4 quadrants | — | ⬜ |
| 4 | Cache hit rate + cost/latency reduction | measured | — | ⬜ |
| 5 | CI gate demonstrably fails on regression | screenshot | — | ⬜ |
| 6 | p95 end-to-end, retrieval path | < 3 s | — | ⬜ |
| 7 | Failure analysis, categorized | written | — | ⬜ |
| 8 | LLM-judge agreement (Cohen's κ) | ≥ 0.6 | — | ⬜ |
| 9 | Adversarial security suite passing | all pass | — | ⬜ |
| 10 | Held-out generalization number | reported | — | ⬜ |

---

## Decision Log

Every non-obvious choice, its alternatives, and why. Append-only.

| Date | Decision | Alternatives rejected | Reasoning |
|---|---|---|---|
| 2026-07-28 | FastAPI as the corpus | Supabase; Kubernetes | Structured Markdown with heavy code blocks, ~11k closed issues as free labels, and a domain the user can personally verify — which is what makes the golden set trustworthy. |
| 2026-07-28 | 21 days, full-time | 14 days | Full-rigor eval is ~30% of the work; 14 days forces cuts to the evidentiary spine. |
| 2026-07-28 | Full-rigor evaluation | Lighter eval; minimal smoke tests | The abstention threshold cannot be calibrated without it, and calibrated abstention is the entire differentiator of the project. |
| 2026-07-28 | **Golden set at Phase 4** (moved from proposal §5.8) | Keep proposal ordering | Every choice in Phases 5–11 needs a measurement to justify it. Evaluation is a prerequisite, not a report card. |
| 2026-07-28 | Added Phase 15 (security) | Leave security as a reading topic | The corpus is user-generated GitHub comments flowing into an LLM prompt — a live indirect-injection surface. |
| 2026-07-28 | Added negative set (40 queries) | Golden set only | Without refusal-expected queries, two quadrants of the confusion matrix are empty and τ is tuned on data where abstention is always wrong. |
| 2026-07-28 | Python 3.12 via `uv` | System Python 3.13; poetry; pip+venv | Parts of the ML stack still lag 3.13. `uv` for speed and a lockfile. |
| 2026-07-28 | Qdrant | pgvector; Chroma; Weaviate | Native sparse vectors + strong payload filtering for hybrid search. |
| 2026-07-28 | Groq primary, Gemini fallback | Single provider | Free tiers usable; multi-provider fallback is a production requirement anyway. |
| 2026-07-28 | LLM-judge model **deferred to Phase 11** | Pick now | Judge quality dominates eval trustworthiness — decide after measuring agreement against hand labels. |
| **2026-07-28** | **v2 revision — build/import rebalanced.** BM25 → `bm25s`; retrieval metrics → `ranx`. Kept building: custom analyzer, heading-aware chunker, RRF, confidence scorer, citation validator, routing policy, eval harness. | v1: hand-roll BM25 and metrics from scratch | User already knows the topics (stated explicitly) — the goal is hiring signal, not re-learning. Writing a mature library's job proves nothing to an interviewer and costs days better spent on the ablation grid and abstention system, which *are* the differentiators. |
| 2026-07-28 | Cut ColBERT implementation → design-record only | Full implementation | ~0.5 day for a result reasonable to predict without building it; bi-vs-cross-encoder comparison already covers the core lesson |
| 2026-07-28 | Three-way encoder bake-off → two (bge-large, BGE-M3) | Keep bge-small as a third full candidate | bge-small kept only as the fast dev/iteration workhorse, not a final candidate — the third full comparison was ~0.5 day for a near-certain result |
| 2026-07-28 | Added Phase 17 — LangGraph port | Skip frameworks entirely; build on LangChain/LangGraph from the start | Framework fluency is a real screening filter. Building on a framework from day one hides the user's own architecture and makes debugging harder. Porting a working, measured system afterward gets both the keyword and a defensible, evidence-based opinion ("here's what it cost, here's what it bought") rather than a preference. |
| 2026-07-28 | Time freed by the above (~1 day) reallocated to the LangGraph port | Compress schedule to 20 days instead | Keeps 21-day target with more buffer in Phases 5–7, which were previously stretched thin |
| 2026-07-29 | Kept folder name `SourceDesk`, did not rename to `aegis` | Rename per plan.md literally | The rename's stated reason (space in path breaking Docker volume mounts) doesn't apply — the actual path has no space. Renaming would also disrupt the already-open editor session for no benefit. The Python package is still named `aegis` (`src/aegis/`), so nothing downstream changes. |
| 2026-07-29 | Qdrant image pinned to `v1.18.3`, not the plan's unstated default | Leave at whatever `docker pull qdrant/qdrant` resolves to | `qdrant-client` 1.18.0 (latest at `uv add` time) warns on incompatibility with server versions more than one minor apart; started on v1.12.4 by accident (stale copy-paste), hit the warning immediately, re-pinned to v1.18.3 to match. Cheap to fix now, expensive to debug later as a "why are recall numbers weird" mystery. |
| 2026-07-29 | Baseline embedding model: `BAAI/bge-small-en-v1.5` via `fastembed` (ONNX) | `sentence-transformers` + MPS | No torch install, no MPS setup, for a component that's thrown away at the end of this phase. Real bake-off (bge-large vs BGE-M3) is Phase 3 and will decide the model actually used going forward. |
| 2026-07-29 | Baseline corpus = 85 files (`tutorial/` + `advanced/` only), not the full 155-file docs tree | Ingest everything now | Phase 0's job is a fast, deliberately-bad demo, not real corpus acquisition — that's Phase 1's structure-preserving parser. The subset was large enough (121,553 tokens) to produce real, reproducible hallucinations on the first run. |
| 2026-07-29 | Generation LLM for the baseline: Groq `llama-3.3-70b-versatile` only (no Gemini fallback yet) | Wire up the fallback now | Fallback logic is a production concern (Phase 9+); Phase 0 only needs one call path to prove the architecture fails. |
| 2026-07-29 | Phase 1 exit criterion revised: ≥300 docs → ≥125 docs (final: 129 loaded) | Keep ≥300 and pad the corpus (all locales, or include `reference/`) | Measured: English `docs/en/docs/` is 155 `.md` files total, 131 excluding `reference/` (24 files of `::: fastapi.ClassName` mkdocstrings auto-doc stubs — no renderable prose without the mkdocstrings plugin actually running against the installed package), 129 after also dropping `_llm-test.md` (FastAPI's own translation-LLM prompt test fixture — not documentation, would inject meaningless filler text into retrieval) and `translation-banner.md` (no H1, a transclusion snippet with no standalone content). 300 was an estimate made without checking the repo; padding with other-language locales or content-free stub pages would hit an arbitrary number without adding real signal — the measurement wins. |
| 2026-07-29 | `product_area` derived from a shared keyword table (path-based for docs, title+body text for issues), not from GitHub labels | Derive issue `product_area` from labels, as plan.md sketched | Measured: fetched the live label list for `fastapi/fastapi` — it's all process/triage labels (`bug`, `p1`-`p5`, `lang-*`, `approved-N`, `dependencies`). No topical labels exist (no `security`, `async`, `cors`, etc.), and the one label that looks topical (`dependencies`) is dependency-version-bump tooling, not the FastAPI DI feature. Labels can't carry the signal the router (Phase 8) needs; text keyword matching can, and using the same taxonomy for both doc types (rather than two incompatible schemes) is what a router actually needs. |
| 2026-07-29 | `{* path *}` include directives resolve relative to `docs/en/` (the mkdocs.yml directory), a fixed base for every file — not relative to each markdown file's own directory | Resolve relative to each file's own parent directory (standard filesystem-relative semantics) | Measured: the literal include text `../../docs_src/...` is identical across files at different nesting depths (`tutorial/query-params-str-validations.md`, 1 level deep, and `tutorial/security/first-steps.md`, 2 levels deep, both use the exact same `../../docs_src/...` prefix). Standard per-file relative resolution would require a longer `../` prefix for deeper files; it doesn't vary, so FastAPI's own mkdocs plugin must resolve these against a fixed base equal to the mkdocs.yml directory, confirmed by resolving against `docs/en/` and checking the target file exists. |
| 2026-07-29 | Also resolve `{!path!}`/`{!> path!}` (pymdownx.snippets syntax) | Ship with `{* *}` support only | Found by manually opening `advanced/templates.md`'s parsed output (the exit-criterion-mandated step, not an automated check) — 3 files use this alternate include syntax instead of `{* *}`, and every automated leftover-syntax check I'd written passed cleanly despite the content being missing. Direct validation of why "actually open them" is a hard requirement, not a nice-to-have. |
| 2026-07-29 | Excluded `_llm-test.md` and `translation-banner.md` from the doc corpus (131 → 129) | Include everything under `docs/en/docs/*.md` | Both are non-content: `_llm-test.md` is FastAPI's own test fixture for their translation-LLM prompts (495 lines of test scaffolding, not documentation); `translation-banner.md` is a transclusion snippet with no H1 and no standalone content. Including either would inject meaningless filler into retrieval. |
| 2026-07-29 | Ingested all 3,530 closed issues (not capped at 2,000–3,000), tagging which have a resolution comment rather than pre-filtering | Filter down to only resolved issues before ingesting, to match the plan's target range | Filtering first would have thrown away the actual required measurement (Phase 1's Measure step: "count of issues with an identifiable resolution — the ceiling on your golden set"). Ingesting everything and reporting the resolved fraction (2,204/3,530, 62.4%) makes that number a fact instead of a choice. |
| 2026-07-29 | `product_area` for issues strips fenced code blocks and a specific issue-template boilerplate line before keyword matching | Match keywords against raw issue text directly | Measured via manual sampling: raw-text matching classified `async` on 695/3,530 issues (20%) — nearly every issue includes an `async def` repro snippet regardless of topic. Worse: FastAPI's old issue template's checklist literally says "I already checked if it is not related to FastAPI but to [Pydantic]/[Swagger UI]/[ReDoc]" on 1,416/3,530 issues (40%), which was silently dominating the `pydantic`/`openapi` categories from boilerplate, not content. Both fixed by stripping before matching (verified with regression tests that fail without the fix). A third, narrower source (canned maintainer replies, e.g. "template" in a stock "open a Discussion" response) was found but not chased further — documented as a residual limitation instead, since cataloging every canned reply is open-ended. |
| 2026-07-29 | Verified hash-reproducibility via `content_hash` as a pure function on fixed input, not by re-running the live GitHub fetch twice | Re-run `aegis.ingest.run` twice and diff the output, as originally planned | Running the full ~3,630-request fetch twice in one session hit GitHub's 5,000/hour core rate limit on the second run's 3,530th call. Recovered by reprocessing the already-fetched raw data locally (saved before the second run) instead of re-fetching — which is also the methodologically correct test: the live issue API is a mutable source (comments can be added between two real fetches), unlike the frozen git-cloned docs corpus, so two live fetches aren't expected to be byte-identical regardless of code correctness. |

---

## Design Records

| # | Doc | Phase | Status |
|---|---|---|---|
| 00 | `the-baseline-that-fails.md` | 0 | ✅ |
| 01 | `ingestion-and-parsing.md` | 1 | ✅ |
| 02 | `chunking-strategies.md` | 2 | ⬜ |
| 03 | `embeddings-distance-and-ann.md` | 3 | ⬜ |
| 04 | `building-a-golden-set.md` | 4 | ⬜ |
| 05 | `lexical-dense-and-fusion.md` | 5 | ⬜ |
| 06 | `retrieval-metrics-and-ablation.md` | 6 | ⬜ |
| 07 | `reranking-and-encoder-architectures.md` | 7 | ⬜ |
| 08 | `agentic-routing-and-query-understanding.md` | 8 | ⬜ |
| 09 | `generation-prompting-and-citations.md` | 9 | ⬜ |
| 10 | `calibrated-abstention.md` | 10 | ⬜ |
| 11 | `evaluating-generation.md` | 11 | ⬜ |
| 12 | `caching-delta-indexing-and-cost.md` | 12 | ⬜ |
| 13 | `observability-and-latency.md` | 13 | ⬜ |
| 14 | `serving-streaming-and-ux.md` | 14 | ⬜ |
| 15 | `rag-security.md` | 15 | ⬜ |
| 16 | `ci-eval-gates.md` | 16 | ⬜ |
| 17 | `frameworks-and-orchestration.md` | 17 | ⬜ |
| 18 | `deployment.md` | 18 | ⬜ |
| 19 | `what-i-built-and-what-i-learned.md` | 19 | ⬜ |

---

## Course Topic Coverage

37 topics from `RAG_Topics_Learned.md`. Marked ✅ only when exercised *and* recorded, whether built or imported-and-measured. Full mapping in `plan.md` §2.

**Module 1 (5):** ✅⬜⬜✅⬜ — topics 1 & 4 done at P0 (baseline demo; cost/latency measurement). Topic 2 spans P0+P5, marked done only when P5 lands.
**Module 2 (8):** ⬜⬜⬜⬜⬜⬜⬜⬜
**Module 3 (6):** ⬜⬜⬜⬜⬜⬜
**Module 4 (9):** ⬜⬜⬜⬜⬜⬜⬜⬜⬜
**Module 5 (9):** ✅⬜⬜⬜⬜⬜⬜⬜⬜ — topic 29 (what makes production challenging) exercised at P1: GitHub API rate limits, a second undocumented include syntax, boilerplate text poisoning a keyword classifier.

**Progress: 3 / 37**

---

## Build vs Import Ledger

Tracks what's actually implemented from scratch vs. imported, so the ratio stated in interviews matches reality. Updated as components land. Full rationale in `plan.md` §1 / `CLAUDE.md` §2.

| Component | Plan | Actual | Phase |
|---|---|---|---|
| Custom analyzer/tokenizer | Build | — | 5 |
| BM25 | Import (`bm25s`) | — | 5 |
| Heading-aware chunker | Build | — | 2 |
| RRF fusion | Build | — | 5 |
| Retrieval metrics | Import (`ranx`) | — | 6 |
| Cross-encoder reranker | Import | — | 7 |
| Confidence scorer | Build | — | 10 |
| Citation validator | Build | — | 9 |
| Routing policy | Build | — | 8 |
| Eval harness | Build | — | 6 |
| Semantic cache | Build | — | 12 |
| RAGAS metrics | Import | — | 11 |
| Tracing | Import (Langfuse) | — | 13 |

---

## Open Questions

| # | Question | Raised | Needed by | Resolution |
|---|---|---|---|---|
| 1 | Which LLM for the judge? Free tier may not clear κ ≥ 0.6 | 2026-07-28 | Phase 11 | Open — decide from the agreement test |
| 2 | Do FastAPI closed issues yield ≥ 500 with clear resolutions? | 2026-07-28 | Phase 4 | **Resolved 2026-07-29** — 2,204 of 3,530 closed issues (62.4%) have an identifiable maintainer resolution comment, well above 500 |
| 3 | Deploy target: HF Spaces vs Render? Free-tier terms shift | 2026-07-28 | Phase 18 | Open — reverify terms at Phase 18 |
| 4 | Groq `llama-3.3-70b-versatile` at temperature=0.0 is not fully deterministic across runs — two Phase 0 baseline runs against the same index produced different fabricated answers to the same question | 2026-07-29 | Phase 6 (eval harness), Phase 11 (judge validation) | Open — decide whether to accept as a property of the system under test, cache generations for reproducibility, or something else |

---

## Session Log

Newest first. One entry per working session.

### 2026-07-29 — Phase 1: Corpus acquisition & ingestion

**Context.** Same session as Phase 0, continued directly. Goal: turn Phase 0's
concrete, reproduced hallucination (missing `pattern=` content because raw
`.read_text()` never followed FastAPI's `{* *}` includes) into a fixed, tested
pipeline, plus acquire the closed-issue half of the corpus for the first time.

**Done**
- TDD'd a structure-preserving markdown parser (`src/aegis/ingest/markdown.py`):
  resolves `{* path *}` and `{!path!}`/`{!> path!}` includes (two syntaxes, only found
  the second by manually opening parsed output — automated leftover-syntax checks
  missed it entirely), flattens `///`/`////` admonitions and tabs including ones
  nested inside list items, strips decorative HTML wrappers
- Discovered (not assumed) that FastAPI's own plugin resolves includes against a
  fixed base directory (`docs/en/`) regardless of a file's nesting depth — confirmed
  by comparing identical include prefixes across files at different depths
- Built the docs loader: 129 pages (revised down from plan.md's ≥300 estimate after
  measuring the real corpus size — see Decision Log), stress-tested against the full
  corpus, 10 samples manually opened and verified
- Authenticated `gh` (needed a live debugging round — `gh auth status` kept failing
  from the tool's shell even after the user logged in; resolved on the user's end,
  cause not fully diagnosed)
- Built the GitHub issues loader: fetched all 3,530 closed issues (PRs excluded) +
  comments via the authenticated REST API, concurrent `httpx` with rate-limit backoff.
  Resolution-comment heuristic: most recent comment from an OWNER/MEMBER/COLLABORATOR
  author. 2,204/3,530 (62.4%) have one
- Found and fixed two systematic `product_area` classifier bugs via manual sampling,
  not automated checks: raw-text keyword matching fired on `async` in ~20% of issues
  from pasted code snippets, and on `openapi`/`pydantic` in 40% of issues from a
  literal old issue-template checklist line naming Pydantic/Swagger UI/ReDoc. Both
  fixed (code-block stripping + boilerplate-line stripping) with regression tests
- Hit GitHub's 5,000/hour rate limit re-running the full fetch a second time to apply
  the fix; recovered by reprocessing already-fetched raw data locally instead of
  re-fetching — see Decision Log, this also changed how hash-reproducibility got
  verified (pure-function check, not live-refetch diff)
- 10 issue documents manually sampled and opened; all structurally correct
- Wrote `data/processed/documents.jsonl`: 3,659 documents (129 docs + 3,530 issues)
- Design record: `docs/explainers/01-ingestion-and-parsing.md`

**Exit criteria — all met** (criterion revised from ≥300 to ≥125 docs, see Decision
Log; 129 docs / 3,530 issues actual)
- `documents.jsonl` with ≥125 docs and ≥2,000 issues
- All metadata fields populated (verified programmatically, zero missing)
- Re-running ingestion produces identical hashes (verified as a pure-function
  property of `content_hash`, not a live re-fetch — see Decision Log)
- Ten randomly sampled parsed docs manually verified, actually opened (done for both
  doc types — 20 total, not just the required 10)

**Surprised by:** the `{!path!}` include syntax that no automated check caught: found
only by reading actual output. Two rounds of boilerplate text (not real content)
dominating a keyword classifier — the issue-template checklist alone affected 40% of
the corpus. `release-notes.md` (686K chars) and issue #2266 (a 193K-char pasted build
log) are both legitimate long-tail content, not bugs, and will pressure-test Phase 2's
chunk-size choice directly.

**Next**
- Phase 2: chunking laboratory. `release-notes.md` and issue #2266 are concrete stress
  cases for whatever chunking strategies get compared. Four strategies (fixed,
  recursive, heading-aware, semantic), one interface, benchmarked against each other.

### 2026-07-29 — Phase 0: Foundations & the baseline that fails

**Context.** First implementation session. Kicked off Phase 0 per plan.md's "First
Three Actions." User opted to keep the project folder as `SourceDesk` rather than
renaming to `aegis` — the rename's rationale (space in path) doesn't apply to the
actual path, and renaming would've disrupted the already-open editor. The Python
package is still `aegis` (`src/aegis/`).

**Done**
- Installed `uv`; git-initialized the repo (was not previously version-controlled);
  built the full `src/aegis/` package skeleton per plan.md §3's repo layout
- `docker-compose.yml` with Qdrant, pinned to `v1.18.3` after hitting a client/server
  version-compatibility warning on the initially-used `v1.12.4`
- `config.py` (pydantic-settings, per-group `env_prefix`, reads `.env`) and `types.py`
  (`Chunk` with the full metadata schema plan.md specifies, not just what Phase 0 uses)
- `pytest` wired up, `Makefile` with the five entry points (`ingest`/`index`/`eval`/
  `serve`/`test` — only `test` is functional yet, the rest are stubs for later phases)
- Naive baseline (`src/aegis/baseline.py`, ~150 lines): shallow-cloned `fastapi/fastapi`,
  raw-text-loaded 85 tutorial/advanced doc pages (no markdown parsing), 500-char
  non-overlapping fixed chunks, `bge-small-en-v1.5` via `fastembed`, Qdrant dense-only
  top-3, Groq `llama-3.3-70b-versatile`, no citations, no abstention
- Ran it against 5 known-answer FastAPI questions, twice. Found and doc-verified 3
  confident hallucinations across two independent, unrelated mechanisms (fabricated
  code / wrong status code on a question with zero relevant retrieved content; a
  confidently-recommended *deprecated* API on a question where retrieval found the
  right file but fixed-size chunking severed the "(deprecated)" heading from its body)
- Design record: `docs/explainers/00-the-baseline-that-fails.md` — includes real
  token/cost math (full docs corpus is 2.84× over the 128K context window before Phase
  1 even adds the issue corpus; retrieval is ~300× cheaper than stuffing the reduced
  subset) sourced from an actual Groq pricing fetch and real file measurements, plus
  full transcripts in `docs/results/00-baseline-transcript.json`

**Exit criteria — all met**
- `make test` green (2 passed)
- Qdrant reachable from Python (`QdrantClient(...).get_collections()` succeeded)
- Baseline answers end-to-end (5/5 questions, both runs)
- ≥ 1 recorded confident hallucination (3 found, doc-verified against the actual
  FastAPI docs, not asserted from memory)

**Surprised by:** two unrelated failure mechanisms in the same 5-question run (missing
content from unresolved `{* ... *}` includes vs. a heading severed from its body at a
chunk boundary) — expected chunking to be a problem in the abstract, didn't expect two
distinct causal paths to show up this fast. Also: temp=0.0 on Groq isn't fully
deterministic run-to-run (logged as open question #4).

**Next**
- Phase 1: corpus acquisition & ingestion. Needs a real structure-preserving markdown
  parser that resolves FastAPI's `{* ... *}` includes and `///` admonition blocks
  (Phase 0 proved concretely what breaks if it doesn't), plus closed-issue collection
  via the GitHub API (target 2,000–3,000, verify the ≥500-with-resolution assumption —
  open question #2).

### 2026-07-28 — Plan revision (v2): rebalanced for hiring, not learning

**Context.** User clarified they already understand all topics in `RAG_Topics_Learned.md` — the goal is landing an industry role, not learning RAG from scratch. v1 defaulted to building nearly everything by hand, which was the wrong optimization target once that was known.

**Done**
- Ranked what actually differentiates a RAG portfolio project in interviews: evaluation rigor > calibrated abstention > production ops > public demo > framework fluency > from-scratch implementations
- Rewrote the build/import split (`plan.md` §1): import BM25 (`bm25s`), embeddings, HNSW, reranker, retrieval metrics (`ranx`), RAGAS, Langfuse, FastAPI. Keep building: custom analyzer, heading-aware chunker, RRF, confidence scorer, citation validator, routing policy, eval harness, semantic cache — the ~900 lines that are genuinely the project's own contribution
- Cut ColBERT to a design-record-only topic; cut the three-way encoder bake-off to two real candidates
- Added Phase 17: build the orchestration directly first, then port to LangGraph and measure the delta — gets the framework keyword plus a defensible, evidence-based opinion instead of a preference
- Freed ~1 day from the import swaps, spent it on the LangGraph phase; net schedule still 21 days, more buffer in Phases 5–7
- Updated `CLAUDE.md` — explainers reframed as engineering design records (decision-and-evidence focused, not concept tutorials), added the build-vs-import test as an explicit rule
- Updated `progress.md` (this entry) — renumbered phase board, added the Build vs Import Ledger to keep the stated ratio honest

**Next**
- Phase 0: rename folder to `aegis`, `uv init` on Python 3.12, package skeleton, Qdrant in Docker, then build the naive baseline and catch it hallucinating.
