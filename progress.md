# Aegis — Progress Log

**Last updated:** 2026-07-29
**Current phase:** Phase 1 — Corpus Acquisition & Ingestion
**Status:** 🟢 Phase 0 complete — naive baseline built, run, and caught hallucinating twice

> This file is the state of the world. It is updated at the end of every phase and
> whenever something significant lands. If you return after a gap, read this first —
> it is designed to be sufficient to reload full context on its own.

---

## Quick Status

| | |
|---|---|
| **Day** | 1 of 21 (complete) |
| **Phase** | 1 of 19 (starting) |
| **Design records written** | 1 of 20 |
| **Golden set** | not built (Phase 4) |
| **Definition of Done** | 0 of 10 criteria met |
| **Blockers** | none |

---

## Phase Board

Legend: ⬜ not started · 🟡 in progress · ✅ complete · ⏭️ cut

| # | Phase | Day | Status | Key output |
|---|---|---|---|---|
| 0 | Foundations & failing baseline | 1 | ✅ | Naive RAG that hallucinates, on purpose |
| 1 | Corpus acquisition & ingestion | 2 | ⬜ | `documents.jsonl` |
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

---

## Design Records

| # | Doc | Phase | Status |
|---|---|---|---|
| 00 | `the-baseline-that-fails.md` | 0 | ✅ |
| 01 | `ingestion-and-parsing.md` | 1 | ⬜ |
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
**Module 5 (9):** ⬜⬜⬜⬜⬜⬜⬜⬜⬜

**Progress: 2 / 37**

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
| 2 | Do FastAPI closed issues yield ≥ 500 with clear resolutions? | 2026-07-28 | Phase 4 | Open — verify in Phase 1 |
| 3 | Deploy target: HF Spaces vs Render? Free-tier terms shift | 2026-07-28 | Phase 18 | Open — reverify terms at Phase 18 |
| 4 | Groq `llama-3.3-70b-versatile` at temperature=0.0 is not fully deterministic across runs — two Phase 0 baseline runs against the same index produced different fabricated answers to the same question | 2026-07-29 | Phase 6 (eval harness), Phase 11 (judge validation) | Open — decide whether to accept as a property of the system under test, cache generations for reproducibility, or something else |

---

## Session Log

Newest first. One entry per working session.

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
