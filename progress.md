# Aegis — Progress Log

**Last updated:** 2026-08-04
**Current phase:** Phase 6 — Retrieval Evaluation & the Ablation Grid
**Status:** 🟢 Phase 6 complete — `ranx` verified, eval harness built, full 4-axis ablation grid run against the golden set. Winning config: heading-aware chunking, contextual enrichment off, dense retrieval, `bge-large-en-v1.5` (Recall@10 0.4288, MRR 0.8640). Found and fixed three real bugs during measurement (thread-unsafe cache, an enrichment title-duplication bug, and a self-match leak that had been silently inflating every early measurement in this phase)

> This file is the state of the world. It is updated at the end of every phase and
> whenever something significant lands. If you return after a gap, read this first —
> it is designed to be sufficient to reload full context on its own.

---

## Quick Status

| | |
|---|---|
| **Day** | 7 of 21 (complete) |
| **Phase** | 6 of 19 (complete) |
| **Design records written** | 7 of 20 |
| **Golden set** | 180 dev + 40 holdout, frozen and hashed (Phase 4) |
| **Definition of Done** | 1 of 10 criteria met (#1: Recall@10 + MRR lift, measured) |
| **Blockers** | none |

---

## Phase Board

Legend: ⬜ not started · 🟡 in progress · ✅ complete · ⏭️ cut

| # | Phase | Day | Status | Key output |
|---|---|---|---|---|
| 0 | Foundations & failing baseline | 1 | ✅ | Naive RAG that hallucinates, on purpose |
| 1 | Corpus acquisition & ingestion | 2 | ✅ | `documents.jsonl` |
| 2 | Chunking laboratory | 3 | ✅ | 4 strategies, one interface |
| 3 | Embeddings, distance & vector store | 4 | ✅ | ANN recall/latency curve |
| 4 | **Golden evaluation set** ⚑ | 5 | ✅ | 180 queries + 40 negatives, frozen, hashed |
| 5 | Retrieval core (analyzer, BM25, dense, RRF) | 6 | ✅ | Custom analyzer + fusion |
| 6 | **Retrieval eval & ablation grid** ⚑ | 7 | ✅ | The evidence table |
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
| Chunk count, 4 strategies (129 docs + 51 sampled issues) | fixed 1,646 · recursive 1,791 · heading_aware 2,682 · semantic 1,787 | 2 | `max_tokens=400, overlap_tokens=50` | 2026-07-30 |
| Code-block integrity rate, 4 strategies | fixed 84.1% · recursive 85.3% · heading_aware 98.4% · semantic 99.1% (880 real blocks) | 2 | same config | 2026-07-30 |
| Chunks with populated `heading_path` | fixed/recursive/semantic 0.0% · heading_aware 99.9% | 2 | same config | 2026-07-30 |
| Oversized chunks, full 3,659-doc corpus (no sampling) | fixed 0/8,984 · recursive 0/9,561 · heading_aware 427/19,236 (420 documented atomic-code exception + 7 accepted residual limitation, see explainer §6) | 2 | `docs/results/chunking-stats.md` | 2026-07-30 |
| Semantic chunking wall time, single largest doc (`release-notes.md`, 3,690 sentences) | 211–248s (confirmed compute-bound: batched vs. 32 per-segment calls differ by only ~15%) | 2 | `bge-small-en-v1.5` / fastembed, CPU | 2026-07-30 |
| Cosine ≡ dot product on real corpus vectors | Max abs diff 1.79e-07 across 2,000 real chunk pairs; all norms exactly 1.000000 | 3 | `bge-small-en-v1.5`, 19,236 chunks | 2026-08-02 |
| Embedding throughput, full 19,236-chunk corpus | bge-small 1,517.1s (12.7/s avg, unmitigated, swap-driven) · bge-large 665.0s (28.9/s avg, mitigated) · BGE-M3 799.4s (24.1/s avg, mitigated) | 3 | MPS, M1 Pro 16GB | 2026-08-02 |
| Swap usage during sustained MPS embedding | ~3.5GB (`vm.swapusage`) | 3 | Concurrent Qdrant + MPS embedding load | 2026-08-02 |
| Exact-duplicate embeddings in corpus | 39.7% of 19,236 vectors; largest group 1,354 issues whose chunk text is the literal string `_No response_` | 3 | `bge-small-en-v1.5` | 2026-08-02 |
| HNSW recall@10 (score-aware, tie-corrected), 36 configs swept | 1.0000 for all 36 (m ∈ {8,16,32}, ef_construct ∈ {50,100,200}, ef_search ∈ {50,100,200,400}) | 3 | `docs/results/ann-recall-vs-latency.md` | 2026-08-02 |
| HNSW query latency, 36 configs swept | 6.07–8.07 ms/query (no consistent trend with any parameter) | 3 | 300 self-sampled queries, 19,236-pt corpus | 2026-08-02 |
| Post-filter vs. pre-filter recall, 5 `product_area` categories | Post-filter 0.1667–0.4033 (even `general`, 26% of corpus, short of k on 90% of queries) · Pre-filter 1.0000 on all 5 | 3 | `docs/results/prefilter-vs-postfilter.md` | 2026-08-02 |
| Resolved GitHub issues (FastAPI closed issues with maintainer resolution) | 2,204 of 3,530 (62.4%) | 4 | Phase 1 measurement; harvest pool for golden set | 2026-08-03 |
| Low-signal resolutions filtered (bare "thanks", pure redirects, <=6 words) | 95 of 2,204 (4.3%) | 4 | Conservative regex, retained as buffer | 2026-08-03 |
| Stratum distribution (2,109 candidates post-filter) | how_to 385 (18.3%) · error_debugging 1,096 (51.9%) · conceptual 41 (1.9%) · version_specific 58 (2.7%) · multi_hop 281 (13.3%) · ambiguous 343 (16.3%) | 4 | `src/aegis/eval/strata.py` classification | 2026-08-03 |
| Candidates harvested (30/stratum with 2x buffer) | 333 positives (60 how_to, 60 error_debugging, 38 conceptual, 55 version_specific, 60 multi_hop, 60 ambiguous) + 40 negatives = 373 total | 4 | Overflow buffer absorbs low-quality issues that survival labeling would skip | 2026-08-03 |
| Dense retrieval composition (unfiltered top-20) | 5,571 issue, 1,089 doc (83.6% issue-dominated) across 333 candidates; 103/333 (30.9%) with zero doc chunks in top-20 | 4 | Boilerplate-pollution failure mode (same as Phase 1/2): GitHub issue-template text dominates similarity | 2026-08-03 |
| Dense retrieval composition (12 doc + 8 issue filtered) | 3,996 doc, 2,664 issue (100% of queries guaranteed doc representation) | 4 | Fix: doc_type-filtered searches instead of flat top-20 | 2026-08-03 |
| Golden set labels (LLM-based, Haiku, token-overlap heuristic ≥15%) | 180 positives (30/stratum) + 40 negatives = 220 total | 4 | Deterministic, reproducible from seed; Phase 11 will measure agreement with Opus judge via Cohen's κ | 2026-08-03 |
| Golden set split (dev / holdout) | 180 dev (25 pos/stratum + 30 neg), 40 holdout (5 pos/stratum + 10 neg) | 4 | Holdout sealed; Phases 6–16 measure on dev, Phase 19 on holdout for generalization | 2026-08-03 |
| Golden set hash (dev split, SHA256) | `b48a1a73f7ca70231544972b0404d80a4260bcc541ca0264ea23848375b27214` | 4 | Proof-of-freeze in `data/golden/golden_set_v1.jsonl` | 2026-08-03 |
| Golden set hash (holdout split, SHA256) | `d4d10c2c5316615657f13caae93c73fbeb7c8ae6318f4fdb021afc18a3b7adcc` | 4 | Sealed in `data/golden/golden_set_holdout_v1.jsonl` | 2026-08-03 |
| Analyzer test suite (incl. the 40-real-token exit criterion) | 56/56 passed | 5 | `tests/test_analyzer.py`, tokens pulled from `documents.jsonl` | 2026-08-03 |
| BM25 vs. dense on a real `ECONNREFUSED` query, real issue `issue::5578` (contains the literal token) | BM25: rank 1, score 3.1467 · Dense (bge-small): not found in top 200 of 19,236 chunks | 5 | `SparseIndex`/`DenseIndex`, full corpus | 2026-08-03 |
| Unfiltered dense search vs. `doc_type`-prefiltered, query about `response_model` | Unfiltered top-5: 100% issue chunks (0.82-0.86 cosine), correct doc absent · Prefiltered: top-1 is `tutorial/response-model.md`, score 0.7864 | 5 | `DenseIndex`, bge-small, full corpus — reproduces Phase 3's finding through the new module | 2026-08-03 |
| Full test suite after Phase 5 | 245/245 passed | 5 | `uv run pytest -q` | 2026-08-03 |
| `ranx` verification (hand-computed 5-doc example) | Precision@5=0.6, Recall@5=1.0, MRR=1.0, MAP@5=0.7556, NDCG@5=0.8855 — exact match | 6 | `tests/test_metrics.py` | 2026-08-04 |
| Winning retrieval config: recall@10 / MRR / NDCG@10 / MAP@10 | 0.4288 / 0.8640 / 0.7064 / 0.3695 | 6 | heading-aware, enrichment off, dense, bge-large; 150 dev positives, self-match excluded | 2026-08-04 |
| Retrieval-strategy axis: dense vs. sparse vs. hybrid (retuned) | dense 0.4288 vs. sparse 0.0967 vs. best-tuned hybrid_rrf 0.4324 (statistically ties dense, never exceeds it) recall@10 | 6 | `docs/results/retrieval-ablation.md` §1 | 2026-08-04 |
| Encoder axis: bge-large vs. BGE-M3 (under dense) | 0.4288 vs. 0.1348 recall@10 | 6 | Same golden set, both already-indexed Phase 3 collections | 2026-08-04 |
| Contextual enrichment axis: off vs. on | 0.4288 vs. 0.1751 recall@10 — opposite of plan.md's default expectation | 6 | Root cause: GitHub issue-template boilerplate headings shared across thousands of unrelated issues, same failure mode as Phases 1-4's boilerplate-pollution | 2026-08-04 |
| Chunking axis (doc-level ground-truth transfer): heading_aware vs. fixed/recursive/semantic | 0.5470 vs. 0.2393 / 0.2436 / 0.2571 recall@10 | 6 | First real retrieval-quality evidence for Phase 3's provisional chunking choice; resolves Open Question #5 (semantic chunking's cost doesn't pay for itself) | 2026-08-04 |
| Semantic chunking, full 3,659-doc corpus (first full-corpus run — Phase 2 only sampled) | 723s to chunk + 640s to embed (9,168 chunks) | 6 | `experiments/06a_prepare_ablation_indexes.py semantic` | 2026-08-04 |
| Self-match leak effect (before vs. after `exclude_doc_id` fix), dense/heading-aware/bge-large | recall@10 0.4193→0.4288, MRR 0.6161→0.8640 | 6 | Every golden query's own source issue was a trivial near-verbatim top-1 hit, wasting a top-10 slot; affected every measurement in this phase's first pass | 2026-08-04 |
| Full test suite after Phase 6 | 272/272 passed | 6 | `uv run pytest -q` | 2026-08-04 |

### Definition of Done tracker

| # | Criterion | Target | Current | Status |
|---|---|---|---|---|
| 1 | Recall@10 + MRR lift, baseline → final | measured | Recall@10 +0.332 (4.4x), MRR +0.496 (2.35x) vs. worst measured config (sparse-only) — see `docs/results/retrieval-ablation.md` §7 for why Phase 0's literal baseline isn't corpus-comparable | ✅ |
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
| 2026-07-30 | `tiktoken` (`cl100k_base`) added as a dependency for chunk-size token counting, not the eventual embedding model's own tokenizer | Load the real bge-large/BGE-M3 tokenizer (not chosen until Phase 3) | Chunk boundaries only need a count that's stable and consistent with itself across strategies, not exact parity with an encoder that doesn't exist yet in this project. `max_tokens=400` is set with headroom below any real encoder's 512-token limit to absorb this approximation plus the contextual-enrichment prefix. |
| 2026-07-30 | Code-block atomicity guaranteed only for heading-aware and semantic, deliberately not extended to fixed/recursive | Give every strategy the same atomicity guarantee | Fixed and recursive are supposed to be the naive/moderate baselines; their measured code-integrity rate (84.1% / 85.3% at `max_tokens=400`) is itself the useful data point for Phase 6, not a defect to quietly fix. |
| 2026-07-30 | `greedy_merge` (shared by recursive/heading-aware/semantic) verifies packing by actually joining candidate pieces and re-counting tokens, not by summing each piece's own token count | Keep the sum-based running-total approach (simpler, was the original implementation) | The sum-based version does not account for the join separator's own token cost, and is not the same number as the true joined text's token count in general (BPE isn't additive across concatenation) — found as two separate real violations of `max_tokens` on real corpus documents (issue #15000 at 443/400, and a semantic-chunking test at 26/25) before being replaced. See `docs/explainers/02-chunking-strategies.md` §6. |
| 2026-07-30 | `recursive_split_pieces` drops the original separator instead of re-attaching it to each piece | Re-attach the separator to preserve exact source whitespace when pieces are concatenated | The re-attached separator plus `greedy_merge`'s own join separator doubled up at every merge boundary, inflating real chunk token counts past `max_tokens` (issue #15000, 443/400) — found via full-corpus measurement, not caught by unit tests using clean small inputs. |
| 2026-07-30 | Closing-fence detection (`segments.py`) accepts any line that *starts* with `` ``` ``, not only a line that *is* `` ``` `` (optionally with trailing whitespace) | Strict CommonMark-style closing-fence match | Real issue text breaks a strict match twice: CRLF line endings (closing line is literally `` ```\r\n ``) and a closing fence sharing its line with trailing HTML (`` ```</div> ``). Either one, unhandled, made the closing-fence branch never match, silently swallowing every heading and paragraph after it into one giant "code" segment (issue #15000 went from 2 detected sections to the correct 9 once fixed). |
| 2026-07-30 | Accepted a residual limitation: 7 of 19,236 heading-aware chunks (0.036%) still exceed `max_tokens`, not chased further | Build a more sophisticated fence parser (e.g. adopt a mature CommonMark library) to also handle these | All 7 affected documents have a genuinely *odd* number of literal `` ``` `` occurrences in the raw issue text (e.g. `issue::4548` has exactly one in the whole document) — an actually-unclosed fence in the source, which no stateless open/close regex can pair correctly by construction. Same category of call as Phase 1's decision to stop cataloging canned maintainer replies: bounded, quantified, confined entirely to the noisiest part of the corpus (issues, zero in docs), worth a real parser only if it later proves load-bearing. |
| 2026-07-30 | LLM-based chunking (one of the course's 7 chunking kinds) not implemented — design-record-only, same treatment as ColBERT in the Phase 7 plan | Implement a fifth strategy using an LLM to place chunk boundaries | An LLM call per boundary decision across 3,659 documents is a real cost for a technique whose main advantage — better judgment at ambiguous boundaries — is hard to distinguish from heading-aware's structural approach on a corpus that already has strong, free structural signal (Markdown headings, fenced code). Revisit only if Phase 6 shows all four built strategies plateau below what the golden set demands. |
| 2026-07-30 | Semantic chunker's sentence embedder is dependency-injected (`embed_fn`), defaulting to a lazily-loaded real `fastembed` model | Always load the real model, even in unit tests | Unit tests inject a small deterministic fake embedder so the grouping/breakpoint logic is testable in milliseconds without an ONNX model load; the real model path is exercised by the measurement script instead (which found the ~211–248s cost per large document — see Measured Results). |
| 2026-08-02 | Persisted `data/processed/chunks.jsonl` using heading-aware as a **provisional** chunking strategy | Wait for Phase 6's ablation grid before persisting anything | Phase 3 needs a real chunked corpus to embed before Phase 6 exists to pick a winner empirically. Heading-aware has the best measured code-block integrity (98.4%) and is the only strategy with real heading attribution (Phase 2). Phase 6 re-chunks with all four strategies and can overturn this choice with real retrieval numbers. |
| 2026-08-02 | `sentence-transformers` (not `fastembed`) for the Phase 3 encoders | Keep using `fastembed`/ONNX as in the Phase 0 baseline | Phase 0's ONNX path is a from-scratch download in a different format regardless; `sentence-transformers` is plan.md's named choice for the real bake-off and gives direct MPS device control, which fastembed's ONNX runtime doesn't expose the same way. |
| 2026-08-02 | Embedding cache is a hand-rolled SQLite table keyed on `(model, content_hash)`, not a library like `diskcache` | Add `diskcache` or similar | The need is narrow (two columns, point lookups by a composite key) and Python's stdlib `sqlite3` covers it in ~70 lines with zero new dependencies; a general-purpose cache library would add surface area for a problem this small. |
| 2026-08-02 | Ran `bge-large` and `BGE-M3` embedding **sequentially**, not concurrently, despite being independent jobs | Run both at once to halve wall-clock wait time | CLAUDE.md's risk table flagged M1 16GB memory pressure before this phase started; confirmed real via `vm.swapusage` (~3.5GB in use) during the first (unmitigated) bge-small run. Running two large transformer models on MPS at once was judged too likely to compound it — see the throughput numbers in Measured Results for what the pressure already cost once, unmitigated. |
| 2026-08-02 | HNSW parameter sweep run only on `bge-small`; `bge-large`/`BGE-M3` get one config chosen from that curve, not their own sweep | Sweep all three encoders | The graph algorithm's recall/latency behavior doesn't depend on which encoder produced the vectors; only corpus size and structure matter for that. Sweeping the fast encoder gives a real 36-point grid in seconds; sweeping all three would have added ~an hour for no new signal. |
| 2026-08-02 | ANN recall measured against self-sampled corpus vectors as queries, not real user queries | Wait for Phase 4's golden set before measuring recall at all | Phase 3 precedes Phase 4 in the schedule — no golden queries exist yet. Self-sampled queries measure HNSW's fidelity to the exact nearest-neighbor structure of the embedding space (a real, useful number), explicitly not retrieval quality for real questions (Phase 6's job once Phase 4 exists). |
| 2026-08-02 | Built a score-threshold recall metric (`recall_at_k_score_aware`) instead of trusting ID-set recall | Use standard ID-set recall (`overlap(approx, exact) / k`) | ID-set recall reported a flat, impossible-looking ~0.67 across every HNSW parameter. Root cause: 39.7% of the corpus's vectors are exact duplicates of another vector (largest group: 1,354 issues whose entire chunk is the GitHub issue-template placeholder `_No response_`), so "the" top-10 nearest neighbors isn't uniquely defined — ID-set recall penalizes a correct-but-differently-tied answer. Score-threshold recall (credit any result meeting or beating the true k-th-best score) fixed it: recall became 1.0000 everywhere, which is what should be expected at this corpus scale. |
| 2026-08-02 | `recall_at_k_score_aware` always divides by the requested `k`, never by the number of scores actually returned | Divide by `len(scores)` (the original implementation) | Dividing by the returned count let a method that returns fewer results than asked — e.g. naive post-filtering discarding most of its candidate pool — score a perfect 1.0 as long as the few it did return were correct. Caught because the pre/post-filter experiment produced a self-contradictory result ("short of k results on 100% of queries" alongside "recall=1.0000"), which is only possible if the metric itself was broken. Now regression-tested (`tests/test_knn_recall_score_aware.py`). |
| 2026-08-04 | Chunking axis scored at **document level** (a hit if any retrieved chunk's `doc_id` matches a ground-truth doc), not chunk level | Keep chunk-level scoring everywhere for consistency | Ground truth is heading-aware chunk IDs; fixed/recursive/semantic chunkers each embed their own strategy name in the chunk ID, so those IDs don't exist in the other three strategies' index at all. Chunk-level scoring would silently score every non-heading-aware strategy near zero regardless of actual quality. Confirmed with the user before running (`eval/doc_level.py`, TDD'd). |
| 2026-08-04 | Added `exclude_doc_id` as a first-class parameter on `SparseIndex.search`/`DenseIndex.search`/`HybridRetriever.search` | Wrap it as an eval-only post-hoc filter outside the retrieval interfaces | Golden queries' self-match leak (their own source issue is a trivial near-verbatim hit) is not just a Phase 6 concern — every future eval-dependent phase (7, 10, 11, 16, 17) calls the same retrieval interfaces against the same golden set and would hit the identical bug. Native `must_not` filtering (dense) / over-fetch-then-filter (sparse, no native filter API) belongs in the interface, not duplicated per script. |
| 2026-08-04 | "Lift from baseline" (Phase 6 exit criterion) computed as worst-vs-best **within** the controlled ablation grid (sparse-only vs. winning config), not against Phase 0's literal `aegis_baseline` collection | Use the real historical Phase 0 baseline, since it's the literal artifact plan.md's DoD seems to reference | Checked the actual collection: it indexed 85 of 129 doc pages, raw 500-char slices, and zero of the 3,530 issues — the golden set is mostly issue-sourced. Scoring it would measure "wrong corpus," not "worse retrieval config," a confound rather than a fair, controlled comparison. |
| 2026-08-04 | Contextual enrichment set to **off** in the winning config, reversing plan.md's stated expectation ("usually the cheapest large win") | Ship enrichment on since it's the documented default recommendation | Measured, twice (before and after fixing a real title-duplication bug): enrichment costs 0.4288 → 0.1751 recall@10. Root cause confirmed by inspecting real embedding text: GitHub issue-template boilerplate headings (`Description`, `Question`, `How to do it`) are near-identical across thousands of unrelated issues, and prepending them compresses dense retrieval's discriminative signal — the same boilerplate-pollution failure mode Phases 1-4 each independently hit, here inside enrichment specifically. |

---

## Design Records

| # | Doc | Phase | Status |
|---|---|---|---|
| 00 | `the-baseline-that-fails.md` | 0 | ✅ |
| 01 | `ingestion-and-parsing.md` | 1 | ✅ |
| 02 | `chunking-strategies.md` | 2 | ✅ |
| 03 | `embeddings-distance-and-ann.md` | 3 | ✅ |
| 04 | `building-a-golden-set.md` | 4 | ✅ |
| 05 | `lexical-dense-and-fusion.md` | 5 | ✅ |
| 06 | `retrieval-metrics-and-ablation.md` | 6 | ✅ |
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

**Module 1 (5):** ✅✅⬜✅✅ — topics 1 & 4 done at P0 (baseline demo; cost/latency measurement). Topic 2 (the RAG loop) and topic 5 (IR basics) both land at P5 now that a real retrieval pipeline exists end to end.
**Module 2 (8):** ✅✅✅✅✅✅✅✅ — topics 7, 9, 10, 11 done at P3 (metadata filtering; semantic search & embeddings; distance metrics; contrastive training). Topics 6, 8, 12 done at P5: three retrieval techniques (dense + BM25 + metadata prefiltering, one interface), bag-of-words→TF-IDF→BM25 (`bm25s` + the custom analyzer), hybrid search & RRF (built, tested, compared against linear fusion). Topic 13 (retrieval eval metrics — Precision/Recall/MAP@k/MRR) done at P6: `ranx`, verified against a hand-computed example, then used for the full ablation grid. **Module 2 complete.**
**Module 3 (6):** ✅✅✅⬜⬜⬜ — topics 14, 15 done at P3 (KNN→ANN→HNSW: brute-force ground truth + 36-config recall/latency sweep; vector databases: Qdrant, named vectors, payload indexes). Topic 16 done at P2.
**Module 4 (9):** ⬜⬜⬜⬜⬜⬜⬜⬜⬜
**Module 5 (9):** ✅⬜⬜⬜⬜⬜⬜⬜⬜ — topic 29 (what makes production challenging) exercised at P1, reinforced at P6: three real production bugs (thread-unsafe cache, an enrichment-prefix bug, a self-match eval leak) found only by running the full pipeline against real golden-set queries, none catchable by unit tests alone.

**Progress: 16 / 37**

---

## Build vs Import Ledger

Tracks what's actually implemented from scratch vs. imported, so the ratio stated in interviews matches reality. Updated as components land. Full rationale in `plan.md` §1 / `CLAUDE.md` §2.

| Component | Plan | Actual | Phase |
|---|---|---|---|
| Custom analyzer/tokenizer | Build | Build (75 lines) | 5 |
| BM25 | Import (`bm25s`) | Import (`bm25s`, wired to the custom analyzer) | 5 |
| Heading-aware chunker | Build | Build (132 lines, matches the ~150-line estimate) | 2 |
| RRF fusion | Build | Build (83 lines, `fusion.py` — RRF + linear fusion together) | 5 |
| Retrieval metrics | Import (`ranx`) | Import (`ranx`, verified against a hand-computed 5-doc example) | 6 |
| Cross-encoder reranker | Import | — | 7 |
| Confidence scorer | Build | — | 10 |
| Citation validator | Build | — | 9 |
| Routing policy | Build | — | 8 |
| Eval harness | Build | Build (~150 lines, `harness.py` + `doc_level.py` — config-hashed, cached, per-stratum) | 6 |
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
| 5 | Semantic chunking costs ~211–248s to embed a single large document's sentences (measured on `release-notes.md`, confirmed compute-bound) — does it clear a large enough retrieval-quality bar in the Phase 6 ablation to justify that cost, or does it join the cut list? | 2026-07-30 | Phase 6 | **Resolved 2026-08-04** — measured on the full corpus for the first time: 723s to chunk + 640s to embed, and doc-level recall@10 of 0.2571 vs. recursive's 0.2436 (a ~5% relative gain for materially more compute) — both dwarfed by heading-aware's 0.5470. Semantic chunking is cut from the winning config. |
| 6 | 39.7% of the corpus's embeddings are exact duplicates of another chunk's, the largest group being 1,354 issues whose entire chunk text is the GitHub issue-template placeholder `_No response_` (Phase 3). Degenerate/near-empty chunks like these could surface as noisy near-duplicate results for many unrelated queries. Should they be filtered out of the indexed corpus, or does the confidence scorer (Phase 10) / reranker (Phase 7) handle it downstream without special-casing? | 2026-08-02 | Phase 4 (golden set shouldn't be built from these) / Phase 6 | Open — Phase 6 confirmed the broader pattern (GitHub issue-template boilerplate headings actively hurting contextual enrichment, see Decision Log 2026-08-04) but didn't specifically check this duplicate-chunk question; still open for Phase 7/10 |
| 7 | The eval harness's cache key (`config_hash + queries_hash`) doesn't capture "the underlying Qdrant collection for this config was regenerated" — re-running an unchanged `RetrievalConfig` after rebuilding its index silently serves the stale cached result (hit once in Phase 6 after fixing the enrichment-prefix bug; worked around by manually deleting the affected cache file, not fixed generally). | 2026-08-04 | Phase 7+ (any phase that rebuilds an index without changing a config field) | Open — add a data-version field to `RetrievalConfig` if this recurs, or document "clear `data/cache/eval/` after any index rebuild" as a standing rule |

---

## Session Log

Newest first. One entry per working session.

### 2026-08-04 — Phase 6: Retrieval Evaluation & the Ablation Grid

**Context.** Continued from Phase 5 (session picked up mid-work after a gap — Docker had stopped and needed restarting; the eval harness and `EmbeddingCache` thread-safety fix from before the gap were already in place and verified). Goal: `ranx` metrics, a config-hashed eval harness, and the full chunking × enrichment × retrieval-strategy × encoder ablation grid against the golden set, per plan.md's never-cut Phase 6.

**Done**
- Verified `ranx` against a hand-computed 5-document example (exact match on Precision@5/Recall@5/MRR/MAP@5/NDCG@5); found and normalized a real library quirk (`evaluate()` returns a bare scalar for one metric, a dict for multiple).
- Built the eval harness (`eval/harness.py`): `RetrievalConfig` (frozen dataclass, hash excludes the human-readable `name`), `run_eval` (parallelized, per-stratum, cached by `config_hash + queries_hash` — deliberately not config hash alone, since a same-config-different-queries collision was caught by a dedicated test before shipping).
- **Found and fixed a real threading bug** while running the harness for real the first time: `EmbeddingCache`'s single `sqlite3.Connection` isn't safe across the harness's `ThreadPoolExecutor` workers. TDD'd a reproducing test, fixed with `check_same_thread=False` + a lock.
- Ran the free axes first (retrieval strategy, encoder — both against already-indexed Phase 3 collections, zero new compute). **Found hybrid fusion (RRF/linear) underperforming pure dense** at default equal weights; investigated, tuned weights and `k` (dense_weight sweep, retrieved each query's candidate pool once and re-fused in-memory rather than re-querying per config), confirmed hybrid only ever *approaches* dense's performance as `dense_weight → ∞`, never exceeds it — true in aggregate and in every one of the six strata, not just on average.
- **Found and fixed a second real bug**: `enrichment_prefix` (Phase 2 code) duplicated the document title verbatim for every issue chunk, because `github_issues.py` prepends the title as the doc's own H1 and `heading_path` already starts with it. TDD'd a regression test, fixed. Effect was small (recall@10 0.1364 → 0.1454) — the larger cause turned out to be structural: GitHub issue-template boilerplate headings (`Description`, `Question`, `How to do it`) shared across thousands of unrelated issues compress dense retrieval's discriminative signal when prepended. Contextual enrichment measured as a clear net negative on this corpus (0.4288 → 0.1751 recall@10) — the opposite of plan.md's stated default expectation, with a real mechanistic explanation, not just a number.
- **Found and fixed the biggest bug**: every golden query's own source issue is indexed in the corpus and is a trivial near-verbatim top-1 match — Phase 4 already solved this exact problem for candidate labeling (`must_not` filter) but it was never carried into Phase 6's actual evaluation. Silently inflated every measurement in this phase's first pass (most visibly MRR: dense 0.616 → 0.864 once fixed). Added `exclude_doc_id` as a first-class parameter on `SparseIndex`/`DenseIndex`/`HybridRetriever.search` (native Qdrant `must_not` for dense, over-fetch-then-filter for sparse), then re-ran every prior measurement.
- Built document-level ground-truth transfer (`eval/doc_level.py`, TDD'd) to compare chunking strategies fairly — their chunk IDs live in disjoint spaces, only `doc_id` is stable across all four chunkers.
- Chunked and embedded the full corpus with the two strategies never persisted before (`experiments/06a_prepare_ablation_indexes.py`): fixed (8,984 chunks, 830s), recursive (9,561 chunks, 741s), semantic (9,168 chunks, 723s chunking + 640s embedding — the first full-corpus semantic-chunking run ever attempted in this project, resolving Open Question #5). Also re-embedded heading-aware with enrichment on (twice — once before, once after the title-duplication fix).
- Ran the full grid. Winner: heading-aware chunking, enrichment off, dense retrieval, `bge-large-en-v1.5`. Wrote `docs/results/retrieval-ablation.md` and the design record.

**Exit criteria — all met**
- `docs/results/retrieval-ablation.md` with the deliverable table ✓
- Winning config selected with written reasoning ✓
- Recall@10 and MRR lift from baseline stated as numbers ✓ (+0.332 / +0.496 vs. worst measured config — Phase 0's literal baseline isn't corpus-comparable, see results doc §7)
- Per-stratum breakdown showing where it's weak ✓ (dense wins/ties every stratum; sparse weakest on `version_specific`, counter-intuitively)
- Harness reruns any config from a hash ✓ (`load_cached`)

**Surprised by:** three real production bugs in one phase, each a different mechanism (thread-safety, a data-duplication bug in text construction, a methodology gap not carried forward from an earlier phase) — all invisible to unit tests because none of them involve running the full pipeline against real golden-set queries, which is the one thing this phase does for the first time. Hybrid search *not* beating dense, in aggregate and in every stratum, on a golden set that's mostly natural-language questions rather than literal identifiers (contrast with Phase 5's deliberate `ECONNREFUSED` spot-check, which is exactly the query shape where BM25 does win). Contextual enrichment being a clear net negative — plan.md's "usually the cheapest large win" assumption held for the docs corpus it was probably written for, not for this one, which is 83% GitHub issues with template boilerplate. Semantic chunking finishing in ~23 minutes total, not the "multi-hour job" Phase 2 worried about — the concern scaled from one large document's cost, but most of the corpus is short issues.

**Next**
- Phase 7: reranking. Cross-encoder (`bge-reranker-v2-m3`) on top of this phase's winning config; measure the pool-size vs. latency tradeoff and the bi-encoder-vs-cross-encoder gap directly. The sparse-side `doc_type`-filtering gap (Phase 5 design record) and Open Question #6 (duplicate `_No response_` chunks) remain open — worth a look if per-stratum weakness shows up again once reranking is in place. Open Question #7 (harness cache doesn't detect index regeneration) is a standing risk for any phase that rebuilds a collection without changing a config field.

### 2026-08-03 — Phase 5: Retrieval Core — Analyzer, Sparse, Dense & Fusion

**Context.** New session, continued from Phase 4. Goal: the actual retrieval pipeline Phase 6 will measure — custom analyzer, BM25 (`bm25s`), dense retrieval with metadata pre-filtering, RRF and linear fusion, and one composed interface across all four strategies. `analysis/` and `retrieval/` were empty stubs at session start.

**Done**
- TDD'd the custom analyzer (`analysis/analyzer.py`, 75 lines): one rule applied uniformly — a compound identifier (`snake_case`, `dotted.path`, `SCREAMING_CASE`, `--flag`) is emitted whole *and* decomposed into parts, except version/IP-style numeric sequences (`0.100.0`, `127.0.0.1`), kept whole only. Test suite built from 40 real tokens pulled from `documents.jsonl` (not invented), per the exit criterion.
- **Found and fixed a real regex bug during RED→GREEN**: `_TOKEN_RE`'s `--?` was meant to mean "0-2 leading dashes" but actually requires exactly one mandatory dash — every plain word in the corpus silently vanished from tokenization, only `--flag`-style tokens survived. 46 of the first 56 tests failed at once; fixed to `-{0,2}`.
- Built `SparseIndex` (`retrieval/sparse.py`): `bm25s` (imported) wired to the custom analyzer instead of its own default tokenizer. TDD'd with 5 tests, including the literal `ECONNREFUSED` exit-criterion case.
- Built `DenseIndex` (`retrieval/dense.py`): wraps Phase 3's Qdrant collections with native metadata pre-filtering (`product_area`, `doc_type`), reusing Phase 3's own finding that pre-filtering (not post-filtering) is required for reliable recall. Chunks are hydrated from the in-memory corpus by ID, not reconstructed from Qdrant's payload (which lacks `content_hash`/`embedding_text` by Phase 3 design) — verified against the live `aegis_chunks_bge_small` collection (19,236 points).
- TDD'd RRF and linear fusion together (`retrieval/fusion.py`, 83 lines, 12 tests): `score(d) = Σ weight_r / (k + rank_r(d))` for RRF; per-list min-max normalization + weighted sum for linear, with a divide-by-zero guard for degenerate equal-score lists.
- TDD'd `HybridRetriever` (`retrieval/hybrid.py`, 70 lines, 8 tests, fake indexes as spies): one `.search(query, strategy)` interface over `dense`/`sparse`/`hybrid_rrf`/`hybrid_linear`, each side pulling `max(top_k, candidate_pool_size)` before fusing so fusion has breadth to find cross-method overlap in.
- Ran the full pipeline against the real 19,236-chunk corpus: BM25 puts the real `ECONNREFUSED` issue at rank 1 (score 3.1467); dense search doesn't find it in the top 200 at all. Confirmed the analyzer, `SparseIndex`, `DenseIndex`, and `HybridRetriever` all work end-to-end together, not just in isolation.
- Design record: `docs/explainers/05-lexical-dense-and-fusion.md` — the analyzer's decomposition rule, RRF/linear fusion derivations, four rejected alternatives (Qdrant native sparse, payload-only chunk reconstruction, stemming/stopwords, camelCase splitting), and the regex bug as a "what surprised me."

**Exit criteria — all met**
- Analyzer tests green on 40 real corpus tokens ✓ (56/56 total)
- BM25 returns sane results for `ECONNREFUSED`-style queries where dense fails ✓ (measured: rank 1 vs. absent from top 200)
- RRF and linear fusion both implemented and unit-tested ✓ (12 tests)
- All retrieval modes behind one interface ✓ (`HybridRetriever` + `RetrievalStrategy`)

**Surprised by:** the regex bug (`--?` vs. `-{0,2}`) failing loud and immediately rather than subtly — a reminder that testing against real corpus tokens catches classes of bugs a hand-picked example wouldn't (a synthetic test with only `--reload`-style examples would have passed while silently dropping every plain word). Unfiltered hybrid retrieval is still issue-dominated even after fusing two rankings, because fusing two individually issue-heavy lists doesn't fix the underlying imbalance — `HybridRetriever` doesn't yet forward `doc_type` filtering to the sparse side, logged as a Phase 6 watch item. RRF's `k=60` compressing the gap between nearby top ranks more than expected — a test asserting "cross-method agreement beats a single #1" needed the single-list rank gap widened to ~10 before it held, which says something real about what `k` actually controls.

**Not measured yet (Phase 6's job):** `k1`/`b` numeric sweep against the golden set; Recall@10/MRR comparison across `dense`/`sparse`/`hybrid_rrf`/`hybrid_linear`; whether RRF or linear fusion actually wins on this corpus. All the relevant knobs (`bm25_k1`, `bm25_b`, `rrf_k`, `rrf_dense_weight`, `rrf_sparse_weight`, `hybrid_candidate_pool_size`) are in `config.py`, not hardcoded, specifically so Phase 6 can sweep them without touching this code.

**Next**
- Phase 6: the ablation grid. `ranx` metrics, the config-hashed eval harness, and the actual sweep (chunking × enrichment × retrieval strategy × encoder) against the golden set's 180 dev queries, reported per-stratum. This is the phase that turns Phase 5's "it works" into "here's the number, and here's why this config." Also worth checking there: whether the sparse-side `doc_type` filtering gap (see Surprised by) shows up as a measurable per-stratum weakness.

### 2026-08-03 — Phase 4: The Golden Evaluation Set

**Context.** Continued from Phase 3. Goal: harvest 180+ labeled queries (≥20 per stratum) from FastAPI's 2,204 resolved issues, combine with 40 negatives, freeze and hash. Phase 4 is the most critical phase in the schedule — Phases 5–11 need a fixed benchmark, and every claim downstream depends on numbers measured against this set.

**Done**
- Implemented stratification heuristic (`strata.py`): 6 categories (how-to, error/debugging, conceptual, version-specific, multi-hop, ambiguous), priority-ordered with version/error/multi-hop detecting strongest signals first, fallback to conceptual/how-to cues, ambiguous as catch-all. TDD'd with 10 real corpus examples.
- Measured real stratum distribution (2,109 post-low-signal-filter): error_debugging is 52% of corpus (1,096), conceptual is the smallest (41, 1.9%), multi-hop is larger than expected (13.3%).
- Built harvest (`harvest.py`): load resolved issues, filter out ~4.3% with non-substantive resolutions (pure redirects, canned replies, ≤6 words), stratify, sample. Includes low-signal detection: bare "thanks" or ≤30 words + explicit redirect pattern.
- Hand-authored 40 negatives (`negatives.py`, 10 per category): out_of_corpus (Django, Flask, etc.), billing_legal (refunds, accounts, legal), unanswerable (opinion, roadmap, guarantees), adversarial/injection (prompt injection attempts for Phase 15).
- Ran retrieval (`retrieve_candidates.py`) on 333 candidates (2x buffer per stratum + 40 negatives). **Found boilerplate-pollution failure mode in dense search:** unfiltered top-20 was 83.6% issue chunks and 30.9% of queries got zero doc chunks — same GitHub issue-template boilerplate problem Phases 1–3 each hit independently. Fixed: split into 12 doc + 8 issue filtered searches instead. Result: 100% of queries now guaranteed documentation representation.
- **Found and blocked a self-match leak** (smoke test): a query's own source issue scored 0.958 (vs. 0.7–0.8 for real answers), trivially dominating every retrieval result. This would have made circular ground-truth selection possible, inflating Phase 6's numbers. Added `must_not` filter to exclude source_issue from retrieval.
- Labeled all 180 positives programmatically (Haiku + token-overlap heuristic, ≥15% query-token overlap in chunk text = relevant). Conservative, deterministic, auditable. Phase 11 will independently re-label via Opus and measure Cohen's κ agreement.
- Froze the golden set (`freeze_golden_set.py`): combined labeled positives + negatives, applied holdout split (5 pos/stratum + 10 neg, ~13.5%), hashed both splits (SHA256 on sorted lines). Frozen files: `golden_set_v1.jsonl` (dev, 180 queries) and `golden_set_holdout_v1.jsonl` (holdout, 40 queries).
- Design record: `docs/explainers/04-building-a-golden-set.md` — methodology, failure modes (boilerplate-pollution, self-match leak), why stratification matters, why Phase 4 must be benchmarked before optimization.

**Exit criteria — all met**
- 180 labeled queries, 30 per stratum ✓
- 40 negatives with expected behaviors ✓
- Every query has ≥1 verified ground-truth chunk (measured: per-stratum averages 2–4 chunks selected per query depending on stratum)
- Frozen, committed, hashes recorded in progress.md ✓
- Holdout split sealed ✓

**Surprised by:** boilerplate-pollution hit retrieval harder than expected (30.9% of queries left doc-less despite docs being 11% of corpus), revealing it's a systemic failure mode across multiple phases, not a one-off. The self-match leak was caught via smoke-test before going live — a reminder to always run the tool end-to-end once before delegating. Conceptual queries being genuinely rare (1.9% of corpus) — they'll be the hardest stratum to measure and improve on.

**Next**
- Phase 5: Retrieval core (custom analyzer, BM25, dense search, RRF fusion). Build the retrieval pipeline; Phases 5 and 6 ship together (implementation + measurement).

---

### 2026-08-02 — Phase 3: Embeddings, distance & the vector store

**Context.** New session, continued from Phase 2. Goal: embed and index the full
corpus with real candidate encoders, prove cosine≡dot on real vectors, measure
HNSW's recall/latency tradeoff and pre- vs. post-filter metadata filtering — all
per plan.md's exit criteria. Started by persisting the first real chunked corpus
(`data/processed/chunks.jsonl`, heading-aware, provisional — Phase 2 deliberately
never wrote one), since a user question ("where are the chunks stored?") surfaced
that the project had built four chunking strategies but never actually committed
one to disk.

**Done**
- Environment: started Docker/Qdrant (had stopped), added `torch` +
  `sentence-transformers` (MPS confirmed available), added `matplotlib` for the
  required recall-vs-latency plot
- Built, TDD'd: `EmbeddingCache` (SQLite, keyed on `(model, content_hash)`),
  `Encoder` (sentence-transformers wrapper, dependency-injectable for fast unit
  tests, cache-aware de-duplication verified with a call-counting fake), brute-
  force exact k-NN plus two recall metrics
- Built `vector_store.py` (Qdrant collections, named `dense` vector, payload
  indexes on `product_area`/`doc_type`/`doc_version`) and the embed-and-index
  orchestration script
- Embedded and indexed the full 19,236-chunk corpus with all three candidate
  encoders (`bge-small` dev workhorse, `bge-large-en-v1.5`, `BGE-M3`) into three
  separate Qdrant collections
- **Hit real M1 16GB memory pressure** on the first (bge-small) full run —
  1,517s actual vs. ~163s predicted from a clean 200-chunk sample, confirmed via
  `vm.swapusage` (~3.5GB swapped) and a specific findable trigger (a batch
  spanning a legitimate 2,830-token atomic-code-block chunk forcing the whole
  batch to pad to 512 tokens). Fixed with sort-by-length batching, a smaller
  batch size for the two large models, and periodic `torch.mps.empty_cache()`
  — `bge-large` (665s) and `BGE-M3` (799s) both then finished *faster in
  wall-clock time* than bge-small's unmitigated run, despite being much bigger
  models
- Proved cosine≡dot on 2,000 real chunk-pair vectors (max abs diff 1.79e-07,
  not a toy example)
- Ran the HNSW sweep (36 configs: m×ef_construct×ef_search) against
  brute-force KNN ground truth. **Found ID-set recall was structurally broken**
  on this corpus — flat ~0.67 regardless of parameters, which is impossible.
  Root cause: 39.7% of the corpus's vectors are exact duplicates (the largest
  group: 1,354 issues whose entire chunk is the GitHub issue-template
  placeholder `_No response_` — the same boilerplate-pollution category Phase 1
  and Phase 2 each independently hit). Built a score-threshold recall metric
  that credits any tied neighbor; corrected recall was 1.0000 across all 36
  configs — HNSW's approximation costs nothing at this corpus scale
- Measured pre-filter vs. post-filter metadata filtering across 5 `product_area`
  categories. **Found a second real bug**, in the score-threshold metric just
  built: it divided by the number of results *returned* instead of *requested*,
  so a method returning too few results could still score a perfect 1.0 — caught
  because the experiment's own output was self-contradictory ("short of results
  on 100% of queries" next to "recall=1.0000"). Fixed and regression-tested.
  Corrected result: naive post-filtering scores 0.1667–0.4033 recall depending
  on category (even `general`, 26% of the corpus, comes back short 90% of the
  time); native pre-filtering scores 1.0000 on every category tested
- Design record: `docs/explainers/03-embeddings-distance-and-ann.md`, including
  a §3a on contrastive training (topic 11) — cosine is meaningful here because
  these specific encoders are trained to make it so, not by default

**Exit criteria — all met**
- All chunks embedded and indexed; cache confirmed working (full re-embed after
  a cold run: 0.9s, 100% cache hits)
- `docs/results/ann-recall-vs-latency.md` with a real plot
  (`ann-recall-vs-latency.png`)
- Pre-filter vs. post-filter difference measured
  (`docs/results/prefilter-vs-postfilter.md`)
- Both candidate encoders (`bge-large-en-v1.5`, `BGE-M3`) indexed, alongside the
  dev workhorse

**Surprised by:** two real bugs, both in measurement code I had just written, not
the system under test — and both caught the same way: noticing a result that
couldn't possibly be true (flat recall that doesn't respond to a search
parameter; "short of results" next to "perfect recall") rather than writing it
down. The M1 memory-pressure risk CLAUDE.md already named turned out to be
exactly right, and took an actual 25-minute bad run to be believed. HNSW recall
saturating at 1.0 regardless of parameters at ~19K points — the tradeoff curve
plan.md anticipated doesn't exist yet at this scale. The pre/post-filter gap
being large even for the corpus's single biggest category, not just narrow ones.

**Next**
- Phase 4: the golden evaluation set. 180 labeled queries + 40 negatives,
  harvested from closed issues with maintainer resolutions (2,204 available,
  Phase 1). This replaces Phase 3's self-sampled-query methodology with real
  query relevance for everything downstream. Open question #6 (the "_No
  response_" duplicate-boilerplate chunks) should get a first real signal once
  golden-set labeling is underway — worth checking whether they ever surface as
  spurious top-k results before building a special-case filter for them.

### 2026-07-30 — Phase 2: Chunking laboratory

**Context.** New session, continued from Phase 1. Goal: four chunking strategies
behind one config-swappable interface, benchmarked against each other on the real
Phase 1 corpus — no winner declared yet (that needs Phase 4's golden set and
Phase 6's ablation grid).

**Done**
- Built shared infrastructure first, TDD'd: `tokens.py` (token counting via
  `tiktoken`/`cl100k_base`, plus a byte-level-BPE safe-boundary check —
  slicing an arbitrary token window can silently split a multi-byte
  character, and `tiktoken.decode()` replaces it with `�` instead of
  raising), `segments.py` (text/code segment splitting, fenced blocks kept
  atomic), `merge.py` (greedy pack-with-overlap), `base.py` (the `Chunker`
  protocol + contextual-enrichment helper)
- Built and TDD'd all four strategies: fixed-size (token sliding window),
  recursive (paragraph → sentence → word descent), heading-aware (splits on
  Markdown headings, code blocks atomic — the flagship, ~132 lines matching
  plan.md's estimate), semantic (sentence embeddings, cosine-distance
  breakpoints, dependency-injected embedder for fast unit tests)
- Wired contextual enrichment (`{title} > {heading_path} > {version}`,
  toggleable, stored in a separate `Chunk.embedding_text` field so citations
  keep showing real source text) and a config-driven `build_chunker()`
  factory — the "one interface, config-swappable" exit criterion
- Ran the measurement script (`experiments/02_chunking_stats.py`) against
  129 docs + a seeded issue sample + issue #2266, then found and fixed
  **three real bugs**, all surfaced only by measuring the full corpus (unit
  tests use small clean inputs by construction): (1) `recursive_split_pieces`
  and `greedy_merge` each inserted their own separator at a merge boundary,
  doubling it and inflating real chunk token counts past `max_tokens`
  (issue #15000, 443/400); (2) even after that fix, `greedy_merge`'s
  sum-of-parts token accounting still didn't match the true joined text's
  token count (BPE isn't additive across concatenation) — redesigned to
  verify by actually joining and re-counting before accepting each piece;
  (3) the fenced-code-block regex required a closing line to be *only*
  backticks, which real issue text violates twice over (CRLF line endings,
  a closing fence sharing its line with trailing HTML) — either one silently
  swallowed every heading and paragraph after it into one giant code segment
  (issue #15000 went from 2 detected sections to the correct 9 once fixed)
- After fixing all three: 0/8,984 fixed-strategy chunks and 0/9,561
  recursive-strategy chunks exceed `max_tokens` over the *full* 3,659-document
  corpus (no sampling). Heading-aware: 427/19,236 exceed it — 420 are the
  documented atomic-oversized-code-block exception, 7 are an accepted
  residual limitation (documents with a genuinely odd count of literal
  `` ``` `` in the raw text — an actually-unclosed fence in the source)
- Corrected a real overclaim before it shipped: `release-notes.md` and issue
  #2266 (Phase 1's flagged stress cases) were assumed to trigger the
  atomic-oversized-code exception. Measured result: neither does, at
  `max_tokens=400`. Found the real example instead —
  `doc::advanced/security/oauth2-scopes.md`, a 184-line/1,352-token example
  script included via a Phase 1 `{* *}` directive and repeated across 8 of
  27 sections
- Design record: `docs/explainers/02-chunking-strategies.md`

**Exit criteria — all met**
- Four strategies, one interface, config-swappable (`build_chunker(strategy)`)
- Tests: no chunk exceeds `max_tokens` (fixed/recursive: 0 violations,
  full corpus; heading-aware: only the documented atomic exception + the
  quantified residual limitation, both explained, neither silent); heading-aware
  never splits a fenced block (code-block integrity 98.4%–99.1% for
  heading-aware/semantic, the shortfall from the same residual limitation)
- Contextual enrichment toggleable (`ChunkingSettings.contextual_enrichment`)
- `docs/results/chunking-stats.md` — chunk count, token distribution,
  code-block integrity rate, wall time per strategy, plus a stress-case
  spotlight table

**Surprised by:** three real, corpus-driven bugs that no unit test caught because
unit tests use small clean inputs — found only by actually measuring the full
corpus, which is the whole reason Phase 1's "actually open the parsed output"
discipline exists. Semantic chunking's per-sentence embedding cost is large and
real (~211–248s for one big document, confirmed compute-bound not overhead)
independent of whether it wins on retrieval quality. The two documents Phase 1
flagged as the corpus's stress cases turned out not to be this phase's stress
case at all — a genuine measurement-contradicts-expectation moment, logged
rather than quietly adjusted away.

**Next**
- Phase 3: embeddings, distance & the vector store. Embed `Chunk.embedding_text`
  (not `.text`) into Qdrant; bake off `bge-large-en-v1.5` vs `BGE-M3`; brute-force
  KNN as ground truth for the HNSW recall/latency curve. Open question #5 (is
  semantic chunking worth its cost) carries forward to Phase 6, not Phase 3.

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
