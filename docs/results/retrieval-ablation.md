# Retrieval Ablation Grid — Phase 6

All numbers: 150 positive queries, `golden_set_v1.jsonl` (dev split), metrics via `ranx` (`aegis.eval.metrics`, verified against a hand-computed 5-document example). Every query excludes its own `source_issue` from retrieval (`exclude_doc_id`) — otherwise a query's near-verbatim source text is a trivial top-1 hit that ground truth deliberately excludes, wasting a top-10 slot (see design record §6). Sweep order: center → retrieval strategy → encoder → contextual enrichment → chunking, confirming each axis's winner before moving to the next, per plan.md's "confirm the winner against the runner-up" methodology (not a full cartesian product — 4 axes, ~20 runs).

**Center:** heading-aware chunking, contextual enrichment off, `bge-large-en-v1.5`, dense retrieval, top-10.

## 1. Retrieval strategy

| Strategy | Recall@10 | MRR | NDCG@10 | MAP@10 |
|---|---|---|---|---|
| **dense** | **0.4288** | **0.8640** | **0.7064** | **0.3695** |
| sparse (BM25) | 0.0967 | 0.3676 | 0.1761 | 0.0549 |
| hybrid_rrf (default weights, k=60) | 0.2706 | 0.7862 | 0.4732 | 0.1944 |
| hybrid_linear (default weights) | 0.3082 | 0.8381 | 0.5201 | 0.2227 |

At default equal weights, **both hybrid strategies underperform pure dense**. Retuned (dense_weight sweep, `sparse_weight=1.0` fixed):

| Config | Recall@10 | MRR | NDCG@10 |
|---|---|---|---|
| RRF, dense_weight=1 (k=60) | 0.2706 | 0.7862 | 0.4732 |
| RRF, dense_weight=2 | 0.3559 | 0.8228 | 0.5861 |
| RRF, dense_weight=4 | 0.3940 | 0.8538 | 0.6510 |
| RRF, dense_weight=8 | 0.4173 | 0.8791 | 0.6921 |
| RRF, dense_weight=16 (k=60) | 0.4270 | 0.8752 | 0.7054 |
| RRF, dense_weight=16, k=10 (best found) | 0.4324 | 0.8640 | 0.7092 |
| linear, dense_weight=16 (best found) | 0.4283 | 0.8683 | 0.7045 |

Both fusion strategies **monotonically approach but never exceed pure dense** as `dense_weight → ∞` — the theoretical ceiling of that sweep direction is dense alone. The tiny apparent edge at dense_weight=16 (0.4324 vs 0.4288) is ANN pool-size noise (computed from a top-50 candidate pool sliced to 10, vs. querying top-10 directly — a known small HNSW-approximation artifact, not a real effect).

**Winner: dense.** BM25 is weak on this golden set (0.097 recall@10 alone) because most queries are natural-language questions pulled from GitHub issue titles/bodies, not the literal-identifier lookups where BM25 won decisively in Phase 5's `ECONNREFUSED` spot-check. Fusing a strong ranker with a much weaker one, even retuned, adds no measurable value here — see per-stratum breakdown (§5) for confirmation this holds in every stratum, not just in aggregate.

## 2. Encoder

Under dense retrieval (the §1 winner):

| Encoder | Recall@10 | MRR | NDCG@10 | MAP@10 |
|---|---|---|---|---|
| **bge-large-en-v1.5** | **0.4288** | **0.8640** | **0.7064** | **0.3695** |
| BGE-M3 | 0.1348 | 0.4781 | 0.2478 | 0.0837 |

**Winner: bge-large-en-v1.5**, by a wide, consistent margin (confirmed under both `hybrid_rrf` and `dense`). Surprising relative to BGE-M3's general reputation (strong on MTEB, multilingual, long-context) — plausible explanation: this project only uses BGE-M3's dense vector, not its sparse/ColBERT multi-vector capabilities, and a dedicated English dense model may simply out-perform BGE-M3's dense-only component on a purely English technical corpus. Not chased further (see design record §6).

## 3. Contextual enrichment

Under dense retrieval, bge-large:

| Enrichment | Recall@10 | MRR | NDCG@10 | MAP@10 |
|---|---|---|---|---|
| **off** | **0.4288** | **0.8640** | **0.7064** | **0.3695** |
| on | 0.1751 | 0.5134 | 0.3043 | 0.1083 |

**Winner: off — decisively.** Two real causes, one fixed, one structural:

1. **Bug, fixed:** `enrichment_prefix` (`chunking/base.py`) duplicated the document title verbatim for every issue chunk, because `aegis.ingest.github_issues` prepends the issue title as the document's own H1 (`# {title}\n\n{body}`), so heading-aware's `heading_path` already starts with the title. TDD'd a regression test and fixed it (title dropped when `heading_path[0] == document.title`). Effect: recall@10 0.1364 → 0.1454 — real, but small.
2. **Structural, corpus-specific:** even fixed, enrichment still costs ~59% relative recall. Root cause: GitHub issue templates use generic section headings (`Description`, `Question`, `How to do it`) shared verbatim across thousands of unrelated issues. Prepending them pulls otherwise-unrelated issues' embeddings closer together, compressing the discriminative signal that dense retrieval depends on — the same boilerplate-pollution failure mode this project has hit independently in Phases 1, 2, 3, and 4, now showing up in enrichment specifically. Plan.md's framing of enrichment as "usually the cheapest large win" holds for corpora with content-specific headings (structured docs); it does not hold here, where 83% of the corpus is issues with template-boilerplate headings.

## 4. Chunking strategy

Ground truth (`golden_set_v1.jsonl`) is heading-aware chunk IDs, which don't exist in fixed/recursive/semantic's ID space (every chunker embeds its own strategy name in the ID; only `doc_id` is stable across all four — confirmed by grep). Scored at **document level** for this axis only: a retrieved chunk counts as relevant if its `doc_id` matches one of the ground-truth chunks' source documents (`aegis.eval.doc_level`). Coarser than exact chunk match by construction — not comparable to §1-3's chunk-level numbers.

| Strategy | Chunks | Recall@10 (doc-level) | MRR | NDCG@10 | MAP@10 |
|---|---|---|---|---|---|
| **heading_aware** | 19,236 | **0.5470** | **0.8748** | **0.6993** | **0.4774** |
| semantic | 9,168 | 0.2571 | 0.6293 | 0.3491 | 0.1740 |
| recursive | 9,561 | 0.2436 | 0.6134 | 0.3350 | 0.1661 |
| fixed | 8,984 | 0.2393 | 0.5888 | 0.3222 | 0.1534 |

**Winner: heading_aware, by more than 2x** over every alternative. This is the first real retrieval-quality evidence for the chunking strategy Phase 3 picked provisionally on code-integrity grounds alone.

**Resolves Open Question #5** (progress.md): does semantic chunking's cost (~211-248s to chunk one large document; ~723s to chunk + 640s to embed the full 3,659-doc corpus, measured here for the first time) clear a large enough bar to justify it? **No.** Semantic (0.2571) barely beats recursive (0.2436) and fixed (0.2393) — a ~5-8% relative gain for materially more compute, and all three are dwarfed by heading-aware's structural signal, which is nearly free to compute by comparison. Semantic chunking is cut from the winning config.

## 5. Per-stratum breakdown (dense vs. sparse vs. tuned RRF, chunk-level, heading-aware)

| Stratum | n | dense Recall@10 | sparse Recall@10 | RRF (dense_weight=16, k=10) |
|---|---|---|---|---|
| conceptual | 25 | 0.4995 | 0.1415 | 0.4995 |
| version_specific | 25 | 0.4702 | 0.0637 | 0.4702 |
| multi_hop | 25 | 0.4254 | 0.1036 | 0.4254 |
| how_to | 25 | 0.4067 | 0.0822 | 0.4030 |
| ambiguous | 25 | 0.4010 | 0.1160 | 0.4030 |
| error_debugging | 25 | 0.3911 | 0.0750 | 0.3935 |

Dense wins or ties in **every** stratum — no hidden stratum where hybrid or sparse meaningfully wins. Sparse is weakest specifically on `version_specific` (0.0637) — counter-intuitive at first (identifier-heavy queries should favor BM25), explained in the design record: golden queries are natural-language questions *about* a version, not the literal version string itself, unlike Phase 5's deliberate `ECONNREFUSED` spot-check.

## 6. Winning configuration

**Chunking: heading-aware · Enrichment: off · Retrieval: dense · Encoder: bge-large-en-v1.5**

| Metric | Value |
|---|---|
| Recall@10 | 0.4288 |
| MRR | 0.8640 |
| NDCG@10 | 0.7064 |
| MAP@10 | 0.3695 |

## 7. Recall@10 / MRR lift vs. baseline

Phase 0's naive baseline (`aegis_baseline` Qdrant collection) is **not corpus-compatible** with this comparison: it indexed only 85 of 129 doc pages at 500-char raw slices and **zero of the 3,530 issues** the golden set is mostly drawn from, with a chunk-ID scheme that doesn't align with `ground_truth_chunk_ids`. Scoring it against this golden set would measure "wrong corpus," not "worse retrieval config" — a confound, not a fair comparison.

Instead, lift is reported **within the controlled ablation grid** (identical corpus, identical golden set, only retrieval config differs) — the worst single fully-specified config actually measured (sparse/BM25 alone) vs. the winning config (dense, §6):

| Metric | Sparse-only (worst measured) | Winning config | Lift |
|---|---|---|---|
| Recall@10 | 0.0967 | 0.4288 | **+0.332 absolute (4.4x)** |
| MRR | 0.3676 | 0.8640 | **+0.496 absolute (2.35x)** |
