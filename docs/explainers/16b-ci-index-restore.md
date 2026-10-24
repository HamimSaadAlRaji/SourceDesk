# 16b — Restoring the Index in CI (and what it found)

**Phase:** 16 · **Date:** 2026-10-23 · **Files:** `src/aegis/eval/ci_index.py`, `tests/test_ci_index.py`, `experiments/16a_ci_index_restore.py`, `experiments/16b_hnsw_recall_diagnosis.py`, `experiments/16c_snapshot_restore_check.py`, `docs/results/ci-index-restore-raw.json`
**Topics from the course:** Retrieval evaluation metrics (Precision, Recall, MAP@k, MRR); Evaluation strategies (system vs. component). The Phase 3 HNSW topic (`m`, `ef_construct`, `ef_search`) turned out to be load-bearing.

## 1. What I built

`ci_index.py`: `check_alignment` (refuses a chunk file and vector file that don't describe the same corpus), `corpus_fingerprint`, and `restore_index` (real `create_collection` + `upsert_chunks`, then waits for the HNSW build). Three experiments measure whether a CI-rebuilt index can stand in for the served one. **It cannot, and the reason is a property of the golden set, not of CI.**

## 2. Why this approach

The plan agreed on 2026-10-22 was: ship `vectors_bge_large.npy` + `chunks.jsonl` as a release asset, rebuild Qdrant on the runner, gate on the metrics. That avoids embedding 19,236 chunks on a 2-core runner. This unit tested the one assumption underneath it — that a rebuilt index scores like the served one — before building a workflow on it.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Rebuild HNSW from the `.npy` on the runner (the original plan) | Small artifact (75 MB + 27 MB), regenerable from source | Scores recall@10 **0.252 / 0.256** and MRR **0.644 / 0.668** on two rebuilds vs **0.437 / 0.864** live; the two rebuilds' top-10 lists agree on only 54%. The gate would start from an arbitrary, non-reproducible baseline unrelated to every published Phase 6–11 number. |
| Exact cosine in CI (numpy or Qdrant `exact=True`) | Deterministic, no HNSW build, no build noise | Scores recall@10 0.320, MRR 0.708, NDCG@10 0.533: also unrelated to published numbers, and it measures a retriever that isn't the one served. Kept as a candidate for a *separate* ANN-health check, not as the gate's baseline. |
| Snapshot of the live collection (**candidate, decision pending**) | Carries the actual HNSW graph; restore is 3.3 s; repeatable | 334 MB raw / 103 MB gzipped; an opaque, Qdrant-version-pinned binary; and it locks in a graph that is itself only ~67% accurate (§5). |

## 4. How it works

`check_alignment` refuses: wrong dimension, chunk/vector count mismatch, duplicate chunk ids (point IDs derive from the id, so a duplicate silently overwrites), and non-unit vectors (cosine over unnormalized vectors ranks differently). `corpus_fingerprint` hashes `(id, embedding_text)` pairs in order plus the vector bytes; the text is hashed directly because the stored `content_hash` field would not change on a hand edit of `chunks.jsonl` (found when my own test caught it). 6 TDD'd tests; `uv run pytest -q` and ruff/mypy unchanged.

## 5. What the numbers said

All 150 dev-positive queries, bge-large, `exclude_doc_id=source_issue`, top-10, M1 Pro (8 cores, CPU encode). Raw: `docs/results/ci-index-restore-raw.json`.

| Index | Recall@10 | MRR | NDCG@10 | MAP@10 |
| --- | --- | --- | --- | --- |
| Live `aegis_chunks_bge_large` | **0.4372** | **0.8640** | **0.7130** | 0.3756 |
| Rebuilt from `.npy`, run A | 0.2522 | 0.6442 | 0.4437 | 0.1899 |
| Rebuilt from `.npy`, run B | 0.2560 | 0.6681 | 0.4495 | 0.1887 |
| Exact cosine (numpy) | 0.3199 | 0.7082 | 0.5330 | 0.2377 |
| **Snapshot restored into fresh Qdrant** | 0.4439 | 0.8640 | 0.7189 | 0.3800 |

Top-10 overlap: rebuild A vs B **0.543**; rebuild vs live 0.467; rebuild vs exact 0.591; live vs exact 0.579. Snapshot vs live: **130 / 150** queries return the identical ordered top-10; snapshot vs a second pass of itself: **150 / 150**.

Costs: restore from `.npy` = 24.4 s and 25.5 s upsert + ~1 s index wait; snapshot restore 3.3 s; bge-large model load + first pass 13.9 s; encoding 150 queries on CPU with the embedding cache bypassed 40.3 s (269 ms/query; also 41.5 s on the first, failed attempt). Search p50 ~12–15 ms. These are M1 Pro numbers, a lower bound for a 2-core hosted runner; the runner timing is measured when `eval.yml` first runs.

**Diagnosis (stopped here per CLAUDE.md: a measurement contradicted an expectation).** Three explanations tested in `16b_hnsw_recall_diagnosis.py`:

| Hypothesis | Result | Verdict |
| --- | --- | --- |
| Ties between duplicate vectors make top-10 membership arbitrary | 41.8% of chunks (8,036 / 19,236) are in an exact-duplicate group, but only **4.0%** of exact top-10 hits are duplicates | rejected |
| The `must_not doc_id` filter degrades filtered HNSW | Qdrant `exact=True` *with* the filter agrees **98.7%** with my numpy exact search; HNSW with **no filter at all** still agrees only 66.5% | rejected (and validates the numpy baseline) |
| The graph is simply inaccurate on this corpus | HNSW top-10 ID overlap with exact **0.671**, score-based recall **0.682** | **supported** |

So the served index returns about two thirds of the true cosine top-10. Phase 3's sweep already showed this symptom — ID recall flat at 0.67–0.69 across `m` 8–32, `ef_construct` 50–200 and `ef_search` 50–400 — and I did not connect it then. A flat curve under `ef_search` means the missing neighbors aren't reachable from the entry point at all, which fits the duplicate-heavy corpus (a node whose 16 neighbors are all identical copies of itself has no edge to anything else). **That mechanism is a hypothesis; I did not test it** (the test would be deduplicating before indexing).

**Why live scores higher than exact**: the golden labels came from candidates the live index surfaced (Phase 4 `retrieve_candidates`), so the labels reward reproducing that specific graph's quirks. Exact cosine is the *better* retriever by definition and scores 27% lower recall on this golden set. This was suspected in 12e from a 57.8% overlap; it is now isolated from ties and filters.

## 6. What surprised me / what to watch for

- **The golden set is partly a fingerprint of one HNSW build.** Any change that alters which neighbors the graph returns, including a harmless re-index, moves the headline metrics by far more than any regression a gate is meant to catch (recall@10 swing of 0.19 between a rebuild and live).
- **Rebuild-to-rebuild noise is real but small beside that**: A vs B differ by 0.004 recall@10, 0.024 MRR, 0.006 NDCG. Only a restored snapshot removes it entirely (150/150 repeat).
- **The snapshot is close to, not equal to, live** (130/150 identical, recall@10 +0.0067). I did not find out why the live collection differs from its own snapshot on 20 queries (segment layout is the likely candidate, untested). The gate baseline must therefore be taken from the restored snapshot, not copied from Phase 6.
- **Docker Desktop had stopped overnight**; the first run died on a refused connection right after the encode step. Nothing was lost; noted because CI-restore code assumes Qdrant is reachable and says so loudly.

## 7. Where this fits and what's next

Open decision for the user (see chat): gate on a restored snapshot of the live collection vs. other options. Whichever is chosen, the ANN-accuracy finding (~67% of the true top-10 returned) is a Phase 19 failure-analysis item and is added to `docs/future-work.md`: testing the duplicate-clique hypothesis, then re-labeling the golden set against exact search, would change every retrieval number in the project and is outside Phase 16's scope.
