# 12e — Quantization Tradeoff: int8, Binary, Matryoshka Truncation

**Phase:** 12 · **Date:** 2026-09-26 · **Files:** `experiments/12c_quantization_sweep.py`
**Topics from the course:** Quantization & Matryoshka (topic 33) — int8, binary, and dimension truncation swept and measured directly, not just described.

## 1. What I built

`experiments/12c_quantization_sweep.py`: a real, brute-force recall/latency/size sweep over 3 dimensions (1024/512/256, Matryoshka-style truncation) x 3 precisions (float32/int8/binary) against real `bge-large-en-v1.5` corpus vectors already computed in Phase 3. Output: `docs/results/quantization-tradeoff.md`, `quantization-tradeoff.png`, `quantization-sweep-raw.json`.

## 2. Why this approach

Plan.md asks for a recall-vs-size-and-latency curve across these three axes. None of int8 quantization, binary quantization, or dimension truncation require re-embedding anything — they're all post-hoc transforms of vectors that already exist (`data/processed/vectors_bge_large.npy`, the real 19,236-chunk corpus embedding from Phase 3). Re-embedding to test this would cost real time for zero additional signal; the entire sweep runs in under 12 seconds because it never needs to.

Measuring this by brute-force NumPy search instead of building 9 separate real Qdrant collections (one per dimension/precision combination) was a deliberate scope call: the question plan.md asks is whether a given *vector representation* preserves nearest-neighbor structure, which is a property of the vectors and the similarity function, not of which ANN index sits on top. Building 9 live Qdrant collections would add real infrastructure work without changing what's being measured.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ----------------- |
| Re-embed the corpus at 512/256 dims by retraining or using a genuine MRL model | Would show what Matryoshka truncation is *supposed* to look like | Out of scope for this project's model choice (Phase 3 already picked `bge-large-en-v1.5`); the actually useful question here is what this project's *shipped* model costs under truncation, not a hypothetical model's behavior |
| Build 9 real Qdrant collections with native scalar/binary quantization configs | Latency numbers would be production-representative | Meaningfully more infrastructure for the same recall/size answer; the latency ordering this would fix is disclosed as a limitation instead (§6), which is honest and far cheaper |
| Report only Recall@10 | Simpler table | NDCG@10 catches within-top-10 *ordering* degradation that a pure membership metric (Recall@10) can miss — worth the one extra column since `_top_k_sorted` was already needed for correctness anyway |

## 4. How it works

Matryoshka-style truncation keeps a vector's first `dim` components and re-normalizes:

```python
def truncate_and_normalize(vecs: np.ndarray, dim: int) -> np.ndarray:
    truncated = vecs[:, :dim]
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return truncated / norms
```

int8 is symmetric linear scalar quantization (unit-normalized components sit almost entirely in [-1, 1], so a fixed scale of 127 keeps every vector's quantization independent — the same scheme Qdrant's own scalar quantization uses):

```python
def quantize_int8(vecs: np.ndarray) -> np.ndarray:
    return np.clip(np.round(vecs * 127.0), -127, 127).astype(np.int8)
```

Binary is sign-based, bit-packed, scored by real popcount-based Hamming distance (`np.bitwise_count`, NumPy >= 2.0) rather than a simulated proxy:

```python
def quantize_binary(vecs: np.ndarray) -> np.ndarray:
    bits = (vecs >= 0).astype(np.uint8)
    return np.packbits(bits, axis=1)
```

A real, non-obvious correctness bug caught before trusting any number: `np.argpartition` only guarantees the top-k *set*, not their order within it. Recall@10 doesn't care, but NDCG@10 is position-weighted — computing it against an arbitrarily-ordered top-10 would silently score against the wrong order. Fixed with a second sort pass (`_top_k_sorted`) applied identically across all three precisions.

## 5. What the numbers said

Real sweep, `uv run python experiments/12c_quantization_sweep.py` (full table and plot: `docs/results/quantization-tradeoff.md`):

| dim | precision | bytes/vector | recall@10 | ndcg@10 |
|---|---|---|---|---|
| 1024 | float32 | 4096 | 0.3194 | 0.5324 |
| 1024 | int8 | 1024 | 0.3085 | 0.5237 |
| 1024 | binary | 128 | 0.2291 | 0.3976 |
| 512 | float32 | 2048 | 0.2934 | 0.5016 |
| 512 | int8 | 512 | 0.2843 | 0.4943 |
| 512 | binary | 64 | 0.1762 | 0.3239 |
| 256 | float32 | 1024 | 0.2495 | 0.4383 |
| 256 | int8 | 256 | 0.2480 | 0.4370 |
| 256 | binary | 32 | 0.1203 | 0.2227 |

Two clean, consistent, real patterns: **int8 is nearly free** — a 4x size reduction costs 1.09-3.4 percentage points of recall@10 at every dimension (e.g., 0.3194 → 0.3085 at 1024d). **Binary is not** — a 32x size reduction over float32 (at the same dim) costs 9-13 points of recall@10, and the cost compounds with truncation (0.3194 at 1024d/float32 down to 0.1203 at 256d/binary, more than a 2.6x relative drop). Truncation alone (float32 across 1024→512→256) costs roughly 2.6-4.4 points per halving — real but far gentler than binary's cliff.

## 6. What surprised me / what to watch for

**The real headline finding of this unit isn't quantization — it's that this script's own float32/1024/no-quantization baseline (recall@10=0.3194) doesn't match `docs/results/rerank-tradeoff.md`'s already-published no-rerank baseline (0.4324) for the identical queries, encoder, and exclude-doc-id rule.** Investigated rather than shipped as-is (CLAUDE.md: stop and investigate when a measurement contradicts expectation). Ruled out, directly: query-vector differences (byte-identical between this script's `device="cpu"` call and the production `device="mps"` default — same cache-key, same result), corpus-vector staleness (byte-identical between `vectors_bge_large.npy` and the live Qdrant collection's stored vectors, checked point-by-point), and a top-k selection bug (a full `np.argsort` reproduces the same low number as `argpartition`, so it isn't a partial-sort artifact).

What's real: live Qdrant HNSW search and exact brute-force cosine search agree on only **57.8%** of their top-10 results on average for these real queries — and of the real ground-truth chunks either search finds, **348 are found only by the live index vs. 95 found only by brute force**, a ~3.6x asymmetry. The likely mechanism: this golden set's ground truth (`aegis/eval/auto_label.py`, Phase 4) was labeled from candidates `retrieve_candidates.py` pulled from the *same live Qdrant index* — so a chunk can only ever become "ground truth" if that specific approximate index surfaced it as a labeling candidate in the first place. Exact brute-force search, run independently here, finds genuinely different (not necessarily worse) nearest neighbors that were never in the candidate pool a human could have labeled. This doesn't invalidate Phase 6/7's numbers — they correctly measure the system as actually served, through the actual live index — but it means recall against this golden set is partly a measure of agreement with the *specific retrieval mechanism used to build the labels*, not a mechanism-independent measure of embedding quality. Every row in this unit's own sweep uses brute-force search consistently, so the *relative* quantization effect is still a clean, isolated measurement; only the *absolute* recall level shouldn't be cross-compared to the live-index numbers, and the report says so prominently rather than leaving readers to notice the gap themselves.

**Binary search came out slower than float32 in this measurement (1.3ms vs. 0.3ms/query at 1024d) — an artifact of this being unoptimized NumPy, not a real property of binary quantization.** A single BLAS `matmul` for float32 is far more optimized than a Python-loop Hamming-distance computation over packed bytes; a real production binary index (Qdrant's own SIMD-accelerated scalar/binary quantization) would show the opposite ordering. Disclosed directly in the results doc rather than left to mislead a reader into thinking binary quantization is slower in general — the size column is the implementation-independent, trustworthy one here.

## 7. Where this fits and what's next

Closes plan.md's Phase 12 step 5 exit criterion (`docs/results/quantization-tradeoff.md`, recall vs. size and latency). The live-vs-brute-force finding is bigger than this unit: it's real, disclosed context that should inform how Phase 16/19 (system-level eval, RAG-vs-fine-tuning writeup) talk about this golden set's recall numbers in general, not just here. Unit 6 (cost model) is next — its cost-per-query table can now cite int8's near-free 4x size reduction as a real candidate for the production config, separate from whatever the eventual delta-indexing re-upsert (12d §7) ends up shipping.
