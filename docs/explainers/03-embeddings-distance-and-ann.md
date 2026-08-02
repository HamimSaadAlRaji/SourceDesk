# 03 — Embeddings, Distance & the Vector Store

**Phase:** 3 · **Date:** 2026-08-02 · **Files:** `src/aegis/embedding/{cache,encoder,knn,vector_store,run}.py`, `src/aegis/chunking/run.py`, `experiments/03a_cosine_dot_proof.py`, `experiments/03b_ann_sweep.py`, `experiments/03c_prefilter_vs_postfilter.py`
**Topics from the course:** Metadata filtering; Semantic search & embeddings; Distance metrics (Euclidean distance, cosine similarity, dot product); Contrastive training; KNN → ANN → HNSW; Vector databases

## 1. What I built

The first persisted chunked corpus (heading-aware, 19,236 chunks → `data/processed/chunks.jsonl` — Phase 2 deliberately never wrote one, since comparing four strategies isn't committing to one). A content-hash-keyed on-disk embedding cache; a `sentence-transformers`/MPS encoder wrapper built on it; brute-force exact k-NN as ground truth; a Qdrant vector store module. All 19,236 chunks embedded and indexed with all three candidate encoders — `bge-small-en-v1.5` (dev workhorse), `bge-large-en-v1.5` and `BGE-M3` (the two real bake-off candidates, full comparison in Phase 6) — into three separate Qdrant collections. A real numpy proof that cosine similarity reduces to a dot product on this project's own normalized vectors. An HNSW recall/latency sweep (36 configs) against the brute-force ground truth. A measured pre-filter vs. post-filter comparison across five `product_area` categories.

## 2. Why this approach

Three real constraints shaped this phase, not abstract best practice. First, Phase 4's golden set doesn't exist yet — Phase 3 precedes it in the schedule — so "how good is my index" had to be measured against something else defensible: the corpus's own vectors, sampled as queries against themselves. That measures HNSW's fidelity to the exact nearest-neighbor structure of the embedding space, not retrieval quality for real questions; the latter is Phase 6's job once Phase 4 exists. Second, this machine is a 16GB M1 Pro — CLAUDE.md's own risk table flagged memory pressure as a real concern before this phase started, and it turned out to be exactly right (§6). Third, the corpus itself is not a clean textbook dataset: Phase 1 found GitHub issue-template boilerplate polluting a keyword classifier; Phase 2 found it breaking a markdown-fence regex; this phase found the *same* boilerplate collapsing thousands of chunks into identical embeddings, which broke the naive way to measure ANN recall (§6). Three phases, one recurring corpus artifact — worth noticing as a pattern, not three unrelated bugs.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Trust a small-sample throughput measurement (200 chunks, warm) to plan the full-corpus embedding run | Fast to measure, seemed like a reasonable ETA basis | Off by ~10x: the clean sample predicted ~118 chunks/sec (~3 min for the full corpus); the actual first full run averaged 12.7 chunks/sec (1,517s, ~25 min), driven by real memory pressure the small sample never triggered. Measured the real thing instead of trusting the extrapolation — see §6. |
| ID-set recall (`overlap(approx_ids, exact_ids) / k`) for measuring HNSW fidelity | Standard, simple ANN-benchmark metric | This corpus has real, large-scale embedding ties (39.7% of vectors are exact duplicates of another vector) — ID-set recall penalizes an approximate search for returning a *different, equally correct* tied neighbor than the arbitrary one brute-force KNN happened to return. Built a score-threshold variant (`recall_at_k_score_aware`) instead, which credits any result meeting or beating the true k-th-best similarity score regardless of which tied ID it is. |
| Run `bge-large` and `BGE-M3` embedding concurrently, to halve wall-clock wait time | Independent models, independent collections — looks parallelizable | CLAUDE.md's risk table already flagged "M1 Pro 16GB memory pressure" as a real risk before this phase started; running two large transformer models on MPS simultaneously was judged too likely to compound it. Ran sequentially instead — turned out to be the right call once swap usage was actually measured (§6). |
| Sweep HNSW parameters (`m`, `ef_construct`, `ef_search`) across all three encoders | More complete picture per encoder | The sweep only needs to characterize HNSW's general recall/latency behavior on this corpus, not per-encoder specifics (the underlying graph algorithm doesn't care which embedding model produced the vectors). Swept only `bge-small` (fast enough for a real 36-point grid); `bge-large`/`BGE-M3` get a single config chosen from that curve. |
| One Qdrant collection with three named dense vectors (`dense_bge_small`, `dense_bge_large`, `dense_bge_m3`) | Avoids duplicating chunk payload data three times | Three separate collections instead — simpler mental model, and Phase 6's ablation grid swaps between encoders by collection name anyway. Each collection keeps a single `dense` named vector, leaving room for Phase 5 to add a `sparse` one without a schema change. |
| Real Phase 4 golden queries for the ANN recall sweep | The methodologically "right" query distribution | Doesn't exist yet — Phase 4 comes after this phase in the schedule. Used self-sampled corpus vectors instead, explicitly documented as measuring index fidelity (does HNSW find the same neighbors exact KNN would), not query relevance (does retrieval answer real questions) — the latter is measured for real once Phase 4 and Phase 6 land. |

## 3a. Why cosine is a meaningful distance here (contrastive training)

Cosine similarity being *computable* on two vectors doesn't make it *meaningful* — a randomly initialized network's output vectors have no reason for their angles to encode anything. `bge-small`, `bge-large`, and `BGE-M3` are trained with a contrastive objective: for a batch of (anchor, positive, negatives) text triples, the loss directly pushes the anchor's embedding *angle* toward its positive's and away from every negative's — typically InfoNCE-style, computed on the normalized vectors' dot products (i.e. their cosine similarity) directly as the training signal. The model is optimized, batch after batch, to make cosine similarity the quantity that separates "means the same thing" from "doesn't" — that's *why* the proof in §5 matters practically, not just as float-precision trivia: the encoder was never trained to make Euclidean distance or any other metric meaningful, so using one would measure something the model was never optimized to encode.

## 4. How it works

**Cache-aware encoding** de-duplicates before ever calling the model — a repeated or already-cached text is never re-encoded, and de-duplication happens *within* a single call too (not just across calls):

```python
missing_hashes = [h for h in dict.fromkeys(hashes) if h not in resolved]
if missing_hashes:
    hash_to_text = dict(zip(hashes, texts))
    missing_texts = [hash_to_text[h] for h in missing_hashes]
    vectors = self._encode_fn(missing_texts)
```

**The memory-pressure mitigation** (found necessary in §6, not applied speculatively): sort by text length before batching so a batch pads to a length close to its own contents rather than an outlier's, use a smaller `sentence-transformers` batch size for the two large models, and explicitly release MPS memory after every outer batch:

```python
order = sorted(range(len(chunks)), key=lambda i: len(chunks[i].embedding_text))
...
if torch is not None and torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

Original document order is restored (`vectors[order] = sorted_vectors`) before saving or indexing, so nothing downstream needs to know the sort happened.

**Score-threshold recall**, the fix for the corpus's duplicate-embedding ties:

```python
def recall_at_k_score_aware(approx_scores, exact_kth_scores, k, epsilon=1e-4):
    per_query = []
    for scores, kth in zip(approx_scores, exact_kth_scores):
        correct = sum(1 for s in scores if s >= kth - epsilon)
        per_query.append(min(correct, k) / k)   # always /k, never /len(scores)
    return sum(per_query) / len(per_query) if per_query else 0.0
```

The `/ k`, not `/ len(scores)`, is load-bearing — see §6 for the bug this fixed.

## 5. What the numbers said

**Cosine ≡ dot product**, on this project's own real vectors (2,000 random pairs from the 19,236-chunk `bge-small` embedding, not a toy example): max absolute difference between the full cosine formula and a plain dot product was `1.79e-07` — equal within float32 precision. All embedding norms measured exactly `1.000000` (`sentence-transformers`' `normalize_embeddings=True` confirmed working, not assumed).

**Embedding throughput**, all measured on this machine:

| Model | Clean 200-chunk sample | Full corpus (19,236 chunks) |
| --- | --- | --- |
| `bge-small-en-v1.5` | 118.0 chunks/sec | 1,517.1s (12.7/s avg) — unmitigated, swap-driven |
| `bge-large-en-v1.5` | 16.8 chunks/sec | 665.0s (28.9/s avg) — mitigated |
| `BGE-M3` | 7.8 chunks/sec | 799.4s (24.1/s avg) — mitigated |

Note that `bge-large` and `BGE-M3` — much bigger models — both finished the full corpus *faster in absolute wall-clock time* than `bge-small`'s first, unmitigated run, despite `bge-small` having the highest raw chunks/sec. The mitigation (§6) is why.

**HNSW recall vs. latency** (`docs/results/ann-recall-vs-latency.md`, plot at `docs/results/ann-recall-vs-latency.png`): **Recall@10 (score-aware) = 1.0000 for all 36 (m, ef_construct, ef_search) configurations tested.** At 19,236 points, HNSW's approximation costs zero measured recall regardless of parameters — the tradeoff curve plan.md anticipated doesn't materialize at this corpus scale. Latency was flat too (6.1–8.1ms/query across every config, no consistent trend). Chosen config: Qdrant's own defaults (`m=16, ef_construct=100, ef_search=100`) — no measured reason to deviate.

**Pre-filter vs. post-filter** (`docs/results/prefilter-vs-postfilter.md`): naive post-filtering (search top-10 unfiltered, discard non-matches) scored recall between **0.1667 and 0.4033** depending on category — even `general`, the single largest category at 26% of the corpus, only reached 0.40 and came back short of 10 results on 90% of queries. Native Qdrant pre-filtering (`query_filter`) scored a perfect **1.0000** on every category tested, narrow or broad.

## 6. What surprised me / what to watch for

**The 10x throughput gap was real memory pressure, not noise — confirmed, not assumed.** `sysctl vm.swapusage` showed ~3.5GB of swap in use mid-run. This is precisely the risk CLAUDE.md's own risk table named before this phase started ("M1 Pro 16GB memory pressure... smaller batches") — it took actually hitting a 1,517-second run to take it seriously enough to fix. Root cause had two parts: general memory pressure from a sustained ~25-minute MPS workload, and a specific, findable trigger — a batch spanning `issue::2812`–`issue::2936` contained a 2,830-token chunk (a legitimate atomic-oversized-code-block chunk, per Phase 2's own documented exception) padded next to ~135-token ones, dragging that batch's throughput to roughly 1/25th of the run's average. Fixed with three changes: sort-by-length batching, a smaller `sentence-transformers` batch size for the two large models, and periodic `torch.mps.empty_cache()` calls.

**Two real bugs in my own measurement code — not the system under test — both caught by the same discipline: notice when a result doesn't make sense, don't write it down.** First: naive ID-set recall reported ~0.67 flat across every HNSW parameter, which is impossible (recall should climb with `ef_search`). Root cause: **39.7% of the corpus's 19,236 vectors are exact-duplicate embeddings** — the single largest group is 1,354 issues whose entire heading-aware chunk is the literal string `_No response_`, a GitHub issue-template placeholder for an unanswered field. This is the *same* boilerplate-pollution category Phase 1 found corrupting a keyword classifier and Phase 2 found breaking a markdown-fence regex — a third, independent failure mode from the same corpus artifact. Fixed by building a score-threshold recall metric that credits any tied neighbor, not only the specific one brute-force KNN happened to return.

Second: that fix had its *own* bug. It divided by the number of results actually *returned* rather than the number *requested* — invisible until the pre/post-filter experiment produced a self-contradicting result ("short of 10 results on 100% of queries" alongside "recall=1.0000"), which is only possible if a method returning 2 results, both correct, scores a perfect 1.0 instead of 0.2. That's exactly the failure mode the experiment existed to catch, silently hidden by the metric measuring it. Fixed by requiring an explicit `k` and always dividing by it, with an empty result list scored 0.0 rather than excluded from the average. Both bugs are now regression-tested (`tests/test_knn_recall_score_aware.py`).

**HNSW recall is saturated at this corpus scale.** 19,236 points is small enough that even the sparsest graph tested (`m=8`) with the shallowest search (`ef_search=50`) already finds every exact neighbor. The parameter choice right now is not a recall decision at all — worth re-running this sweep once the corpus is an order of magnitude larger (real production scale, or after Phase 5's sparse vectors grow the collection), since that's when these parameters are expected to actually start trading recall against latency.

**The pre/post-filter gap was bigger, and more universal, than expected.** I anticipated narrow categories (`graphql`, 56 chunks) would suffer under naive post-filtering. I did not expect the *largest* category in the corpus (`general`, 5,081 chunks, 26% of everything) to also fail — a plain top-10 search still comes back short of 10 matching results 90% of the time, because the other 74% of the corpus crowds out the pool even when the target category is large in absolute terms.

## 7. Where this fits and what's next

Phase 5 adds sparse vectors (`bm25s`) as a second named vector (`sparse`) alongside `dense` in these same three collections — the schema already anticipates this. Phase 4's golden set replaces this phase's self-sampled queries with real query relevance, letting the ANN sweep (and everything downstream) be re-measured against ground truth that actually matters. Phase 6's ablation grid is where `bge-large` vs. `BGE-M3` gets a real winner — this phase only proves both are indexed, queryable, and comparably fast to search (HNSW recall/latency didn't depend on the encoder). The provisional heading-aware chunking choice carried in from Phase 2 is still provisional; Phase 6 confirms or overturns it with real retrieval numbers, not code-block-integrity proxies. Retrieval-layer metadata filters (`product_area`, `doc_type`, `doc_version`) are now a settled decision, not an assumption: always `query_filter`, never post-hoc discard — confirmed necessary here, with numbers, not by convention.
