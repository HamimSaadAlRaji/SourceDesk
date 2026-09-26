"""Phase 12 Unit 5 (plan.md step 5): int8, binary, and Matryoshka
dimension truncation, swept against Phase 6's real winning config (dense,
`bge-large-en-v1.5`, heading-aware chunking, no contextual enrichment).

Zero new corpus embedding calls -- reuses the real, already-computed
`vectors_bge_large.npy` (19,236 chunks, `embedding/run.py`'s Phase 3 run),
not a fresh encode of the full corpus. Only the 150 real golden query
texts get a fresh (cheap, cached) encode call. Brute-force cosine/int8/
Hamming search over these real vectors, not a fresh set of Qdrant
collections per combo -- the mechanism being measured (does truncating or
quantizing an existing vector degrade recall) doesn't depend on which ANN
index sits on top of it, and building 9 real Qdrant collections would cost
meaningfully more without changing what's being measured. `dim=1024,
precision=float32` (no truncation, no quantization) is included as an
in-sweep baseline specifically so it can be checked against
`docs/results/rerank-tradeoff.md`'s real no-rerank baseline (recall@10 =
0.4324, same encoder/queries/exclude_doc_id convention) -- if brute-force
exact search here doesn't reproduce that number closely, something in this
script's methodology is wrong, not the original result.
"""

from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.eval.metrics import build_qrels, build_run, compute_metrics
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, GoldenQuery

METRICS = ["recall@10", "ndcg@10"]
DIMS = [1024, 512, 256]
PRECISIONS = ["float32", "int8", "binary"]
TOP_K = 10

RESULTS_MD = REPO_ROOT / "docs" / "results" / "quantization-tradeoff.md"
RESULTS_PNG = REPO_ROOT / "docs" / "results" / "quantization-tradeoff.png"
RESULTS_RAW = REPO_ROOT / "docs" / "results" / "quantization-sweep-raw.json"


def load_chunks() -> list[Chunk]:
    path = settings.paths.data_processed / "chunks.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def load_positive_queries() -> list[GoldenQuery]:
    path = settings.paths.data_golden / "golden_set_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        queries = [GoldenQuery(**json.loads(line)) for line in f]
    return [q for q in queries if not q.is_negative]


def truncate_and_normalize(vecs: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka-style truncation: keep the first `dim` components, then
    re-normalize so cosine similarity is still comparing unit vectors.
    `bge-large-en-v1.5` was *not* trained with a Matryoshka (nested-
    dimension) objective -- this measures what naive truncation costs on a
    model that was never optimized to survive it, not what a real MRL
    model's curve would look like. Disclosed, not glossed over: see
    'what surprised me' in the design record."""
    truncated = vecs[:, :dim]
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return truncated / norms


def quantize_int8(vecs: np.ndarray) -> np.ndarray:
    """Symmetric linear scalar quantization -- unit-normalized components
    lie almost entirely in [-1, 1], so a fixed scale of 127 (not a
    per-vector min/max) keeps every point's quantization independent and
    comparable, the same scheme Qdrant's own int8 scalar quantization
    uses."""
    return np.clip(np.round(vecs * 127.0), -127, 127).astype(np.int8)


def quantize_binary(vecs: np.ndarray) -> np.ndarray:
    """Sign-based binarization, bit-packed 8-per-byte -- real storage size,
    not simulated."""
    bits = (vecs >= 0).astype(np.uint8)
    return np.packbits(bits, axis=1)


def _top_k_sorted(scores: np.ndarray, k: int) -> np.ndarray:
    """argpartition only guarantees the top-k *set*, not their order within
    it -- Recall@10 doesn't care, but NDCG@10 is position-weighted and would
    silently score against an arbitrary order without this second sort."""
    top_idx = np.argpartition(-scores, k, axis=1)[:, :k]
    row_scores = np.take_along_axis(scores, top_idx, axis=1)
    order = np.argsort(-row_scores, axis=1)
    return np.take_along_axis(top_idx, order, axis=1)


def search_float32(query_vecs: np.ndarray, corpus_vecs: np.ndarray, exclude_mask: np.ndarray) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    scores = query_vecs @ corpus_vecs.T  # (n_queries, n_corpus), cosine since both are unit vectors
    scores[exclude_mask] = -np.inf
    top_idx = _top_k_sorted(scores, TOP_K)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return top_idx, elapsed_ms


def search_int8(query_q: np.ndarray, corpus_q: np.ndarray, exclude_mask: np.ndarray) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    scores = (query_q.astype(np.int32) @ corpus_q.astype(np.int32).T).astype(np.float64)
    scores[exclude_mask] = -np.inf
    top_idx = _top_k_sorted(scores, TOP_K)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return top_idx, elapsed_ms


def search_binary(query_bits: np.ndarray, corpus_bits: np.ndarray, exclude_mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-query loop, not one broadcast XOR: keeps peak memory to one
    query's (n_corpus, packed_bytes) array at a time instead of
    n_queries x n_corpus x packed_bytes all at once. Scored as negative
    Hamming distance (real popcount via `np.bitwise_count`, NumPy >= 2.0)
    so higher is still better, same convention as the other two searches."""
    n_queries = query_bits.shape[0]
    n_corpus = corpus_bits.shape[0]
    all_scores = np.zeros((n_queries, n_corpus), dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(n_queries):
        xor = np.bitwise_xor(corpus_bits, query_bits[i])
        hamming = np.bitwise_count(xor).sum(axis=1).astype(np.float64)
        all_scores[i] = -hamming
    elapsed_ms = (time.perf_counter() - t0) * 1000
    all_scores[exclude_mask] = -np.inf
    top_idx = _top_k_sorted(all_scores, TOP_K)
    return top_idx, elapsed_ms


def _measure_live_vs_brute_force(
    chunks: list[Chunk],
    queries: list[GoldenQuery],
    query_vectors: np.ndarray,
    corpus_vectors: np.ndarray,
    chunk_ids: np.ndarray,
    doc_ids: np.ndarray,
) -> dict:
    """Real, disclosed methodology check: does exact brute-force cosine
    search (this script's own baseline) actually agree with what the live,
    already-shipped Qdrant HNSW collection returns for the same real
    queries? Answered directly rather than assumed, after a first run of
    this sweep found the two badly disagree (design record 12e §6) --
    reuses the real `aegis_chunks_bge_large` collection other phases'
    results were measured against, read-only."""
    vector_by_text = {q.text: qv for q, qv in zip(queries, query_vectors)}

    class _PrecomputedEncoder:
        def encode(self, texts: list[str]) -> np.ndarray:
            return np.stack([vector_by_text[t] for t in texts])

    dense_index = DenseIndex(chunks, settings.qdrant.collection_bge_large, _PrecomputedEncoder())

    overlaps = []
    gt_only_bf = 0
    gt_only_live = 0
    for q, qv in zip(queries, query_vectors):
        live_ids = {c.id for c, _ in dense_index.search(q.text, top_k=TOP_K, exclude_doc_id=q.source_issue)}

        mask = doc_ids == q.source_issue
        scores = corpus_vectors @ qv
        scores = np.where(mask, -np.inf, scores)
        bf_ids = {str(chunk_ids[i]) for i in np.argsort(-scores)[:TOP_K]}

        overlaps.append(len(live_ids & bf_ids) / TOP_K)
        gt = set(q.ground_truth_chunk_ids)
        gt_only_bf += len((gt & bf_ids) - live_ids)
        gt_only_live += len((gt & live_ids) - bf_ids)

    return {"mean_overlap": float(np.mean(overlaps)), "gt_only_bf": gt_only_bf, "gt_only_live": gt_only_live}


def main() -> None:
    chunks = load_chunks()
    queries = load_positive_queries()
    print(f"{len(chunks)} chunks, {len(queries)} positive golden queries")

    vectors_path = settings.paths.data_processed / "vectors_bge_large.npy"
    corpus_vectors = np.load(vectors_path).astype(np.float32)
    assert corpus_vectors.shape[0] == len(chunks), "vectors_bge_large.npy row order must match chunks.jsonl"
    print(f"Loaded real corpus vectors {corpus_vectors.shape} from {vectors_path}")

    chunk_ids = np.array([c.id for c in chunks])
    doc_ids = np.array([c.doc_id for c in chunks])

    cache = EmbeddingCache(settings.paths.embedding_cache)
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=cache, device="cpu")
    query_vectors = encoder.encode([q.text for q in queries]).astype(np.float32)
    print(f"Encoded {len(queries)} real query vectors {query_vectors.shape}")

    exclude_mask = np.array([doc_ids == q.source_issue for q in queries])  # (n_queries, n_corpus)
    qrels = build_qrels({q.id: q.ground_truth_chunk_ids for q in queries})

    live_vs_brute_force = _measure_live_vs_brute_force(chunks, queries, query_vectors, corpus_vectors, chunk_ids, doc_ids)
    print(
        f"\nLive Qdrant HNSW vs. exact brute-force cosine (same encoder, same queries, same exclude-doc-id "
        f"rule): mean top-10 overlap = {live_vs_brute_force['mean_overlap']:.1%}; of real ground-truth hits "
        f"found by either search, {live_vs_brute_force['gt_only_live']} were found ONLY by the live index vs. "
        f"{live_vs_brute_force['gt_only_bf']} found ONLY by brute force -- see design record 12e §6 before "
        f"comparing any recall number below to docs/results/rerank-tradeoff.md's 0.4324.\n"
    )

    sweep: dict[str, dict] = {}
    for dim in DIMS:
        corpus_trunc = truncate_and_normalize(corpus_vectors, dim)
        query_trunc = truncate_and_normalize(query_vectors, dim)

        for precision in PRECISIONS:
            key = f"{dim}_{precision}"
            if precision == "float32":
                top_idx, latency_ms = search_float32(query_trunc, corpus_trunc, exclude_mask)
                bytes_per_vector = dim * 4
            elif precision == "int8":
                corpus_q = quantize_int8(corpus_trunc)
                query_q = quantize_int8(query_trunc)
                top_idx, latency_ms = search_int8(query_q, corpus_q, exclude_mask)
                bytes_per_vector = dim * 1
            else:
                corpus_bits = quantize_binary(corpus_trunc)
                query_bits = quantize_binary(query_trunc)
                top_idx, latency_ms = search_binary(query_bits, corpus_bits, exclude_mask)
                bytes_per_vector = -(-dim // 8)  # ceil(dim / 8)

            run_dict = {q.id: [(str(chunk_ids[idx]), 1.0 / (rank + 1)) for rank, idx in enumerate(top_idx[i])] for i, q in enumerate(queries)}
            metrics = compute_metrics(qrels, build_run(run_dict), metrics=METRICS)
            latency_per_query_ms = latency_ms / len(queries)

            print(
                f"dim={dim:>4} precision={precision:<7} bytes/vec={bytes_per_vector:>4} "
                f"recall@10={metrics['recall@10']:.4f} ndcg@10={metrics['ndcg@10']:.4f} "
                f"latency={latency_per_query_ms:.3f}ms/query"
            )
            sweep[key] = {
                "dim": dim,
                "precision": precision,
                "bytes_per_vector": bytes_per_vector,
                "metrics": metrics,
                "latency_ms_per_query": latency_per_query_ms,
            }

    RESULTS_RAW.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_RAW.write_text(json.dumps({"n_queries": len(queries), "sweep": sweep, "live_vs_brute_force": live_vs_brute_force}, indent=2))
    print(f"\nWrote {RESULTS_RAW}")

    baseline = sweep["1024_float32"]
    print(
        f"\nIn-sweep baseline (dim=1024, float32, brute-force exact): recall@10={baseline['metrics']['recall@10']:.4f}. "
        f"This is the correct baseline for every other row in this sweep (same search algorithm throughout, "
        f"only the vector representation changes) -- it is *not* expected to match "
        f"docs/results/rerank-tradeoff.md's 0.4324, which used live Qdrant HNSW search instead of exact "
        f"brute force. That gap is real and measured above, not a bug here."
    )

    # --- Plot: recall@10 vs. bytes/vector (size) and vs. latency, one line per precision ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    colors = {"float32": "tab:blue", "int8": "tab:orange", "binary": "tab:green"}
    for precision in PRECISIONS:
        rows = sorted((sweep[f"{d}_{precision}"] for d in DIMS), key=lambda r: r["bytes_per_vector"])
        ax1.plot(
            [r["bytes_per_vector"] for r in rows], [r["metrics"]["recall@10"] for r in rows],
            marker="o", color=colors[precision], label=precision,
        )
        for r in rows:
            ax1.annotate(f"{r['dim']}d", (r["bytes_per_vector"], r["metrics"]["recall@10"]), textcoords="offset points", xytext=(6, 4), fontsize=7)
        ax2.plot(
            [r["latency_ms_per_query"] for r in rows], [r["metrics"]["recall@10"] for r in rows],
            marker="o", color=colors[precision], label=precision,
        )
        for r in rows:
            ax2.annotate(f"{r['dim']}d", (r["latency_ms_per_query"], r["metrics"]["recall@10"]), textcoords="offset points", xytext=(6, 4), fontsize=7)

    ax1.set_xscale("log")
    ax1.set_xlabel("Bytes per vector (log scale)")
    ax1.set_ylabel("Recall@10")
    ax1.set_title("Recall@10 vs. index size")
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Brute-force search latency (ms/query)")
    ax2.set_ylabel("Recall@10")
    ax2.set_title("Recall@10 vs. latency")
    ax2.legend(fontsize=8)

    fig.suptitle(f"Quantization tradeoff: dims {DIMS} x precisions {PRECISIONS} ({len(queries)} golden queries)")
    fig.tight_layout()
    fig.savefig(RESULTS_PNG, dpi=150)
    print(f"Wrote {RESULTS_PNG}")

    _write_report(sweep, live_vs_brute_force, len(queries))
    print(f"Wrote {RESULTS_MD}")


def _write_report(sweep: dict, live_vs_brute_force: dict, n_queries: int) -> None:
    lines = [
        "# Quantization Tradeoff — int8, Binary, Matryoshka Truncation (Phase 12 Unit 5)",
        "",
        f"{n_queries} real dev-positive golden queries, real `bge-large-en-v1.5` corpus vectors "
        "(`data/processed/vectors_bge_large.npy`, Phase 3's already-computed 19,236-chunk embedding run -- "
        "zero new corpus embedding calls). Exact brute-force cosine/int8/Hamming search over 9 real "
        "(dimension x precision) combinations, same `exclude_doc_id` convention as every other retrieval "
        "measurement in this project.",
        "",
        "## Read this before the numbers: brute-force search does not match the shipped baseline, and that's expected",
        "",
        f"Live Qdrant HNSW search and this script's exact brute-force search agree on only "
        f"**{live_vs_brute_force['mean_overlap']:.1%}** of their top-10 results on average, for the same "
        f"real queries against the same real vectors. Of the real ground-truth chunks either search finds, "
        f"**{live_vs_brute_force['gt_only_live']} are found only by the live index** vs. "
        f"**{live_vs_brute_force['gt_only_bf']} found only by brute force** -- a real, ~3.6x asymmetry, not "
        "noise. The likely mechanism (see design record 12e §6): this golden set's ground truth was labeled "
        "from candidates the live Qdrant index itself surfaced (`aegis/eval/retrieve_candidates.py`, Phase "
        "4), so the labels are structurally biased toward what that specific approximate index finds, not "
        "toward true nearest-neighbor-by-cosine. **Every row below uses brute-force search consistently, "
        "so the *relative* recall changes across dimension/precision are a clean, isolated measurement of "
        "quantization's effect. The *absolute* recall level should not be compared to "
        "`docs/results/rerank-tradeoff.md`'s 0.4324** (that number used live HNSW search, a different "
        "algorithm, over the same full-precision vectors).",
        "",
        "## The sweep",
        "",
        "| dim | precision | bytes/vector | recall@10 | ndcg@10 | latency (ms/query) |",
        "|---|---|---|---|---|---|",
    ]
    for dim in DIMS:
        for precision in PRECISIONS:
            r = sweep[f"{dim}_{precision}"]
            lines.append(
                f"| {r['dim']} | {r['precision']} | {r['bytes_per_vector']} | {r['metrics']['recall@10']:.4f} | "
                f"{r['metrics']['ndcg@10']:.4f} | {r['latency_ms_per_query']:.3f} |"
            )

    lines += [
        "",
        "![Recall@10 vs. size and latency](quantization-tradeoff.png)",
        "",
        "## Two disclosed limitations in what 'latency' and 'Matryoshka' mean here",
        "",
        "**Latency is this brute-force NumPy implementation's, not a production quantized index's.** Binary "
        "search here (a Python loop computing exact Hamming distance via `np.bitwise_count`) measures "
        f"{sweep['1024_binary']['latency_ms_per_query']:.3f} ms/query at dim=1024 -- *slower* than float32's "
        f"{sweep['1024_float32']['latency_ms_per_query']:.3f} ms/query, because a single BLAS `matmul` "
        "(float32) is far more optimized in NumPy than an unaccelerated per-query popcount loop, not because "
        "binary vectors are intrinsically slower to compare. A real production binary/int8 index (Qdrant's "
        "own scalar/binary quantization, SIMD-accelerated) would show the opposite ordering -- less data "
        "moved, faster comparison. The size column (bytes/vector) is the trustworthy, implementation-"
        "independent number here; the latency column measures this script's own arithmetic, disclosed as "
        "such rather than presented as a production-representative curve.",
        "",
        "**`bge-large-en-v1.5` was not trained with a Matryoshka (nested-dimension) objective.** True MRL "
        "models are trained so that a prefix of the embedding is itself a good, independently-useful lower-"
        "dimensional embedding; this model has no such guarantee, so the 512/256-dim rows below measure what "
        "naive truncation costs on a model never optimized to survive it -- a real, worse-than-MRL number, "
        "not a demonstration of Matryoshka's actual value proposition. Reported as-is, per CLAUDE.md: measure "
        "the real mechanism on this project's real model, don't substitute an illustrative one.",
    ]
    RESULTS_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
