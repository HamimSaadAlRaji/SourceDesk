"""HNSW recall/latency sweep vs. brute-force exact KNN (plan.md Phase 3).

Uses bge-small (the dev workhorse) so the full (m, ef_construct) x ef_search
grid runs in a reasonable time — bge-large/BGE-M3 get a single sensible
config chosen from this curve, not their own sweep.

Query set: 300 vectors sampled from the corpus itself (self-match excluded
from both exact and approximate result sets before scoring), not Phase 4's
golden set — that doesn't exist until the next phase. This measures HNSW's
fidelity to the exact nearest-neighbor structure of the embedding space
itself, not retrieval quality for real questions (Phase 6's job).
"""

from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
from qdrant_client.models import SearchParams

from aegis.config import settings
from aegis.embedding.knn import brute_force_knn_batch, recall_at_k, recall_at_k_score_aware
from aegis.embedding.vector_store import DENSE_VECTOR_NAME, create_collection, get_client, upsert_chunks
from aegis.types import Chunk

K = 10
N_QUERIES = 300
RANDOM_SEED = 20260730
M_VALUES = [8, 16, 32]
EF_CONSTRUCT_VALUES = [50, 100, 200]
EF_SEARCH_VALUES = [50, 100, 200, 400]

RESULTS_MD = settings.paths.data_processed.parent.parent / "docs" / "results" / "ann-recall-vs-latency.md"
RESULTS_PNG = settings.paths.data_processed.parent.parent / "docs" / "results" / "ann-recall-vs-latency.png"


def load_chunks() -> list[Chunk]:
    chunks = []
    with (settings.paths.data_processed / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk(**json.loads(line)))
    return chunks


def wait_until_indexed(client, collection_name: str, timeout_s: float = 60.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        info = client.get_collection(collection_name)
        if str(info.status) in ("CollectionStatus.GREEN", "green"):
            return
        time.sleep(0.2)


def main() -> None:
    chunks = load_chunks()
    vectors = np.load(settings.paths.data_processed / "vectors_bge_small.npy")
    print(f"Loaded {len(chunks)} chunks, vectors shape={vectors.shape}")

    rng = np.random.default_rng(RANDOM_SEED)
    query_idx = rng.choice(len(vectors), size=N_QUERIES, replace=False)
    queries = vectors[query_idx]

    print("Computing exact KNN ground truth (brute force)...")
    exact_raw = brute_force_knn_batch(queries, vectors, k=K + 1)
    exact = [[int(idx) for idx in row if idx != qi][:K] for qi, row in zip(query_idx, exact_raw)]

    # Score-threshold ground truth (handles the corpus's real duplicate-
    # embedding ties correctly — see recall_at_k's docstring). Computed once
    # as a full (N_QUERIES, corpus) score matrix; self-similarity is masked
    # out before taking the k-th-best score per query.
    print("Computing score-threshold ground truth...")
    full_scores = queries @ vectors.T
    for row, qi in enumerate(query_idx):
        full_scores[row, qi] = -np.inf
    exact_kth_scores = -np.partition(-full_scores, K - 1, axis=1)[:, K - 1]

    dup_fraction = 1 - len(np.unique(np.round(vectors, decimals=5), axis=0)) / len(vectors)
    print(f"Corpus duplicate-embedding fraction (rounded to 5dp): {dup_fraction:.1%}")

    client = get_client()
    results: list[dict] = []

    for m in M_VALUES:
        for ef_construct in EF_CONSTRUCT_VALUES:
            collection_name = f"aegis_ann_sweep_m{m}_efc{ef_construct}"
            print(f"Building m={m} ef_construct={ef_construct}...")
            t0 = time.time()
            create_collection(client, collection_name, dim=vectors.shape[1], hnsw_m=m, hnsw_ef_construct=ef_construct)
            upsert_chunks(client, collection_name, chunks, vectors)
            wait_until_indexed(client, collection_name)
            build_time = time.time() - t0

            for ef_search in EF_SEARCH_VALUES:
                approx_ids = []
                approx_scores = []
                t0 = time.time()
                for qi, qvec in zip(query_idx, queries):
                    hits = client.query_points(
                        collection_name=collection_name,
                        query=qvec.tolist(),
                        using=DENSE_VECTOR_NAME,
                        limit=K + 1,
                        search_params=SearchParams(hnsw_ef=ef_search),
                    ).points
                    filtered = [h for h in hits if h.id != qi][:K]
                    approx_ids.append([h.id for h in filtered])
                    approx_scores.append([h.score for h in filtered])
                latency_ms = (time.time() - t0) / N_QUERIES * 1000

                r_naive = recall_at_k(approx_ids, exact)
                r_score = recall_at_k_score_aware(approx_scores, list(exact_kth_scores), k=K)
                results.append(
                    {
                        "m": m,
                        "ef_construct": ef_construct,
                        "ef_search": ef_search,
                        "recall_at_10_id": r_naive,
                        "recall_at_10_score": r_score,
                        "latency_ms": latency_ms,
                        "build_time_s": round(build_time, 2),
                    }
                )
                print(f"  ef_search={ef_search}: recall_id={r_naive:.4f} recall_score={r_score:.4f} latency={latency_ms:.2f}ms")

            client.delete_collection(collection_name)

    # --- Plot: recall vs latency, one line per (m, ef_construct) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in M_VALUES:
        for ef_construct in EF_CONSTRUCT_VALUES:
            subset = [r for r in results if r["m"] == m and r["ef_construct"] == ef_construct]
            subset.sort(key=lambda r: r["latency_ms"])
            ax.plot(
                [r["latency_ms"] for r in subset],
                [r["recall_at_10_score"] for r in subset],
                marker="o",
                label=f"m={m}, ef_construct={ef_construct}",
            )
    ax.set_xlabel("Latency (ms/query, avg over 300 queries)")
    ax.set_ylabel("Recall@10 (score-threshold, tie-aware)")
    ax.set_title(f"HNSW recall vs. latency — bge-small, {len(chunks)} chunks")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_PNG, dpi=150)
    print(f"Wrote {RESULTS_PNG}")

    # --- Markdown table ---
    lines = [
        "# HNSW Recall vs. Latency — Phase 3",
        "",
        f"Encoder: `{settings.embedding.dev_model}` (dev workhorse — bge-large/BGE-M3 use a config chosen "
        "from this curve, not their own sweep). Corpus: all 19,236 heading-aware chunks. "
        f"Query set: {N_QUERIES} vectors sampled from the corpus itself (self-match excluded), "
        "since Phase 4's golden set doesn't exist yet — this measures HNSW's fidelity to the exact "
        "nearest-neighbor structure, not retrieval quality for real questions (Phase 6).",
        "",
        f"**{dup_fraction:.1%} of the corpus's 19,236 vectors are exact-duplicate embeddings** "
        "(rounded to 5 decimal places) — the largest single group is 1,354 issues whose entire "
        "heading-aware chunk is the literal string `_No response_`, a GitHub issue-template "
        "placeholder (same boilerplate-pollution category as Phase 1's product_area findings). "
        "With that many exact ties, \"the\" top-10 nearest neighbors is not uniquely defined: "
        "naive ID-set recall (`Recall@10 (ID)` below) penalizes HNSW for returning a *different, "
        "equally-correct* tied neighbor than the arbitrary one brute-force KNN happened to return. "
        "`Recall@10 (score)` is the corrected metric — it credits any result whose similarity score "
        "matches or beats the true k-th-best score, tie or not — and is the one the plot uses.",
        "",
        "![Recall vs latency](ann-recall-vs-latency.png)",
        "",
        "| m | ef_construct | ef_search | Recall@10 (ID) | Recall@10 (score) | Latency (ms) | Build time (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: (r["m"], r["ef_construct"], r["ef_search"])):
        lines.append(
            f"| {r['m']} | {r['ef_construct']} | {r['ef_search']} | {r['recall_at_10_id']:.4f} | "
            f"{r['recall_at_10_score']:.4f} | {r['latency_ms']:.2f} | {r['build_time_s']} |"
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {RESULTS_MD}")


if __name__ == "__main__":
    main()
