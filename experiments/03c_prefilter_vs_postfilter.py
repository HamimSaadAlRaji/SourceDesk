"""Pre-filter vs. post-filter metadata filtering (plan.md Phase 3) — measured,
not assumed.

Post-filtering (search top-K unfiltered by vector similarity, then discard
results that fail a metadata check) can silently return fewer results than
asked for — or none at all — when the filter targets a narrow category and
none of the unfiltered top-K happen to belong to it. Qdrant's native
pre-filtering (`query_filter`, payload-index-aware) searches *within* the
filtered subset directly and doesn't have this failure mode. The question is
how often it actually matters on this corpus, across category sizes from
huge (`general`, 5,081 chunks) to narrow (`graphql`, 56 chunks).
"""

from __future__ import annotations

import json

import numpy as np
from qdrant_client.models import FieldCondition, Filter, MatchValue

from aegis.config import settings
from aegis.embedding.knn import recall_at_k_score_aware
from aegis.embedding.vector_store import DENSE_VECTOR_NAME, get_client
from aegis.types import Chunk

K = 10
N_QUERIES_PER_CATEGORY = 30
POST_FILTER_POOL = 10  # a naive post-filter step searching "top-10, then filter" — the common real mistake
RANDOM_SEED = 20260730
CATEGORIES = ["general", "cors", "background-tasks", "how-to", "graphql"]

RESULTS_MD = settings.paths.data_processed.parent.parent / "docs" / "results" / "prefilter-vs-postfilter.md"


def load_chunks() -> list[Chunk]:
    chunks = []
    with (settings.paths.data_processed / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk(**json.loads(line)))
    return chunks


def main() -> None:
    chunks = load_chunks()
    vectors = np.load(settings.paths.data_processed / "vectors_bge_small.npy")
    product_areas = np.array([c.product_area for c in chunks])
    collection_name = settings.qdrant.collection_bge_small
    client = get_client()

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for category in CATEGORIES:
        category_indices = np.where(product_areas == category)[0]
        n_in_corpus = len(category_indices)
        n_q = min(N_QUERIES_PER_CATEGORY, n_in_corpus)
        query_idx = rng.choice(category_indices, size=n_q, replace=False)
        print(f"{category}: {n_in_corpus} chunks in corpus, querying {n_q}")

        # Exact ground truth *within the category*: brute-force KNN restricted
        # to category members only (excluding self).
        category_vectors = vectors[category_indices]
        query_vectors = vectors[query_idx]
        full_scores = query_vectors @ category_vectors.T
        for row, qi in enumerate(query_idx):
            local_self = np.where(category_indices == qi)[0]
            if len(local_self):
                full_scores[row, local_self[0]] = -np.inf
        kth = min(K, n_in_corpus - 1)
        exact_kth_scores = -np.partition(-full_scores, kth - 1, axis=1)[:, kth - 1]

        post_scores, pre_scores = [], []
        post_short_count = 0

        for row, qi in enumerate(query_idx):
            qvec = vectors[qi]

            # Naive post-filter: unfiltered top-POST_FILTER_POOL, then discard non-matches.
            hits = client.query_points(
                collection_name=collection_name, query=qvec.tolist(), using=DENSE_VECTOR_NAME, limit=POST_FILTER_POOL + 1
            ).points
            post_filtered = [h for h in hits if h.id != qi and h.payload["product_area"] == category][:K]
            if len(post_filtered) < kth:
                post_short_count += 1
            post_scores.append([h.score for h in post_filtered])

            # Native pre-filter.
            pre_hits = client.query_points(
                collection_name=collection_name,
                query=qvec.tolist(),
                using=DENSE_VECTOR_NAME,
                limit=K + 1,
                query_filter=Filter(must=[FieldCondition(key="product_area", match=MatchValue(value=category))]),
            ).points
            pre_filtered = [h for h in pre_hits if h.id != qi][:K]
            pre_scores.append([h.score for h in pre_filtered])

        post_recall = recall_at_k_score_aware(post_scores, list(exact_kth_scores), k=kth)
        pre_recall = recall_at_k_score_aware(pre_scores, list(exact_kth_scores), k=kth)
        short_pct = 100 * post_short_count / n_q

        rows.append(
            {
                "category": category,
                "corpus_size": n_in_corpus,
                "n_queries": n_q,
                "post_filter_recall": post_recall,
                "pre_filter_recall": pre_recall,
                "post_filter_short_pct": short_pct,
            }
        )
        print(
            f"  post-filter recall={post_recall:.4f} (short of {kth} results on "
            f"{short_pct:.0f}% of queries) | pre-filter recall={pre_recall:.4f}"
        )

    lines = [
        "# Pre-filter vs. Post-filter Metadata Filtering — Phase 3",
        "",
        f"Encoder: `{settings.embedding.dev_model}`. For each `product_area` category, "
        f"{N_QUERIES_PER_CATEGORY} queries sampled from within that category (or all available "
        "members if fewer). Ground truth: exact top-k *within the category* via brute-force KNN "
        "restricted to that category's vectors. \"Post-filter\" = the common naive "
        f"implementation — search top-{POST_FILTER_POOL} unfiltered, then discard results that "
        "don't match the category. \"Pre-filter\" = Qdrant's native `query_filter`.",
        "",
        "| Category | Corpus size | Post-filter recall | Post-filter short of k on | Pre-filter recall |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['category']} | {r['corpus_size']} | {r['post_filter_recall']:.4f} | "
            f"{r['post_filter_short_pct']:.0f}% of queries | {r['pre_filter_recall']:.4f} |"
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {RESULTS_MD}")


if __name__ == "__main__":
    main()
