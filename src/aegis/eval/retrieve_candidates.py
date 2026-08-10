"""Permissive top-K retrieval (plan.md Phase 4 step 2) over every harvested
positive candidate — the pool the human labeler picks true ground-truth
chunks from. "Permissive": plain dense search, no metadata filter, over one
of the two real bake-off encoders, not `bge_small` (Phase 3's dev-only
workhorse) — see `GoldenSettings.permissive_model_key`.

I/O-bound glue over already-built components (`Encoder`, the Qdrant client)
— integration-tested by actually running it, per CLAUDE.md §6, the same
treatment as `aegis.embedding.run`.
"""

from __future__ import annotations

import json

from qdrant_client.models import FieldCondition, Filter, MatchValue

from aegis.config import settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.embedding.vector_store import DENSE_VECTOR_NAME, get_client

_MODEL_NAMES = {
    "bge_small": settings.embedding.dev_model,
    "bge_large": settings.embedding.bge_large_model,
    "bge_m3": settings.embedding.bge_m3_model,
}
_COLLECTIONS = {
    "bge_small": settings.qdrant.collection_bge_small,
    "bge_large": settings.qdrant.collection_bge_large,
    "bge_m3": settings.qdrant.collection_bge_m3,
}


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _hits_to_dicts(hits) -> list[dict]:
    return [
        {
            "chunk_id": h.payload["chunk_id"],
            "doc_id": h.payload["doc_id"],
            "doc_type": h.payload["doc_type"],
            "heading_path": h.payload["heading_path"],
            "product_area": h.payload["product_area"],
            "text": h.payload["text"],
            "score": h.score,
        }
        for h in hits
    ]


def retrieve_for_candidates(candidates: list[dict], top_k_docs: int, top_k_issues: int, model_key: str) -> dict[str, list[dict]]:
    """Two doc_type-filtered searches per candidate, not one flat top-K —
    see GoldenSettings.permissive_top_k_docs for why a single unfiltered
    search under-represents documentation on this corpus.

    Every candidate's own source issue is excluded (`must_not` on `doc_id`)
    — positives are harvested from the same corpus that's indexed, so
    without this the source issue's own chunk is a near-verbatim match of
    the query and dominates every result (measured: score 0.958 vs. 0.7-0.8
    for everything else). Selecting it as "ground truth" would be circular:
    any embedding trivially finds a near-duplicate of its own input, which
    inflates Phase 6's retrieval numbers on a case that can't occur for a
    real, previously-unseen user query.
    """
    model_name = _MODEL_NAMES[model_key]
    collection_name = _COLLECTIONS[model_key]

    cache = EmbeddingCache(settings.paths.embedding_cache)
    encoder = Encoder(model_name=model_name, cache=cache)
    vectors = encoder.encode([c["text"] for c in candidates])

    client = get_client()
    results: dict[str, list[dict]] = {}
    for candidate, vector in zip(candidates, vectors):
        exclude_self = [FieldCondition(key="doc_id", match=MatchValue(value=candidate["source_issue"]))]
        doc_hits = client.query_points(
            collection_name=collection_name,
            query=vector.tolist(),
            using=DENSE_VECTOR_NAME,
            limit=top_k_docs,
            query_filter=Filter(
                must=[FieldCondition(key="doc_type", match=MatchValue(value="documentation"))], must_not=exclude_self
            ),
        ).points
        issue_hits = client.query_points(
            collection_name=collection_name,
            query=vector.tolist(),
            using=DENSE_VECTOR_NAME,
            limit=top_k_issues,
            query_filter=Filter(
                must=[FieldCondition(key="doc_type", match=MatchValue(value="issue"))], must_not=exclude_self
            ),
        ).points
        results[candidate["id"]] = _hits_to_dicts(doc_hits) + _hits_to_dicts(issue_hits)
    return results


def main() -> None:
    positives_path = settings.paths.data_golden / "candidates_positive.jsonl"
    candidates = load_jsonl(positives_path)
    top_k_docs, top_k_issues = settings.golden.permissive_top_k_docs, settings.golden.permissive_top_k_issues
    print(
        f"Retrieving top-{top_k_docs} docs + top-{top_k_issues} issues for {len(candidates)} candidates "
        f"via {settings.golden.permissive_model_key}..."
    )

    results = retrieve_for_candidates(candidates, top_k_docs, top_k_issues, settings.golden.permissive_model_key)

    out_path = settings.paths.data_golden / "retrieval_candidates.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for query_id, hits in results.items():
            f.write(json.dumps({"query_id": query_id, "candidates": hits}) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
