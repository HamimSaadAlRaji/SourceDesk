"""Phase 16, unit 16i (design record 16i): export every (query, chunk) pair the CI
escalation check could need, for scoring on Kaggle.

Why Kaggle: the cross-encoder costs 1.32 s per pair on this M1 (measured on 12 real
pairs, 2026-10-25), so the ~540-900 pairs here are ~12-20 minutes of CPU, over the 10-minute
local limit (CLAUDE.md section 10). Kaggle is NOT connected to this session: you must paste
`16f_kaggle_rerank_pairs.py` into a Kaggle notebook and run it yourself.

The workload is the confidence scorer's input for the 180 dev queries: the top
`rerank_pool_size` candidates of the dense search. To keep CI from missing the cache when
a different CPU reorders near-ties, this exports the top-`EXPORT_DEPTH` (5) candidates, from
every index given, deduplicated. Also writes 12 locally scored `reference` pairs so the
Kaggle result can be checked against this machine's scores.

    uv run python experiments/16e_export_rerank_pairs.py [--extra-qdrant-port 6342]

Output: ci-fixtures/rerank_pairs_export.json (gitignored; upload it to the Kaggle notebook).
"""

from __future__ import annotations

import argparse
import json

from qdrant_client import QdrantClient

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.embedding.vector_store import get_client
from aegis.eval.rerank_cache import pair_key
from aegis.factory import load_chunks
from aegis.rerank.cross_encoder import CrossEncoderReranker
from aegis.retrieval.dense import DenseIndex
from aegis.types import GoldenQuery

EXPORT_DEPTH = 5
N_REFERENCE = 12
OUT = REPO_ROOT / "ci-fixtures" / "rerank_pairs_export.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-qdrant-port", type=int, action="append", default=[],
                        help="a second, key-less Qdrant (e.g. the CI-style restored snapshot) to union candidates from")
    args = parser.parse_args()

    chunks = load_chunks()
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        queries = [GoldenQuery(**json.loads(line)) for line in f]
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
    clients: list[QdrantClient] = [get_client()] + [
        QdrantClient(host="localhost", port=p) for p in args.extra_qdrant_port
    ]

    model = settings.rerank.model_name
    pairs: dict[str, dict[str, str]] = {}
    per_client_top3 = []
    for client in clients:
        index = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder, client=client)
        top3 = set()
        for q in queries:
            hits = index.search(q.text, top_k=settings.generation.retrieval_top_k, exclude_doc_id=q.source_issue)
            for rank, (chunk, _score) in enumerate(hits[:EXPORT_DEPTH]):
                key = pair_key(model, q.text, chunk.text)
                entry = pairs.setdefault(key, {"key": key, "query": q.text, "chunk_text": chunk.text, "min_rank": rank})
                entry["min_rank"] = min(entry["min_rank"], rank)  # 0 = top hit; the gate needs rank < rerank_pool_size
                if rank < settings.confidence.rerank_pool_size:
                    top3.add(key)
        per_client_top3.append(top3)
    union_top3 = set().union(*per_client_top3)
    print(f"{len(queries)} queries; {len(clients)} index(es); exported {len(pairs)} distinct pairs "
          f"(top-{EXPORT_DEPTH}); distinct top-{settings.confidence.rerank_pool_size} pairs across indexes: {len(union_top3)}")

    # Reference scores computed on THIS machine, for the Kaggle script to compare against.
    score_fn = CrossEncoderReranker(device="cpu")._score_fn
    ref_items = list(pairs.values())[:: max(1, len(pairs) // N_REFERENCE)][:N_REFERENCE]
    ref_scores = score_fn([(p["query"], p["chunk_text"]) for p in ref_items])
    reference = {p["key"]: float(s) for p, s in zip(ref_items, ref_scores)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model_name": model, "pairs": list(pairs.values()), "reference": reference}))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB), {len(reference)} reference pairs")


if __name__ == "__main__":
    main()
