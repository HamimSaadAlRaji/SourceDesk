"""Phase 16, unit 16b (design record 16b): can CI rebuild the served index from
`chunks.jsonl` + `vectors_bge_large.npy`, and does the rebuilt index score the
same as the one every earlier phase measured?

Zero LLM calls, zero spend. Answers four questions with numbers:
  1. How long do restore + HNSW build take (the cost CI pays on every run)?
  2. How long does CPU query encoding take, with the embedding cache bypassed?
  3. Is a rebuild reproducible -- do two rebuilds from identical files return the
     same top-10 lists? (If not, a gate threshold must absorb that noise.)
  4. How far is the rebuilt index from the live collection and from exact cosine?

Throwaway collections only (`_ci_restore_check_a/_b`); the real `aegis_chunks_*`
collections are read, never written. Local timings are a lower bound for a 2-core
hosted runner and are labelled as such in the record.
"""

from __future__ import annotations

import json
import time

import numpy as np

from aegis.config import REPO_ROOT, settings
from aegis.embedding.encoder import Encoder
from aegis.embedding.vector_store import get_client
from aegis.eval.ci_index import check_alignment, corpus_fingerprint, restore_index
from aegis.eval.harness import RetrievalConfig, run_eval
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, GoldenQuery

TOP_K = 10
COLLECTIONS = ("_ci_restore_check_a", "_ci_restore_check_b")
RAW = REPO_ROOT / "docs" / "results" / "ci-index-restore-raw.json"


def load_chunks() -> list[Chunk]:
    with open(settings.paths.data_processed / "chunks.jsonl", encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def load_positives() -> list[GoldenQuery]:
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        return [q for q in (GoldenQuery(**json.loads(line)) for line in f) if not q.is_negative]


class _Precomputed:
    """Hands DenseIndex the already-encoded query vector, so each index is
    scored on the *same* vectors and encoding time is measured once, separately."""

    def __init__(self, by_text: dict[str, np.ndarray]) -> None:
        self._by_text = by_text

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._by_text[t] for t in texts])


def _score(name: str, collection: str, chunks, queries, encoder, client) -> tuple[dict, dict[str, list[str]], list[float]]:
    retrieved: dict[str, list[str]] = {}
    latencies: list[float] = []
    index = DenseIndex(chunks, collection, encoder, client=client)

    def search(q: GoldenQuery) -> list[tuple[str, float]]:
        t0 = time.perf_counter()
        hits = index.search(q.text, top_k=TOP_K, exclude_doc_id=q.source_issue)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved[q.id] = [c.id for c, _ in hits]
        return [(c.id, s) for c, s in hits]

    cfg = RetrievalConfig(
        name=name, chunking_strategy="heading_aware", contextual_enrichment=False,
        retrieval_strategy="dense", encoder="bge_large",
    )
    result = run_eval(cfg, queries, search, max_workers=1)  # serial: honest per-query latency
    return result.overall, retrieved, latencies


def _exact(queries, chunks, vectors, query_vectors) -> tuple[dict[str, list[str]], dict]:
    from aegis.eval.metrics import build_qrels, build_run, compute_metrics

    ids = np.array([c.id for c in chunks])
    doc_ids = np.array([c.doc_id for c in chunks])
    retrieved: dict[str, list[str]] = {}
    for q, qv in zip(queries, query_vectors):
        scores = vectors @ qv
        scores[doc_ids == q.source_issue] = -np.inf
        retrieved[q.id] = [str(ids[i]) for i in np.argsort(-scores)[:TOP_K]]
    qrels = build_qrels({q.id: q.ground_truth_chunk_ids for q in queries})
    run = build_run({q.id: [(cid, float(TOP_K - r)) for r, cid in enumerate(retrieved[q.id])] for q in queries})
    return retrieved, compute_metrics(qrels, run)


def _overlap(a: dict[str, list[str]], b: dict[str, list[str]]) -> float:
    return float(np.mean([len(set(a[k]) & set(b[k])) / TOP_K for k in a]))


def main() -> None:
    chunks = load_chunks()
    queries = load_positives()
    vectors = np.load(settings.paths.data_processed / "vectors_bge_large.npy").astype(np.float32)
    check_alignment(chunks, vectors, expected_dim=vectors.shape[1])
    fingerprint = corpus_fingerprint(chunks, vectors)
    print(f"{len(chunks)} chunks, {vectors.shape} vectors, {len(queries)} positive queries, fingerprint {fingerprint}")

    # 2. Query encoding, cache bypassed (cache=None): the CI cost if the embedding cache is cold.
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=None, device="cpu")
    t0 = time.perf_counter()
    encoder.warm_up("warm up")
    model_load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    query_vectors = encoder.encode([q.text for q in queries]).astype(np.float32)
    encode_s = time.perf_counter() - t0
    print(f"model load+first pass {model_load_s:.1f}s; encoded {len(queries)} queries in {encode_s:.1f}s")
    pre = _Precomputed({q.text: v for q, v in zip(queries, query_vectors)})

    client = get_client()  # admin key: writes the throwaway collections only
    out: dict = {
        "fingerprint": fingerprint, "n_chunks": len(chunks), "n_queries": len(queries),
        "query_encode": {"model_load_s": model_load_s, "encode_s": encode_s, "per_query_ms": encode_s / len(queries) * 1000},
        "restore": {}, "metrics": {}, "latency_ms": {}, "overlap": {},
    }
    runs: dict[str, dict[str, list[str]]] = {}
    try:
        for name in COLLECTIONS:
            timing = restore_index(client, name, chunks, vectors)
            print(f"{name}: upsert {timing['upsert_s']:.1f}s, index wait {timing['index_wait_s']:.1f}s")
            out["restore"][name] = timing
            overall, retrieved, lat = _score(name, name, chunks, queries, pre, client)
            out["metrics"][name], runs[name] = overall, retrieved
            out["latency_ms"][name] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}

        overall, retrieved, lat = _score("live", settings.qdrant.collection_bge_large, chunks, queries, pre, client)
        out["metrics"]["live"], runs["live"] = overall, retrieved
        out["latency_ms"]["live"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}

        runs["exact"], out["metrics"]["exact"] = _exact(queries, chunks, vectors, query_vectors)
    finally:
        for name in COLLECTIONS:
            client.delete_collection(name)

    a, b = COLLECTIONS
    for label, x, y in [("rebuild_a_vs_b", a, b), ("rebuild_a_vs_live", a, "live"), ("rebuild_a_vs_exact", a, "exact"),
                        ("live_vs_exact", "live", "exact")]:
        out["overlap"][label] = _overlap(runs[x], runs[y])

    RAW.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("restore", "metrics", "latency_ms", "overlap")}, indent=2))


if __name__ == "__main__":
    main()
