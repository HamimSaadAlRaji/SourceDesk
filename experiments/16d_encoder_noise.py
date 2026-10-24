"""Phase 16, unit 16c (design record 16c): how much do the retrieval metrics move
when only the query *encoding* is perturbed at float-rounding scale?

Same model, same machine, same restored snapshot; only the encode batch size
changes (different padding/BLAS paths -> vectors that differ in the ~1e-6 range).
This is the closest local proxy for the M1 -> x86 runner difference, and the
evidence for the gate's tolerance band. Needs the snapshot restored in a key-less
Qdrant on :6335 (see experiments/16c_snapshot_restore_check.py) and
QDRANT_PORT=6335 QDRANT_API_KEY= in the environment. Zero LLM cost.
"""

from __future__ import annotations

import json

import numpy as np

from aegis.config import REPO_ROOT, settings
from aegis.embedding.encoder import Encoder
from aegis.embedding.vector_store import get_client
from aegis.eval.harness import RetrievalConfig, run_eval
from aegis.eval.retrieval_gate import load_positive_dev_queries
from aegis.factory import load_chunks
from aegis.retrieval.dense import DenseIndex

BATCH_SIZES = [1, 4, 16, 32, 64]
RAW = REPO_ROOT / "docs" / "results" / "encoder-noise-raw.json"


def main() -> None:
    chunks, queries, client = load_chunks(), load_positive_dev_queries(), get_client()
    cfg = RetrievalConfig(name="noise", chunking_strategy="heading_aware", contextual_enrichment=False,
                          retrieval_strategy="dense", encoder="bge_large")
    texts = [q.text for q in queries]
    ref_vectors = None
    rows = []
    for bs in BATCH_SIZES:
        enc = Encoder(model_name=settings.embedding.bge_large_model, cache=None, device="cpu", batch_size=bs)
        vectors = enc.encode(texts).astype(np.float32)
        if ref_vectors is None:
            ref_vectors = vectors
        by_text = {t: v for t, v in zip(texts, vectors)}

        class _P:
            def encode(self, ts, by_text=by_text):  # noqa: ANN001
                return np.stack([by_text[t] for t in ts])

        idx = DenseIndex(chunks, settings.qdrant.collection_bge_large, _P(), client=client)
        res = run_eval(cfg, queries, lambda q, idx=idx: [(c.id, s) for c, s in idx.search(q.text, top_k=10, exclude_doc_id=q.source_issue)],
                       max_workers=1)
        cos = (vectors * ref_vectors).sum(axis=1)
        rows.append({"batch_size": bs, "max_abs_vector_diff": float(np.abs(vectors - ref_vectors).max()),
                     "min_cosine_to_ref": float(cos.min()),
                     **{k: res.overall[k] for k in ("recall@10", "mrr", "ndcg@10")}})
        print(rows[-1])
    spread = {m: float(max(r[m] for r in rows) - min(r[m] for r in rows)) for m in ("recall@10", "mrr", "ndcg@10")}
    print("spread (max-min):", spread)
    RAW.write_text(json.dumps({"rows": rows, "spread": spread}, indent=2))


if __name__ == "__main__":
    main()
