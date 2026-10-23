"""Phase 15 unit 15d: does boundary redaction change what retrieval returns for
the dev queries it touches? Zero LLM calls (served dense path, bge-large CPU,
real Qdrant). Only queries `redact` changes are re-run; the rest are
untouched by construction.

    uv run python experiments/15d_redaction_retrieval_impact.py
"""

from __future__ import annotations

import json

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.factory import load_chunks
from aegis.guardrails.input_boundary import redact
from aegis.retrieval.dense import DenseIndex

OUT = REPO_ROOT / "docs" / "results" / "redaction-retrieval-impact-raw.json"


def main() -> None:
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        dev = [json.loads(line) for line in f]
    touched = [(q, redact(q["text"])) for q in dev]
    touched = [(q, r) for q, r in touched if r[1]]
    dense = DenseIndex(load_chunks(), settings.qdrant.collection_bge_large,
                       Encoder(model_name=settings.embedding.bge_large_model,
                               cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu"))
    rows = []
    for q, (red, kinds) in touched:
        before = [c.id for c, _ in dense.search(q["text"], top_k=10)]
        after = [c.id for c, _ in dense.search(red, top_k=10)]
        gt = set(q.get("ground_truth_chunk_ids") or [])
        rows.append({"query_id": q["id"], "kinds": kinds, "chars": len(q["text"]),
                     "top10_overlap": len(set(before) & set(after)), "top1_same": before[:1] == after[:1],
                     "gt_hits_before": len(gt & set(before)), "gt_hits_after": len(gt & set(after))})
    out = {"n_dev": len(dev), "n_touched": len(rows), "rows": rows}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
