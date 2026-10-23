"""Phase 15 unit 15b "after" measurements on real data (design record 15b).
Zero LLM calls: bge-large on CPU (embedding cache), real Qdrant, real BM25.

  1. A1 re-measured with the same protocol as 15a: HybridRetriever,
     doc_type=documentation, 180 dev queries, top-10 -> issue-chunk slots
     (before: 609/1,800).
  2. Bypass sweep: for each disjoint partition (docs, issues), every dev query
     plus the bypass-shaped queries from tests/security, through
     PartitionedRetriever over (a) the served DenseIndex and (b) the hybrid
     retriever. Counts out-of-partition chunks, PartitionViolations raised,
     and short result lists.

    uv run python experiments/15b_partition_bypass.py
"""

from __future__ import annotations

import importlib.util
import json
import time

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.factory import load_chunks
from aegis.retrieval.dense import DenseIndex
from aegis.retrieval.hybrid import HybridRetriever, RetrievalStrategy
from aegis.retrieval.sparse import SparseIndex
from aegis.tenancy import PartitionedRetriever, PartitionViolation, resolve_partition
from aegis.types import DocType

OUT = REPO_ROOT / "docs" / "results" / "partition-bypass-raw.json"
TOP_K = 10


def _bypass_queries(chunks) -> list[str]:
    path = REPO_ROOT / "tests" / "security" / "test_attack_tenant_bypass.py"
    spec = importlib.util.spec_from_file_location("bypass", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.bypass_queries(chunks)


def main() -> None:
    t0 = time.perf_counter()
    chunks = load_chunks()
    encoder = Encoder(model_name=settings.embedding.bge_large_model,
                      cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
    dense = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)
    hybrid = HybridRetriever(sparse_index=SparseIndex(chunks), dense_index=dense)
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        dev = [json.loads(line)["text"] for line in f]
    bypass = _bypass_queries(chunks)

    issue_slots = 0
    for q in dev:
        res = hybrid.search(q, top_k=TOP_K, strategy=RetrievalStrategy.HYBRID_RRF, doc_type=DocType.DOCUMENTATION)
        issue_slots += sum(c.doc_type == DocType.ISSUE for c, _ in res)
    a1 = {"n_queries": len(dev), "issue_slots": issue_slots, "slots": len(dev) * TOP_K, "before": "609/1800"}

    sweep = {}
    for name in ("docs", "issues"):
        partition = resolve_partition(name)
        for label, base in (("dense", dense), ("hybrid", hybrid)):
            violations = raised = short = 0
            for q in bypass + dev:
                try:
                    res = PartitionedRetriever(base, partition).search(q, top_k=TOP_K)
                except PartitionViolation:
                    raised += 1
                    continue
                violations += sum(c.doc_type not in partition.doc_types for c, _ in res)
                short += len(res) < TOP_K
            sweep[f"{name}/{label}"] = {"n_queries": len(bypass) + len(dev), "n_bypass_queries": len(bypass),
                                        "out_of_partition_chunks": violations,
                                        "partition_violations_raised": raised, "short_result_lists": short}
    out = {"A1_after": a1, "bypass_sweep": sweep, "elapsed_s": round(time.perf_counter() - t0, 1)}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
