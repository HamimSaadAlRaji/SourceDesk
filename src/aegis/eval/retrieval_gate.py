"""The retrieval half of the CI gate (design record 16c): score the restored index
on the frozen dev golden set and compare to a committed baseline.

    python -m aegis.eval.retrieval_gate run     --fixtures ci-fixtures --out results/retrieval.json
    python -m aegis.eval.retrieval_gate compare --current results/retrieval.json [--markdown results/retrieval.md]
    python -m aegis.eval.retrieval_gate baseline --current results/retrieval.json    # deliberate rebaseline

Queries are encoded live with the embedding cache *bypassed*: a cached vector
would hide a regression in the encoder path (prefixing, truncation, model name),
which is exactly what a change to this code could break. The price is ~40 s of
CPU encoding on an M1 Pro, measured in design record 16b.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from aegis.config import settings
from aegis.embedding.encoder import Encoder
from aegis.embedding.vector_store import get_client
from aegis.eval.ci_fixtures import verify_fixtures
from aegis.eval.gate import GateReport, MetricSpec, evaluate, render_markdown
from aegis.eval.harness import RetrievalConfig, run_eval
from aegis.factory import load_chunks
from aegis.retrieval.dense import DenseIndex
from aegis.types import GoldenQuery

TOP_K = 10
BASELINE_NAME = "retrieval.json"


def gate_specs() -> list[MetricSpec]:
    g = settings.gate
    return [
        MetricSpec(name="recall@10", tolerance=g.retrieval_recall_tolerance),
        MetricSpec(name="mrr", tolerance=g.retrieval_mrr_tolerance),
        MetricSpec(name="ndcg@10", tolerance=g.retrieval_ndcg_tolerance),
    ]


def load_positive_dev_queries() -> list[GoldenQuery]:
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        queries = [GoldenQuery(**json.loads(line)) for line in f]
    return [q for q in queries if not q.is_negative and q.split == "dev"]


def run(fixtures_dir: Path) -> GateReport:
    fingerprint = verify_fixtures(fixtures_dir).fingerprint
    chunks = load_chunks()
    queries = load_positive_dev_queries()

    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=None, device="cpu")
    encoder.warm_up("warm up")  # model load is a fixed cost, reported apart from per-query encoding
    t0 = time.perf_counter()
    vectors = encoder.encode([q.text for q in queries]).astype(np.float32)
    encode_s = time.perf_counter() - t0
    by_text = {q.text: v for q, v in zip(queries, vectors)}

    class _Precomputed:
        def encode(self, texts: list[str]) -> np.ndarray:
            return np.stack([by_text[t] for t in texts])

    index = DenseIndex(chunks, settings.qdrant.collection_bge_large, _Precomputed(), client=get_client())  # type: ignore[arg-type]
    latencies: list[float] = []

    def search(q: GoldenQuery) -> list[tuple[str, float]]:
        start = time.perf_counter()
        hits = index.search(q.text, top_k=TOP_K, exclude_doc_id=q.source_issue)
        latencies.append((time.perf_counter() - start) * 1000)
        return [(c.id, s) for c, s in hits]

    config = RetrievalConfig(
        name="ci-gate", chunking_strategy="heading_aware", contextual_enrichment=False,
        retrieval_strategy="dense", encoder="bge_large", top_k=TOP_K,
    )
    result = run_eval(config, queries, search, max_workers=1)  # serial: honest per-query latency
    metrics = dict(result.overall)
    metrics["search_p95_ms"] = float(np.percentile(latencies, 95))  # recorded, not gated (16d decides)
    metrics["encode_s_per_query"] = encode_s / len(queries)
    return GateReport(suite="retrieval", fingerprint=fingerprint, n_queries=result.n_queries, metrics=metrics)


def _baseline_path() -> Path:
    return settings.gate.baseline_dir / BASELINE_NAME


def compare(current_path: Path, markdown_path: Path | None) -> int:
    current = GateReport.model_validate_json(current_path.read_text())
    baseline_path = _baseline_path()
    if not baseline_path.exists():
        print(f"no baseline at {baseline_path}: a gate with nothing to compare against cannot pass", file=sys.stderr)
        return 1
    baseline = GateReport.model_validate_json(baseline_path.read_text())
    result = evaluate(current, baseline, gate_specs())
    md = render_markdown("Retrieval (dense bge-large, 150 dev queries)", result)
    print(md)
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(md)
    return 0 if result.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--fixtures", type=Path, required=True)
    p_run.add_argument("--out", type=Path, required=True)
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--current", type=Path, required=True)
    p_cmp.add_argument("--markdown", type=Path, default=None)
    p_base = sub.add_parser("baseline")
    p_base.add_argument("--current", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.cmd == "run":
        report = run(args.fixtures)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report.model_dump_json(indent=2))
        print(report.model_dump_json(indent=2))
        return 0
    if args.cmd == "compare":
        return compare(args.current, args.markdown)
    path = _baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(GateReport.model_validate_json(args.current.read_text()).model_dump_json(indent=2) + "\n")
    print(f"baseline written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
