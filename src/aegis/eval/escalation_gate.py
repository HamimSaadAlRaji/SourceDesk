"""The escalation half of the CI gate (design record 16i): replay Phase 10's escalate-or-answer
decision over the 180 dev queries and compare to a committed baseline.

    python -m aegis.eval.escalation_gate run --fixtures ci-fixtures --out results/escalation.json
    python -m aegis.eval.escalation_gate compare --current results/escalation.json [--markdown ...]
    python -m aegis.eval.escalation_gate baseline --current results/escalation.json
    python -m aegis.eval.escalation_gate run ... --save-cache   # extend ci/caches/rerank_scores.json

North-star check: the failure this project exists to prevent is a confident, uncited, unescalated
wrong answer. `air_canada_fn` counts negatives (queries that should escalate) the scorer answered
anyway, and has zero tolerance. `deflection_rate` is the other side of the dial: positives answered.

Cost: the cross-encoder dominates (1.32 s per pair on an M1 CPU, 538 distinct pairs). Scores come
from a content-addressed cache (aegis.eval.rerank_cache); only pairs not in it are computed, up to a
miss budget, so a change to weights, bounds or tau re-evaluates for free while a change that alters
what the reranker sees is computed or fails loudly. Query and chunk embeddings come from the tracked
embedding cache; retrieval itself is the retrieval gate's job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.embedding.vector_store import get_client
from aegis.eval.ci_fixtures import verify_fixtures
from aegis.eval.custom_metrics import escalation_appropriateness
from aegis.eval.gate import GateReport, MetricSpec, compare_files, write_baseline
from aegis.eval.rerank_cache import ScoreCache, make_score_fn
from aegis.factory import load_chunks
from aegis.guardrails.confidence import ConfidenceScorer, decide_escalation
from aegis.rerank.cross_encoder import CrossEncoderReranker, _default_score_fn
from aegis.retrieval.dense import DenseIndex
from aegis.types import GoldenQuery

BASELINE_NAME = "escalation.json"
CACHE_PATH = REPO_ROOT / "ci" / "caches" / "rerank_scores.json"


def score_escalation(escalated: list[bool], is_negative: list[bool]) -> dict[str, float]:
    c = escalation_appropriateness(escalated, is_negative)
    return {"air_canada_fn": float(c.fn), "negative_recall": c.recall, "deflection_rate": c.deflection_rate}


def gate_specs() -> list[MetricSpec]:
    g = settings.gate
    return [
        MetricSpec(name="air_canada_fn", tolerance=g.escalation_fn_tolerance, direction="lower"),
        MetricSpec(name="deflection_rate", tolerance=g.escalation_deflection_tolerance),
    ]


def load_dev_queries() -> list[GoldenQuery]:
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        return [q for q in (GoldenQuery(**json.loads(line)) for line in f) if q.split == "dev"]


def run(fixtures_dir: Path, save_cache: bool = False, max_uncached: int | None = None) -> GateReport:
    fingerprint = verify_fixtures(fixtures_dir).fingerprint
    chunks, queries = load_chunks(), load_dev_queries()
    encoder = Encoder(model_name=settings.embedding.bge_large_model,
                      cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
    index = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder, client=get_client())

    cache = ScoreCache.load(CACHE_PATH)
    limit = settings.gate.escalation_max_uncached_pairs if max_uncached is None else max_uncached
    inner = _default_score_fn(settings.rerank.model_name, "cpu", settings.rerank.batch_size)  # loads the model only on a miss
    reranker = CrossEncoderReranker(score_fn=make_score_fn(settings.rerank.model_name, inner, cache, limit))
    scorer = ConfidenceScorer(reranker=reranker, encode_fn=encoder.encode)

    escalated: list[bool] = []
    for q in queries:
        candidates = index.search(q.text, top_k=settings.generation.retrieval_top_k, exclude_doc_id=q.source_issue)
        escalated.append(decide_escalation(scorer.score(q.text, candidates).confidence).escalate)
    if save_cache and cache.misses:
        cache.meta["n_pairs"] = str(len(cache.scores))
        cache.save(CACHE_PATH)

    metrics = score_escalation(escalated, [q.is_negative for q in queries])
    info = {"tau": str(settings.confidence.tau), "rerank_cache_hits": str(cache.hits),
            "rerank_cache_misses": str(cache.misses), "rerank_cache_size": str(len(cache.scores))}
    return GateReport(suite="escalation", fingerprint=fingerprint, n_queries=len(queries),
                      metrics={k: v for k, v in metrics.items() if v == v}, info=info)  # NaN omitted: a gated metric then fails


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--fixtures", type=Path, required=True)
    p_run.add_argument("--out", type=Path, required=True)
    p_run.add_argument("--save-cache", action="store_true", help="persist newly scored pairs to ci/caches/rerank_scores.json")
    p_run.add_argument("--max-uncached", type=int, default=None)
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--current", type=Path, required=True)
    p_cmp.add_argument("--markdown", type=Path, default=None)
    p_base = sub.add_parser("baseline")
    p_base.add_argument("--current", type=Path, required=True)
    args = parser.parse_args(argv)

    baseline = settings.gate.baseline_dir / BASELINE_NAME
    if args.cmd == "run":
        report = run(args.fixtures, args.save_cache, args.max_uncached)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report.model_dump_json(indent=2))
        print(report.model_dump_json(indent=2))
        return 0
    if args.cmd == "compare":
        return compare_files(args.current, baseline, gate_specs(),
                             f"Escalation (tau {settings.confidence.tau}, 180 dev queries)", args.markdown)
    write_baseline(args.current, baseline)
    print(f"baseline written to {baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
