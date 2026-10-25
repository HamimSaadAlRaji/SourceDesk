"""Phase 17 (design record 17d): do the direct and LangGraph pipelines make the same decisions
on the 180 dev queries?

    python experiments/17b_equivalence.py            # record on direct (paid), replay on graph
    python experiments/17b_equivalence.py --control  # negative control: graph with tau shifted +0.1
    python experiments/17b_equivalence.py --offline  # no LLM at all (stubs); wiring check only

Everything except the orchestration is shared and cached: query vectors and chunk embeddings
from the embedding cache, reranker scores from ci/caches/rerank_scores.json (misses are
computed and counted), retrieval excludes each query's source issue exactly as the Phase 10
escalation gate does. The semantic cache is off (its hit pattern depends on arrival order and the
queries run on threads); the exact cache is on, and every query is asked twice, so the second pass
is a real-data check of the cache-hit branch.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aegis.cache.exact_match import ExactMatchCache
from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.embedding.vector_store import get_client
from aegis.eval.custom_metrics import escalation_appropriateness
from aegis.eval.equivalence import ReplayMissError, Tape, TapedLLM, compare_results, taped_fn
from aegis.eval.escalation_gate import CACHE_PATH, load_dev_queries
from aegis.eval.rerank_cache import ScoreCache, make_score_fn
from aegis.factory import load_chunks
from aegis.generation.llm import default_client
from aegis.generation.pipeline import GenerationPipeline
from aegis.graph.pipeline import LangGraphPipeline
from aegis.guardrails.citations import CitationValidator
from aegis.guardrails.confidence import ConfidenceScorer
from aegis.observability.tracing import get_tracer
from aegis.pipeline import AnswerPipeline, PipelineResult
from aegis.rerank.cross_encoder import CrossEncoderReranker, _default_score_fn
from aegis.retrieval.dense import DenseIndex
from aegis.router.classifier import RouterClassifier
from aegis.types import Query

OUT = REPO_ROOT / "docs" / "results" / "orchestration-equivalence-raw.json"
TAPE = REPO_ROOT / "data" / "cache" / "eval" / "equivalence_tape.json"
WORKERS = 8
MAX_UNCACHED_PAIRS = 60
CONTROL_TAU_SHIFT = 0.1


class ExcludingRetriever:
    """Excludes each query's own source issue (the golden set's leakage guard), keyed by query text."""

    def __init__(self, index: DenseIndex, source_by_text: dict[str, str | None]) -> None:
        self.index, self.source_by_text = index, source_by_text

    def search(self, query: str, top_k: int, **kwargs):
        return self.index.search(query, top_k=top_k, exclude_doc_id=self.source_by_text.get(query), **kwargs)


def _unavailable(*_a, **_k):
    raise AssertionError("the replaying path must never reach a live LLM")


class _OfflineLLM:
    """--offline wiring check: canned, deterministic, free."""

    def generate_with_provider(self, messages, temperature, top_p, max_tokens):
        return "Offline stub answer [1]", "offline"


def build(cls, tape: Tape, scorer: ConfidenceScorer, retriever, live: bool, offline: bool, **extra) -> AnswerPipeline:
    if offline and live:
        classify, llm, entail = (lambda q: '{"route": "docs_howto"}'), _OfflineLLM(), (lambda claim, chunk: True)
    elif live:
        classify = RouterClassifier()._classify_fn
        llm, entail = default_client(), CitationValidator()._entailment_fn
    else:  # replay: the inner functions must never run
        classify, llm, entail = _unavailable, None, _unavailable
    router = RouterClassifier(classify_fn=taped_fn(tape, "router", classify))
    generation = GenerationPipeline(
        llm=TapedLLM(llm, tape),
        validator=CitationValidator(entailment_fn=taped_fn(tape, "entailment", entail)),
    )
    return cls(
        exact_cache=ExactMatchCache(Path(tempfile.mkdtemp(prefix="equiv-")) / "exact.sqlite"),
        semantic_cache=None, router=router, retriever=retriever, confidence_scorer=scorer,
        generation=generation, tracer=get_tracer(), corpus_version="equivalence", **extra,
    )


def run_pass(pipeline: AnswerPipeline, queries) -> tuple[dict[str, PipelineResult | str], float]:
    def one(q):
        try:
            return q.id, pipeline.answer(Query(id=q.id, text=q.text))
        except ReplayMissError as exc:
            return q.id, f"replay_miss: {exc}"

    start = time.perf_counter()
    with ThreadPoolExecutor(WORKERS) as pool:
        out = dict(pool.map(one, queries))
    return out, time.perf_counter() - start


def summarize(results: dict[str, PipelineResult | str], queries) -> dict:
    ok = {q.id: results[q.id] for q in queries if isinstance(results[q.id], PipelineResult)}
    escalated = [ok[q.id].escalate for q in queries if q.id in ok]
    negatives = [q.is_negative for q in queries if q.id in ok]
    c = escalation_appropriateness(escalated, negatives)
    answered = [r for r in ok.values() if not r.escalate]
    return {
        "n": len(ok), "escalated": sum(escalated), "answered": len(answered),
        "air_canada_fn": c.fn, "negative_recall": c.recall, "deflection_rate": c.deflection_rate,
        "cited_answer_rate": (sum(1 for r in answered if r.citations) / len(answered)) if answered else None,
        "causes": dict(Counter((r.escalation_reason or "")[:24] for r in ok.values() if r.escalate)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    queries = load_dev_queries()
    assert len({q.text for q in queries}) == len(queries), "retriever keys on query text"
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
    index = DenseIndex(load_chunks(), settings.qdrant.collection_bge_large, encoder, client=get_client())
    retriever = ExcludingRetriever(index, {q.text: q.source_issue for q in queries})

    cache = ScoreCache.load(CACHE_PATH)
    inner = _default_score_fn(settings.rerank.model_name, "cpu", settings.rerank.batch_size)
    lock = threading.Lock()  # torch inference is not safe to run from 8 threads at once
    cached = make_score_fn(settings.rerank.model_name, inner, cache, MAX_UNCACHED_PAIRS)
    scorer = ConfidenceScorer(
        reranker=CrossEncoderReranker(score_fn=lambda pairs: _locked(lock, cached, pairs)), encode_fn=encoder.encode
    )

    from aegis.generation.llm.openai_provider import default_ledger

    spent_before = default_ledger().summary()["spent_usd"]
    report: dict = {"offline": args.offline, "control": args.control, "n_queries": len(queries)}

    runs: dict[str, dict] = {}
    if args.control:  # both paths replay the tape the real run recorded
        tape = Tape.load(TAPE, replay_only=True)
        direct = build(AnswerPipeline, Tape(tape.entries, replay_only=True), scorer, retriever, live=False, offline=False)
        d1, dt1 = run_pass(direct, queries)
        runs["direct"] = {"pass1": d1, "pass2": {}, "wall_s": [dt1, 0.0]}
        replay = Tape(tape.entries, replay_only=True)
    else:
        tape = Tape()
        live_direct = build(AnswerPipeline, tape, scorer, retriever, live=True, offline=args.offline)
        p1, t1 = run_pass(live_direct, queries)
        p2, t2 = run_pass(live_direct, queries)
        runs["direct"] = {"pass1": p1, "pass2": p2, "wall_s": [t1, t2]}
        if not args.offline:
            tape.save(TAPE)
        replay = Tape(tape.entries, replay_only=True)

    extra = {"tau": settings.confidence.tau + CONTROL_TAU_SHIFT} if args.control else {}
    graph = build(LangGraphPipeline, replay, scorer, retriever, live=False, offline=args.offline, **extra)
    g1, gt1 = run_pass(graph, queries)
    g2, gt2 = run_pass(graph, queries) if not args.control else ({}, 0.0)
    runs["langgraph"] = {"pass1": g1, "pass2": g2, "wall_s": [gt1, gt2]}

    report["replay_unused_entries"] = replay.unused
    report["rerank_cache"] = {"hits": cache.hits, "misses": cache.misses}
    report["spent_usd"] = round(default_ledger().summary()["spent_usd"] - spent_before, 6)
    if "direct" in runs:
        for label in ("pass1", "pass2"):
            if not runs["langgraph"][label]:
                continue
            diffs = {}
            for q in queries:
                a, b = runs["direct"][label][q.id], runs["langgraph"][label][q.id]
                if isinstance(a, str) or isinstance(b, str):
                    diffs[q.id] = [b if isinstance(b, str) else a]
                elif d := compare_results(a, b):
                    diffs[q.id] = d
            report[label] = {
                "identical": len(queries) - len(diffs), "differing": len(diffs),
                "differing_fields": dict(Counter(f for d in diffs.values() for f in d)),
                "differing_ids": sorted(diffs)[:20],
                "direct": summarize(runs["direct"][label], queries), "langgraph": summarize(runs["langgraph"][label], queries),
                "cache_status": {n: dict(Counter(r.cache_status.value for r in runs[n][label].values()
                                                 if isinstance(r, PipelineResult))) for n in runs},
            }
    report["wall_s"] = {n: runs[n]["wall_s"] for n in runs}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    target = OUT.with_name(OUT.stem + ("-control" if args.control else "-offline" if args.offline else "") + ".json")
    target.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


def _locked(lock: threading.Lock, fn, pairs):
    with lock:
        return fn(pairs)


if __name__ == "__main__":
    main()
