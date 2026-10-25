"""The generation half of the CI gate (design record 16j): generate answers for a fixed
sample of dev queries with the served generator and validator, and compare citation
quality, code validity and latency to a committed baseline.

    python -m aegis.eval.generation_gate run     --out results/generation.json
    python -m aegis.eval.generation_gate compare --current results/generation.json [--markdown ...]
    python -m aegis.eval.generation_gate baseline --current results/generation.json

LLM-backed and paid (generation model + tier-2 entailment), capped per run by the spend
ledger. Needs the restored index, but not a live query encode: retrieval is not under test
here (the retrieval gate owns it), so query vectors come from the tracked embedding cache.

Two rules carried over from the Phase 15 bake-off (experiments/15g1):
  * Entailment does NOT fail closed while measuring. The served validator turns an API
    error into "unsupported" (right for serving); here that would score an outage as bad
    citations, so a persistent failure raises and fails the job.
  * The generator is OpenAI-only. The served fallback chain would silently measure Groq or
    Gemini if OpenAI failed, i.e. a different model than the one the baseline describes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from aegis.config import settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.embedding.vector_store import get_client
from aegis.eval.custom_metrics import code_block_syntax_validity
from aegis.eval.gate import GateReport, MetricSpec, compare_files, write_baseline
from aegis.factory import load_chunks
from aegis.generation.llm.chat import LLMCallError, chat_completion
from aegis.generation.llm.fallback import FallbackLLMClient
from aegis.generation.llm.openai_provider import OpenAIProvider
from aegis.generation.pipeline import GenerationPipeline
from aegis.guardrails.citations import ENTAILMENT_SYSTEM_PROMPT, CitationValidator
from aegis.retrieval.dense import DenseIndex
from aegis.types import GoldenQuery, Query, Retrieved

N_QUERIES = 20
BASELINE_NAME = "generation.json"
_WORKERS = 4
_MAX_RETRIES = 3


class EntailmentFailed(RuntimeError):
    """The entailment judge's API call failed persistently; not a verdict."""


class AnswerRow(BaseModel):
    valid_citation_rate: float
    n_supported_citations: int
    code_checked: int
    code_valid: int
    empty: bool
    generation_ms: float


def _nan() -> float:
    return float("nan")


def score_generation(rows: list[AnswerRow]) -> dict[str, float]:
    """An empty answer has nothing to validate, so its valid_citation_rate would be the
    vacuous 1.0; it scores 0.0 here instead, and empty / uncited answers are tracked as
    their own rates so a model that stops answering or citing cannot look better."""
    if not rows:
        return dict.fromkeys(
            ("valid_citation_rate", "cited_answer_rate", "empty_answer_rate", "code_syntax_valid_rate",
             "generation_p95_ms"), _nan())
    n = len(rows)
    checked = sum(r.code_checked for r in rows)
    return {
        "valid_citation_rate": sum(0.0 if r.empty else r.valid_citation_rate for r in rows) / n,
        "cited_answer_rate": sum(1 for r in rows if r.n_supported_citations > 0) / n,
        "empty_answer_rate": sum(1 for r in rows if r.empty) / n,
        "code_syntax_valid_rate": sum(r.code_valid for r in rows) / checked if checked else _nan(),
        "generation_p95_ms": float(np.percentile([r.generation_ms for r in rows], 95)),
    }


def load_queries() -> list[GoldenQuery]:
    """The same seeded 20 dev positives as the Phase 9 sweep and the 15g bake-off."""
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        queries = [GoldenQuery(**json.loads(line)) for line in f]
    positives = [q for q in queries if not q.is_negative]
    return random.Random(settings.golden.random_seed).sample(positives, N_QUERIES)


def make_strict_entailment_fn(stats: dict[str, int], lock: threading.Lock) -> Callable[[str, str], bool]:
    model = settings.openai.entailment_model

    def entails(claim: str, chunk_text: str) -> bool:
        messages = [{"role": "system", "content": ENTAILMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{chunk_text}\n\nClaim:\n{claim}"}]
        for attempt in range(_MAX_RETRIES):
            try:
                raw = chat_completion("openai", model, messages, temperature=0.0, max_tokens=10,
                                      purpose="entailment", json_mode=True)
                break
            except LLMCallError as exc:
                if exc.status_code == 429 and attempt < _MAX_RETRIES - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise EntailmentFailed(f"entailment call failed: {exc.__cause__!r}") from exc
        try:
            verdict = bool(json.loads(raw)["supported"])
        except (json.JSONDecodeError, KeyError, TypeError):
            with lock:
                stats["unparseable"] += 1
            verdict = False  # same as served
        with lock:
            stats["calls"] += 1
        return verdict

    return entails


def gate_specs() -> list[MetricSpec]:
    g = settings.gate
    return [
        MetricSpec(name="valid_citation_rate", tolerance=g.generation_citation_tolerance),
        MetricSpec(name="cited_answer_rate", tolerance=g.generation_cited_answer_tolerance),
        MetricSpec(name="empty_answer_rate", tolerance=g.generation_empty_answer_tolerance, direction="lower"),
        MetricSpec(name="generation_p95_ms", tolerance=g.generation_p95_tolerance_ms, direction="lower"),
    ]


def run(model: str | None = None) -> GateReport:
    """`model` overrides the generator for one-off comparisons only; the gate runs the served one."""
    queries = load_queries()
    chunks = load_chunks()
    encoder = Encoder(model_name=settings.embedding.bge_large_model,
                      cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
    index = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder, client=get_client())
    retrieved: dict[str, list[Retrieved]] = {}
    for q in queries:
        hits = index.search(q.text, top_k=settings.generation.retrieval_top_k, exclude_doc_id=q.source_issue)
        retrieved[q.id] = [Retrieved(chunk=c, score=s, rank=i + 1) for i, (c, s) in enumerate(hits)]

    stats = {"calls": 0, "unparseable": 0}
    lock = threading.Lock()
    generator = model or settings.openai.generation_model
    pipeline = GenerationPipeline(
        llm=FallbackLLMClient([OpenAIProvider(model=generator, purpose="ci_generation")]),
        validator=CitationValidator(entailment_fn=make_strict_entailment_fn(stats, lock)),
    )

    def one(q: GoldenQuery) -> AnswerRow:
        start = time.perf_counter()
        draft = pipeline.generate(Query(id=q.id, text=q.text), retrieved[q.id])
        generation_ms = (time.perf_counter() - start) * 1000
        result = pipeline.validate(draft)
        code = code_block_syntax_validity(draft.text)
        return AnswerRow(
            valid_citation_rate=result.validation.valid_citation_rate, n_supported_citations=len(result.citations),
            code_checked=code.n_checked, code_valid=code.n_valid, empty=not draft.text.strip(),
            generation_ms=generation_ms,
        )

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        rows = list(pool.map(one, queries))
    fingerprint_src = json.dumps(sorted(q.id for q in queries))
    info = {
        "generator": generator, "prompt": pipeline.prompt.version, "entailment_model": settings.openai.entailment_model,
        "entailment_calls": str(stats["calls"]), "entailment_unparseable": str(stats["unparseable"]),
    }
    info["code_blocks_checked"] = str(sum(r.code_checked for r in rows))
    # NaN means "no support" (e.g. no code blocks in this sample). It is not a number a report
    # can carry (JSON null), and a gated metric that is absent fails the gate, as it should.
    metrics = {k: v for k, v in score_generation(rows).items() if not math.isnan(v)}
    return GateReport(
        suite="generation", fingerprint=hashlib.sha256(fingerprint_src.encode()).hexdigest()[:16],
        n_queries=len(queries), metrics=metrics, info=info,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--out", type=Path, required=True)
    p_run.add_argument("--model", default=None, help="one-off comparison only; the gate runs the served generator")
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--current", type=Path, required=True)
    p_cmp.add_argument("--markdown", type=Path, default=None)
    p_base = sub.add_parser("baseline")
    p_base.add_argument("--current", type=Path, required=True)
    args = parser.parse_args(argv)

    baseline = settings.gate.baseline_dir / BASELINE_NAME
    if args.cmd == "run":
        report = run(args.model)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report.model_dump_json(indent=2))
        print(report.model_dump_json(indent=2))
        return 0
    if args.cmd == "compare":
        title = f"Generation ({settings.openai.generation_model}, {N_QUERIES} dev queries)"
        return compare_files(args.current, baseline, gate_specs(), title, args.markdown)
    write_baseline(args.current, baseline)
    print(f"baseline written to {baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
