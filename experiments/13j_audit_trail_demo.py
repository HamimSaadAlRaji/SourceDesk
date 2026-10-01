"""Phase 13 Unit 5 -- the audit-trail demo (plan.md build item 5).

Runs real queries through the real `AnswerPipeline` (real Qdrant, real
cross-encoder, real Groq router + generation, real Langfuse tracing), then for
each one rebuilds "what evidence produced this answer" FROM THE STORED TRACE
ONLY, and checks that reconstruction against what the live pipeline actually
returned -- independent of the trace's own root-span copy. Also joins the
structured log lines by trace ID.

Output: docs/results/audit-trail-example.md. Fresh throwaway caches per run, so
every query takes the real path (a cache hit would be a legitimate but boring
1-span trace).
"""

from __future__ import annotations

import json
import logging
import sys
import time

from aegis.cache.exact_match import ExactMatchCache
from aegis.cache.semantic import SemanticCache
from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.generation.llm.fallback import FallbackLLMClient
from aegis.generation.llm.gemini_provider import GeminiProvider
from aegis.generation.llm.groq_provider import GroqProvider
from aegis.generation.pipeline import GenerationPipeline
from aegis.guardrails.citations import CitationValidator
from aegis.guardrails.confidence import ConfidenceScorer
from aegis.observability.audit import AuditTrail, build_audit_trail, fetch_observations, render_markdown, resolve_evidence
from aegis.observability.logging import JsonFormatter, configure_logging
from aegis.observability.tracing import get_tracer
from aegis.pipeline import AnswerPipeline, PipelineResult
from aegis.rerank.cross_encoder import CrossEncoderReranker
from aegis.retrieval.dense import DenseIndex
from aegis.router.classifier import RouterClassifier
from aegis.types import Chunk, GoldenQuery, Query

QUERY_IDS = [
    ("answered", "golden::issue::604"),
    ("hard-route escalation", "negative::billing_legal::4"),
    ("confidence escalation", "golden::issue::2689"),
]
OUT = REPO_ROOT / "docs" / "results" / "audit-trail-example.md"
LOG_PATH = REPO_ROOT / "docs" / "results" / "audit-trail-example.log.jsonl"


def differences_from_live_result(trail: AuditTrail, result: PipelineResult) -> list[str]:
    """Trace-reconstructed trail vs the object the pipeline actually returned."""
    diffs: list[str] = []

    def cmp(field: str, a, b) -> None:
        if a != b:
            diffs.append(f"{field}: trace says {a!r}, live result says {b!r}")

    cmp("cache_status", trail.cache_status, result.cache_status.value)
    cmp("route", trail.route, result.route.value if result.route else None)
    cmp("escalated", trail.escalated, result.escalate)
    cmp("final_text", trail.final_text, result.text)
    cmp("citations", trail.citations, result.citations)
    cmp("provider", trail.provider, result.provider)
    cmp("prompt_version", trail.prompt_version, result.prompt_version)
    cmp("retrieved ids", [r.chunk_id for r in trail.retrieved], [r.chunk.id for r in result.retrieved])
    cmp("retrieved scores", [round(r.score, 4) for r in trail.retrieved], [round(r.score, 4) for r in result.retrieved])
    cmp("context (citation_id, chunk_id, hash)", [(c["citation_id"], c["chunk_id"], c.get("content_hash")) for c in trail.context],
        [(e.citation_id, e.chunk_id, e.content_hash) for e in result.context])
    if result.confidence is not None:
        cmp("confidence", round((trail.confidence or {}).get("confidence", -1), 6), round(result.confidence.confidence, 6))
    return diffs


def main() -> int:
    configure_logging()
    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    logging.getLogger("aegis").addHandler(file_handler)

    with open(settings.paths.data_processed / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f]
    chunk_by_id = {c.id: c for c in chunks}
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        golden = {q.id: q for q in (GoldenQuery(**json.loads(l)) for l in f)}

    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
    for name in ("phase13j_exact.sqlite", "phase13j_semantic.sqlite"):
        (REPO_ROOT / "data" / "cache" / name).unlink(missing_ok=True)
    pipeline = AnswerPipeline(
        exact_cache=ExactMatchCache(REPO_ROOT / "data" / "cache" / "phase13j_exact.sqlite"),
        semantic_cache=SemanticCache(REPO_ROOT / "data" / "cache" / "phase13j_semantic.sqlite", encoder=encoder, threshold=settings.cache.semantic_similarity_threshold),
        router=RouterClassifier(),
        retriever=DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder),
        confidence_scorer=ConfidenceScorer(reranker=CrossEncoderReranker(), encode_fn=encoder.encode),
        generation=GenerationPipeline(llm=FallbackLLMClient([GroqProvider(), GeminiProvider()]), validator=CitationValidator()),
        tracer=get_tracer(),
        corpus_version="phase13j",
    )
    print(f"warm_up: {pipeline.warm_up()}", flush=True)

    runs: list[tuple[str, PipelineResult]] = []
    for label, qid in QUERY_IDS:
        q = golden[qid]
        result = pipeline.answer(Query(id=q.id, text=q.text))
        print(f"{label}: {qid} -> escalate={result.escalate} route={result.route} trace={result.trace_id}", flush=True)
        runs.append((label, result))
    get_tracer().flush()

    doc = ["# Worked audit-trail reconstruction", "",
           "Each section below was rebuilt **from the stored Langfuse trace alone** (`fetch_observations` -> `build_audit_trail` ->",
           "`resolve_evidence`), then checked field-by-field against what the live pipeline actually returned, and joined to the",
           "trace-ID-tagged log lines. Source: `experiments/13j_audit_trail_demo.py`. Real queries, real Groq, real Qdrant; nothing edited by hand.", ""]
    all_ok = True
    for label, result in runs:
        print(f"\nreconstructing {label} ...", flush=True)
        obs = fetch_observations(get_tracer(), result.trace_id)
        trail = build_audit_trail(obs, trace_id=result.trace_id)
        evidence = resolve_evidence(trail, chunk_by_id.get)
        diffs = differences_from_live_result(trail, result)
        all_ok &= not diffs and not trail.warnings and not trail.inconsistencies
        log_lines = [l for l in LOG_PATH.read_text().splitlines() if l.strip() and json.loads(l).get("trace_id") == result.trace_id]

        doc += ["---", "", f"## Case: {label}", "", render_markdown(trail, evidence), "",
                "### Reconstruction vs. the live pipeline result", ""]
        doc += ["- **Identical** on every audited field (cache status, route, outcome, answer text, citations, provider, prompt version, "
                "retrieved ids+scores, context map incl. content hashes, confidence)."] if not diffs else [f"- ✗ {d}" for d in diffs]
        doc += ["", f"### Log lines for this trace ({len(log_lines)})", "", "```json"] + log_lines + ["```", ""]
        print(f"  warnings={trail.warnings} inconsistencies={trail.inconsistencies} diffs_vs_live={diffs}", flush=True)

    OUT.write_text("\n".join(doc))
    print(f"\nwrote {OUT}; fully verified: {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
