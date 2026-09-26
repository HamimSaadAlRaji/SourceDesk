"""Phase 11 Unit 3 smoke test: one real generated answer, scored by both
real judges (Gemini `gemini-3.5-flash`, `openai/gpt-oss-120b` on Groq)
before trusting this wiring for Unit 4's N=30 batch. Also the first real
call for `build_gemini_judge_llm` (written in Unit 1, never smoke-tested --
`docs/results/ragas-smoke-test.md`'s open item).

Real dense retrieval, real Groq generation (`qwen/qwen3.8-27b`), same query
used in the RAGAS smoke test for direct comparability.
"""

from __future__ import annotations

import json
import random
import time

from aegis.config import settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.eval.agreement import cohen_kappa, raw_agreement_rate
from aegis.eval.judge import build_gemini_judge, build_groq_judge
from aegis.generation.budgeter import format_context
from aegis.generation.llm.fallback import FallbackLLMClient
from aegis.generation.llm.groq_provider import GroqProvider
from aegis.generation.pipeline import GenerationPipeline
from aegis.guardrails.citations import CitationValidator
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, GoldenQuery, Query, Retrieved


def load_chunks() -> list[Chunk]:
    path = settings.paths.data_processed / "chunks.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def load_sample_query() -> GoldenQuery:
    path = settings.paths.data_golden / "golden_set_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        queries = [GoldenQuery(**json.loads(line)) for line in f]
    positives = [q for q in queries if not q.is_negative and q.ground_truth_chunk_ids]
    return random.Random(settings.golden.random_seed).sample(positives, 1)[0]


def main() -> None:
    chunks = load_chunks()
    q = load_sample_query()
    print(f"query: {q.id!r} — {q.text[:100]!r}")

    cache = EmbeddingCache(settings.paths.embedding_cache)
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=cache, device="cpu")
    dense_index = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)
    results = dense_index.search(q.text, top_k=settings.generation.retrieval_top_k, exclude_doc_id=q.source_issue)
    retrieved = [Retrieved(chunk=c, score=s, rank=i + 1) for i, (c, s) in enumerate(results)]

    pipeline = GenerationPipeline(llm=FallbackLLMClient([GroqProvider()]), validator=CitationValidator())
    result = pipeline.answer(Query(id=q.id, text=q.text), retrieved)
    print(f"generation: provider={result.provider} valid_citation_rate={result.validation.valid_citation_rate}")
    print(f"answer (first 300 chars): {result.text[:300]!r}\n")

    context_block = format_context(result.context)

    judges = [("groq (openai/gpt-oss-120b)", build_groq_judge()), ("gemini (gemini-3.5-flash)", build_gemini_judge())]
    scores: dict[str, int] = {}
    for label, judge in judges:
        start = time.perf_counter()
        try:
            judgment = judge.judge(query=q.text, answer=result.text, context_block=context_block)
        except Exception as exc:  # noqa: BLE001 -- log and continue, don't lose the other judge's result
            print(f"[{label}] FAILED: {exc!r}")
            continue
        latency_ms = (time.perf_counter() - start) * 1000
        scores[judge.name] = judgment.score
        print(f"[{label}] score={judgment.score} latency_ms={latency_ms:.0f}")
        print(f"  reasoning: {judgment.reasoning!r}\n")

    if len(scores) == 2:
        print(f"scores: {scores}")
        # N=1: kappa/raw-agreement are not meaningful statistics at this
        # sample size (a single-item comparison is trivially either 1.0 or
        # undefined) -- printed only as a sanity check that the functions
        # run against real judge output, not as a real agreement measurement.
        # The real measurement is Unit 4's job, at N=30.
        labels_a, labels_b = [scores["groq"]], [scores["gemini"]]
        print(f"[N=1 sanity check only] raw_agreement_rate={raw_agreement_rate(labels_a, labels_b)}")
        try:
            print(f"[N=1 sanity check only] cohen_kappa={cohen_kappa(labels_a, labels_b)}")
        except Exception as exc:  # noqa: BLE001 -- kappa is undefined for some degenerate N=1 cases
            print(f"[N=1 sanity check only] cohen_kappa undefined at N=1: {exc!r}")


if __name__ == "__main__":
    main()
