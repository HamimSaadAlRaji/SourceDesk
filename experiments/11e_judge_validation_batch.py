"""Phase 11 Unit 4: the real N=30 batch run -- generation -> RAGAS -> both
judges -> inter-judge Cohen's kappa. This is where DoD tracker item #8
(kappa >= 0.6) and the RAGAS faithfulness >= 0.90 exit criterion
(plan.md) get real numbers, not the N=1 sanity checks from Units 1 and 3's
smoke tests.

**Field-level resumable, on purpose.** Real quota risk identified before
this ran (design record 11c §7): Gemini's free tier is 20 requests/day/
model (`docs/results/llm-bakeoff.md`), and this batch alone asks for 30
Gemini judge calls plus whatever fraction of 30 generations fail over to
Gemini -- plausibly over the daily ceiling in one sitting. Re-running the
whole batch from zero after a quota wall would re-spend every dollar/quota
unit already spent on the parts that *did* succeed (generation, RAGAS,
the Groq judge) just to retry the one piece that didn't. Each query's row
is checkpointed to `RESULTS_RAW` after every sub-step; a re-run loads the
existing checkpoint and only computes whatever's still missing or errored
for that query, same principle as the resumable hand-labeling CLI this
project would have built if the user hadn't declined to hand-label
(progress.md Decision Log, 2026-09-25) -- the resumability need didn't go
away when the hand-labeling did, it just moved to a different real
constraint (external quota instead of manual time).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, IDBasedContextPrecision, IDBasedContextRecall, ResponseRelevancy

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.eval.agreement import cohen_kappa, raw_agreement_rate
from aegis.eval.judge import build_gemini_judge, build_groq_judge
from aegis.eval.ragas_adapters import RagasLocalEmbeddings, build_groq_ragas_llm
from aegis.generation.budgeter import format_context
from aegis.generation.llm.fallback import FallbackLLMClient
from aegis.generation.llm.gemini_provider import GeminiProvider
from aegis.generation.llm.groq_provider import GroqProvider
from aegis.generation.pipeline import GenerationPipeline
from aegis.guardrails.citations import CitationValidator
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, GoldenQuery, Query, Retrieved

N_QUERIES = int(os.environ.get("AEGIS_BATCH_N", "30"))  # locked in with the user, Phase 11 scoping session (progress.md)
RESULTS_RAW = REPO_ROOT / "docs" / "results" / os.environ.get("AEGIS_BATCH_OUT", "judge-agreement-batch-raw.json")


def load_chunks() -> list[Chunk]:
    path = settings.paths.data_processed / "chunks.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def load_sample_queries(n: int) -> list[GoldenQuery]:
    path = settings.paths.data_golden / "golden_set_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        queries = [GoldenQuery(**json.loads(line)) for line in f]
    positives = [q for q in queries if not q.is_negative and q.ground_truth_chunk_ids]
    return random.Random(settings.golden.random_seed).sample(positives, n)


def load_checkpoint() -> dict[str, dict]:
    if not RESULTS_RAW.exists():
        return {}
    with open(RESULTS_RAW, encoding="utf-8") as f:
        rows = json.load(f)
    return {row["query_id"]: row for row in rows}


def save_checkpoint(rows_by_id: dict[str, dict]) -> None:
    RESULTS_RAW.parent.mkdir(parents=True, exist_ok=True)
    # Sorted for a stable diff across runs -- easier to eyeball what changed.
    RESULTS_RAW.write_text(json.dumps(list(rows_by_id.values()), indent=2, sort_keys=False))


async def main() -> None:
    chunks = load_chunks()
    queries = load_sample_queries(N_QUERIES)
    print(f"{len(chunks)} chunks, {len(queries)} sampled dev positives")

    cache = EmbeddingCache(settings.paths.embedding_cache)
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=cache, device="cpu")
    dense_index = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)
    validator = CitationValidator()
    pipeline = GenerationPipeline(llm=FallbackLLMClient([GroqProvider(), GeminiProvider()]), validator=validator)

    ragas_llm = build_groq_ragas_llm()
    ragas_embeddings = RagasLocalEmbeddings(encoder)
    faithfulness = Faithfulness(llm=ragas_llm)
    response_relevancy = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    id_precision = IDBasedContextPrecision()
    id_recall = IDBasedContextRecall()

    groq_judge = build_groq_judge()
    gemini_judge = build_gemini_judge()

    rows_by_id = load_checkpoint()

    for i, q in enumerate(queries, 1):
        row = rows_by_id.setdefault(q.id, {"query_id": q.id})
        print(f"\n[{i}/{len(queries)}] {q.id}")

        # --- Generation (also needed to build context_block for the rest) ---
        if "generation" not in row or "error" in row["generation"]:
            results = dense_index.search(q.text, top_k=settings.generation.retrieval_top_k, exclude_doc_id=q.source_issue)
            retrieved = [Retrieved(chunk=c, score=s, rank=j + 1) for j, (c, s) in enumerate(results)]
            try:
                result = pipeline.answer(Query(id=q.id, text=q.text), retrieved)
                row["generation"] = {
                    "provider": result.provider,
                    "valid_citation_rate": result.validation.valid_citation_rate,
                    "answer_text": result.text,
                    "context_block": format_context(result.context),
                    "retrieved_context_ids": [pc.chunk.id for pc in result.context.chunks],
                }
                print(f"  generation: provider={result.provider} valid_citation_rate={result.validation.valid_citation_rate}")
            except Exception as exc:  # noqa: BLE001
                row["generation"] = {"error": repr(exc)}
                print(f"  generation FAILED: {exc!r}")
        else:
            print("  generation: cached")

        save_checkpoint(rows_by_id)

        if "error" in row["generation"]:
            continue  # nothing downstream is possible without a real answer

        gen = row["generation"]
        sample = SingleTurnSample(
            user_input=q.text,
            response=gen["answer_text"],
            retrieved_contexts=[gen["context_block"]],
            retrieved_context_ids=gen["retrieved_context_ids"],
            reference_context_ids=q.ground_truth_chunk_ids,
        )

        # --- RAGAS ---
        row.setdefault("ragas", {})
        ragas_metrics = {
            "faithfulness": faithfulness,
            "response_relevancy": response_relevancy,
            "id_based_context_precision": id_precision,
            "id_based_context_recall": id_recall,
        }
        for name, metric in ragas_metrics.items():
            if name in row["ragas"] and "error" not in row["ragas"][name]:
                continue
            start = time.perf_counter()
            try:
                score = await metric.single_turn_ascore(sample)
                row["ragas"][name] = {"score": score, "latency_ms": (time.perf_counter() - start) * 1000}
                print(f"  ragas.{name}={score:.4f}")
            except Exception as exc:  # noqa: BLE001
                row["ragas"][name] = {"error": repr(exc)}
                print(f"  ragas.{name} FAILED: {exc!r}")
            save_checkpoint(rows_by_id)

        # --- Judges ---
        for judge in (groq_judge, gemini_judge):
            key = f"judge_{judge.name}"
            if key in row and "error" not in row[key]:
                continue
            start = time.perf_counter()
            try:
                judgment = judge.judge(query=q.text, answer=gen["answer_text"], context_block=gen["context_block"])
                row[key] = {
                    "score": judgment.score,
                    "reasoning": judgment.reasoning,
                    "latency_ms": (time.perf_counter() - start) * 1000,
                }
                print(f"  {key}={judgment.score}")
            except Exception as exc:  # noqa: BLE001
                row[key] = {"error": repr(exc)}
                print(f"  {key} FAILED: {exc!r}")
            save_checkpoint(rows_by_id)

    # --- Summary ---
    rows = list(rows_by_id.values())
    print(f"\n{'=' * 60}\nSummary over {len(rows)} queries\n{'=' * 60}")

    gen_ok = [r for r in rows if "error" not in r.get("generation", {"error": 1})]
    print(f"generation succeeded: {len(gen_ok)}/{len(rows)}")

    for metric_name in ["faithfulness", "response_relevancy", "id_based_context_precision", "id_based_context_recall"]:
        scores = [r["ragas"][metric_name]["score"] for r in gen_ok if "score" in r.get("ragas", {}).get(metric_name, {})]
        if scores:
            print(f"ragas.{metric_name}: n={len(scores)} mean={sum(scores) / len(scores):.4f}")
        else:
            print(f"ragas.{metric_name}: n=0")

    paired = [
        r
        for r in gen_ok
        if "score" in r.get("judge_groq", {}) and "score" in r.get("judge_gemini", {})
    ]
    print(f"\nboth judges succeeded: {len(paired)}/{len(rows)}")
    if paired:
        groq_scores = [r["judge_groq"]["score"] for r in paired]
        gemini_scores = [r["judge_gemini"]["score"] for r in paired]
        print(f"judge_groq mean={sum(groq_scores) / len(groq_scores):.2f}")
        print(f"judge_gemini mean={sum(gemini_scores) / len(gemini_scores):.2f}")
        print(f"raw_agreement_rate={raw_agreement_rate(groq_scores, gemini_scores):.4f}")
        print(f"cohen_kappa={cohen_kappa(groq_scores, gemini_scores):.4f}")
        print(f"cohen_kappa (quadratic)={cohen_kappa(groq_scores, gemini_scores, weights='quadratic'):.4f}")

        disagreements = [r for r in paired if abs(r["judge_groq"]["score"] - r["judge_gemini"]["score"]) >= 2]
        print(f"\ndisagreements (|diff| >= 2): {len(disagreements)}/{len(paired)}")
        for r in disagreements:
            print(f"  {r['query_id']}: groq={r['judge_groq']['score']} gemini={r['judge_gemini']['score']}")

    print(f"\nRaw results: {RESULTS_RAW}")


if __name__ == "__main__":
    asyncio.run(main())
