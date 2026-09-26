"""Prompt v1 vs v2 check (follow-up to plan.md Phase 9, step 3): v1's real
measured gap was 21.1% `missing_citation` (N=20, Groq, see design record 09
§5), spot-checked to a specific pattern -- opening/topic sentences
disproportionately lacking a citation. v2 adds an explicit first-sentence
rule and a worked example targeting that exact pattern (`generation/
prompts/v2.py`).

First attempt ran against Gemini (Groq's quota was exhausted at the time)
and got 2/10 real v1 completions, 0/10 v2, before Gemini's *own* limit (20
requests/day/model, discovered by that run) shut it down too -- not enough
to validate anything, and not the production-primary model anyway. This
run targets Groq directly, at reduced `max_tokens` (400, down from the
default 1024) specifically to conserve Groq's token-shaped daily quota
across 2 versions x N_QUERIES real calls, and writes incrementally after
each prompt version so a second wall doesn't cost the first version's data.
"""

from __future__ import annotations

import json
import random

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.generation.llm.fallback import FallbackLLMClient
from aegis.generation.llm.groq_provider import GroqProvider
from aegis.generation.pipeline import GenerationPipeline
from aegis.generation.prompts import get_prompt
from aegis.guardrails.citations import CitationValidator
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, GoldenQuery, Query, Retrieved

N_QUERIES = 8  # conservative -- Groq's token-shaped quota, not request-shaped, but still finite
MAX_TOKENS = 400
RESULTS_RAW = REPO_ROOT / "docs" / "results" / "prompt-v1-vs-v2-groq-raw.json"


def load_chunks() -> list[Chunk]:
    path = settings.paths.data_processed / "chunks.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def load_sample_queries(n: int) -> list[GoldenQuery]:
    path = settings.paths.data_golden / "golden_set_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        queries = [GoldenQuery(**json.loads(line)) for line in f]
    positives = [q for q in queries if not q.is_negative]
    # Different offset from the sampling sweep/bake-off's own samples
    # (same seed, different n) so this isn't literally re-scoring the same
    # 15/20 queries those already covered.
    return random.Random(settings.golden.random_seed).sample(positives, n)


def main() -> None:
    chunks = load_chunks()
    queries = load_sample_queries(N_QUERIES)
    print(f"{len(chunks)} chunks, {len(queries)} sampled dev positives")

    cache = EmbeddingCache(settings.paths.embedding_cache)
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=cache, device="cpu")
    dense_index = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)

    retrieved_by_query: dict[str, list[Retrieved]] = {}
    for q in queries:
        results = dense_index.search(q.text, top_k=settings.generation.retrieval_top_k, exclude_doc_id=q.source_issue)
        retrieved_by_query[q.id] = [Retrieved(chunk=c, score=s, rank=i + 1) for i, (c, s) in enumerate(results)]

    validator = CitationValidator()
    llm = FallbackLLMClient([GroqProvider()])
    RESULTS_RAW.parent.mkdir(parents=True, exist_ok=True)

    raw: list[dict] = []
    for version in ["v1", "v2"]:
        pipeline = GenerationPipeline(
            llm=llm, validator=validator, prompt=get_prompt(version), temperature=0.0, max_tokens=MAX_TOKENS
        )
        rates, missing_counts, total_claims = [], 0, 0
        for q in queries:
            try:
                result = pipeline.answer(Query(id=q.id, text=q.text), retrieved_by_query[q.id])
            except Exception as exc:  # noqa: BLE001
                print(f"  [{version}] {q.id} failed: {exc!r}")
                continue
            rates.append(result.validation.valid_citation_rate)
            for claim in result.validation.claims:
                if claim.status.value != "exempt":
                    total_claims += 1
                if claim.status.value == "missing_citation":
                    missing_counts += 1
            raw.append(
                {
                    "prompt_version": version,
                    "query_id": q.id,
                    "valid_citation_rate": result.validation.valid_citation_rate,
                    "answer_text": result.text,
                }
            )
            # Incremental write after every query -- a later failure (e.g. a
            # quota wall mid-version) shouldn't cost data already collected.
            RESULTS_RAW.write_text(json.dumps(raw, indent=2))

        mean_rate = sum(rates) / len(rates) if rates else float("nan")
        missing_frac = missing_counts / total_claims if total_claims else float("nan")
        print(
            f"[{version}] N={len(rates)}  mean_valid_citation_rate={mean_rate:.4f}  "
            f"missing_citation_frac={missing_frac:.4f}  ({missing_counts}/{total_claims} claims)"
        )

    print(f"\nRaw results written to {RESULTS_RAW}")


if __name__ == "__main__":
    main()
