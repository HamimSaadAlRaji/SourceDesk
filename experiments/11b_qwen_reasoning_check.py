"""Companion to `11a_gpt_oss_reasoning_tax.py`. Last session's investigation
(progress.md, 2026-08-18) tested `qwen/qwen3.6-27b` with a trivial
`max_tokens=10` call and found it reasoning-first (burned its budget on
hidden tokens, no visible output). Re-checking Groq's live catalog this
session (2026-09-25) found that model renamed to `qwen/qwen3.8-27b`, and a
fresh trivial call came back clean (`'OK'`, `reasoning_tokens=None`) --
a reversal worth checking at real answer length before trusting it, same
discipline as the gpt-oss-120b check (CLAUDE.md: "not just the trivial call
already tested").

Also folds in the second live-catalog finding from this session:
`groq/compound-mini` -- the model already verified last session for the
four small-task roles (`RouterSettings`, `QuerySettings.rewrite/expansion`,
`CitationSettings.entailment_model`) -- is now a 404 too. Groq's catalog
churned again in the five weeks since last session. If qwen3.8-27b is
clean at real length, it's evaluated here as a single-model candidate for
*all* five roles (primary + the four small-task ones), not just primary --
simpler than juggling two live-fire-tested models when the small-task
replacement just died again.

Same structure as `11a_gpt_oss_reasoning_tax.py`: real retrieval, real
packed context, real v2 prompt, real `GroqSettings.max_tokens=1024` budget,
non-streaming so `usage.completion_tokens_details` is readable.
"""

from __future__ import annotations

import json
import random
import time

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.generation.budgeter import format_context, pack_context
from aegis.generation.prompts import get_prompt
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, GoldenQuery, Retrieved

N_QUERIES = 6
MODEL = "qwen/qwen3.8-27b"
MAX_TOKENS = settings.groq.max_tokens  # 1024 -- real generation budget
RESULTS_RAW = REPO_ROOT / "docs" / "results" / "qwen3-8-27b-reasoning-check-raw.json"


def load_chunks() -> list[Chunk]:
    path = settings.paths.data_processed / "chunks.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def load_sample_queries(n: int) -> list[GoldenQuery]:
    path = settings.paths.data_golden / "golden_set_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        queries = [GoldenQuery(**json.loads(line)) for line in f]
    positives = [q for q in queries if not q.is_negative]
    return random.Random(settings.golden.random_seed).sample(positives, n)


def main() -> None:
    from groq import Groq

    chunks = load_chunks()
    queries = load_sample_queries(N_QUERIES)
    print(f"{len(chunks)} chunks, {len(queries)} sampled dev positives")

    cache = EmbeddingCache(settings.paths.embedding_cache)
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=cache, device="cpu")
    dense_index = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)
    prompt = get_prompt(settings.generation.prompt_version)
    client = Groq(api_key=settings.groq.api_key)

    raw: list[dict] = []
    for q in queries:
        results = dense_index.search(q.text, top_k=settings.generation.retrieval_top_k, exclude_doc_id=q.source_issue)
        retrieved = [Retrieved(chunk=c, score=s, rank=i + 1) for i, (c, s) in enumerate(results)]
        context = pack_context(retrieved, settings.generation.context_budget_tokens)
        messages = prompt.build_messages(q.text, format_context(context))

        start = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                temperature=settings.groq.temperature,
                top_p=settings.groq.top_p,
                max_tokens=MAX_TOKENS,
                messages=messages,
                stream=False,
            )
        except Exception as exc:  # noqa: BLE001 -- log and continue, don't lose an otherwise-real run
            print(f"  query {q.id} failed: {exc!r}")
            raw.append({"query_id": q.id, "error": repr(exc)})
            continue
        latency_ms = (time.perf_counter() - start) * 1000

        choice = completion.choices[0]
        usage = completion.usage
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
        visible_text = choice.message.content or ""

        row = {
            "query_id": q.id,
            "finish_reason": choice.finish_reason,
            "completion_tokens": usage.completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "visible_chars": len(visible_text),
            "visible_empty": len(visible_text.strip()) == 0,
            "latency_ms": latency_ms,
            "answer_text": visible_text,
        }
        raw.append(row)
        print(
            f"  [{q.id}] finish={row['finish_reason']} completion_tokens={row['completion_tokens']} "
            f"reasoning_tokens={row['reasoning_tokens']} visible_chars={row['visible_chars']} "
            f"latency_ms={latency_ms:.0f}"
        )
        RESULTS_RAW.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_RAW.write_text(json.dumps(raw, indent=2))

    succeeded = [r for r in raw if "error" not in r]
    empty = [r for r in succeeded if r["visible_empty"]]
    print(f"\n{len(succeeded)}/{len(raw)} succeeded, {len(empty)}/{len(succeeded)} came back with empty visible text")
    if succeeded:
        mean_reasoning = sum(r["reasoning_tokens"] or 0 for r in succeeded) / len(succeeded)
        mean_completion = sum(r["completion_tokens"] for r in succeeded) / len(succeeded)
        mean_latency = sum(r["latency_ms"] for r in succeeded) / len(succeeded)
        print(f"mean reasoning_tokens={mean_reasoning:.1f} mean completion_tokens={mean_completion:.1f}")
        print(f"mean latency_ms={mean_latency:.0f}")
    print(f"Raw results written to {RESULTS_RAW}")


if __name__ == "__main__":
    main()
