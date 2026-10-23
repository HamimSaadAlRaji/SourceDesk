"""Phase 15 step 1 (provider migration, design record 15g1): generation
bake-off on OpenAI -- gpt-4.1-mini vs gpt-4o-mini vs gpt-4.1-nano -- against
qwen/qwen3.8-27b's Phase 9 number (valid_citation_rate 0.7296).

Same protocol as `09a_sampling_sweep.py`'s temperature=0.0 config: the same 20
seeded dev positives (asserted against the sweep's raw file), dense bge-large
top-10 with `exclude_doc_id`, 4,096-token packing, prompt v2, temperature 0.0,
max_tokens 400. Retrieval runs once and is shared by every model.

The validator's entailment judge is held fixed. qwen's 0.7296 was scored with
qwen-on-Groq entailment; the served default is now gpt-4.1-nano. Changing both
generator and judge at once would make any difference unattributable, so every
answer set -- including qwen's 20 stored answers -- is validated under BOTH
judges:
  - `qwen` judge: like-for-like against 0.7296 (free, Groq quota)
  - `nano` judge: the judge actually served (paid, through the ledger)
Stage 0 re-validates qwen's stored answers with the qwen judge; if that does not
reproduce 0.7296 the harness is not trusted and the run stops.

Entailment here does NOT fail closed. The served `_default_entailment_fn`
returns False on an API error (right for serving, wrong for measuring: a 429
would silently become "unsupported" and lower the metric). This wrapper retries
rate limits and raises on persistent failure. Unparseable verdicts still count
as False (same as served) and are tallied.

Spend: every OpenAI call goes through the ledger with purpose
`bakeoff_generation` / `entailment`; per-answer cost is read back from the
ledger row the call settled. Resumable: generations already in the raw file are
never paid for twice.
"""

from __future__ import annotations

import json
import random
import sqlite3
import statistics
import sys
import time

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.generation.budgeter import PackedContext, pack_context
from aegis.generation.llm.chat import LLMCallError, chat_completion
from aegis.generation.llm.fallback import FallbackLLMClient
from aegis.generation.llm.openai_provider import OpenAIProvider
from aegis.generation.pipeline import DraftAnswer, GenerationPipeline
from aegis.guardrails.citations import ENTAILMENT_SYSTEM_PROMPT, CitationValidator
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, GoldenQuery, Query, Retrieved

N_QUERIES = 20
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 400  # identical to the qwen run (09a)
MODELS = ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1-nano"]
# Step 1d (15g2 follow-up, user OK 2026-10-19): reasoning models, added with `--extra-models`.
# max_completion_tokens caps hidden reasoning + answer together, so 400 is kept for parity and
# truncation (output_tokens == 400) and reasoning share are reported rather than avoided.
JUDGES = {"qwen": ("groq", "qwen/qwen3.8-27b"), "nano": ("openai", "gpt-4.1-nano")}
QWEN_REFERENCE_RATE = 0.7296
GROQ_PAUSE_S = 4.0  # 7,000 input tokens/min cap on Groq; entailment calls are ~300-700 tokens
MAX_RETRIES = 5
SWEEP_RAW = REPO_ROOT / "docs" / "results" / "sampling-sweep-raw.json"
RESULTS_RAW = REPO_ROOT / "docs" / "results" / "generation-bakeoff-openai-raw.json"
RESULTS_MD = REPO_ROOT / "docs" / "results" / "generation-bakeoff-openai-table.md"
GEN_PURPOSE = "bakeoff_generation"


class EntailmentFailed(RuntimeError):
    pass


def make_entailment_fn(judge: str, stats: dict):
    provider, model = JUDGES[judge]
    memo: dict[tuple[str, str], bool] = {}

    def entails(claim: str, chunk_text: str) -> bool:
        key = (claim, chunk_text)
        if key in memo:  # identical pair across answer sets: same verdict, no second call
            stats["memo_hits"] += 1
            return memo[key]
        messages = [{"role": "system", "content": ENTAILMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{chunk_text}\n\nClaim:\n{claim}"}]
        for attempt in range(MAX_RETRIES):
            if provider == "groq":
                time.sleep(GROQ_PAUSE_S)
            try:
                raw = chat_completion(provider, model, messages, temperature=0.0, max_tokens=10,
                                      purpose="entailment", json_mode=True)
                break
            except LLMCallError as exc:
                stats["api_errors"] += 1
                if exc.status_code == 429 and attempt < MAX_RETRIES - 1:
                    time.sleep(20 * (attempt + 1))
                    continue
                raise EntailmentFailed(f"{judge}: {exc.__cause__!r}") from exc
        stats["calls"] += 1
        try:
            verdict = bool(json.loads(raw)["supported"])
        except (json.JSONDecodeError, KeyError, TypeError):
            stats["unparseable"] += 1
            verdict = False
        memo[key] = verdict
        return verdict

    return entails


def _o200k():
    import tiktoken

    return tiktoken.get_encoding("o200k_base")


def load_queries() -> list[GoldenQuery]:
    path = settings.paths.data_golden / "golden_set_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        queries = [GoldenQuery(**json.loads(line)) for line in f]
    positives = [q for q in queries if not q.is_negative]
    return random.Random(settings.golden.random_seed).sample(positives, N_QUERIES)


def qwen_stored_answers() -> dict[str, str]:
    sweep = json.loads(SWEEP_RAW.read_text())
    rows = [r for r in sweep["raw"] if r["temperature"] == 0.0 and r["top_p"] == 1.0]
    return {r["query_id"]: r["answer_text"] for r in rows}


def last_generation_cost(model: str, purpose: str = GEN_PURPOSE) -> dict:
    with sqlite3.connect(settings.openai.ledger_path) as c:
        row = c.execute("SELECT input_tokens, cached_input_tokens, output_tokens, cost_usd FROM spend "
                        "WHERE model=? AND purpose=? ORDER BY ts DESC LIMIT 1", (model, purpose)).fetchone()
    return dict(zip(["input_tokens", "cached_input_tokens", "output_tokens", "cost_usd"], row))


def validate(validator: CitationValidator, text: str, context: PackedContext) -> dict:
    v = validator.validate(text, context)
    counts: dict[str, int] = {}
    for claim in v.claims:
        counts[claim.status.value] = counts.get(claim.status.value, 0) + 1
    return {
        "valid_citation_rate": v.valid_citation_rate,
        "tier2_calls": v.tier2_calls,
        "claim_status_counts": counts,
        "claims": [{"sentence": c.sentence, "citation_ids": c.citation_ids, "status": c.status.value} for c in v.claims],
    }


def save(data: dict) -> None:
    RESULTS_RAW.write_text(json.dumps(data, indent=2))


def main() -> None:
    if "--extra-models" in sys.argv:
        MODELS.extend(sys.argv[sys.argv.index("--extra-models") + 1].split(","))
    queries = load_queries()
    qwen_answers = qwen_stored_answers()
    if [q.id for q in queries] != list(qwen_answers):
        sys.exit("query sample differs from the Phase 9 sweep -- not comparable, stopping")

    chunks = [Chunk(**json.loads(l)) for l in open(settings.paths.data_processed / "chunks.jsonl", encoding="utf-8")]
    encoder = Encoder(model_name=settings.embedding.bge_large_model,
                      cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
    dense = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)
    contexts: dict[str, PackedContext] = {}
    retrieved: dict[str, list[Retrieved]] = {}
    for q in queries:
        res = dense.search(q.text, top_k=settings.generation.retrieval_top_k, exclude_doc_id=q.source_issue)
        retrieved[q.id] = [Retrieved(chunk=c, score=s, rank=i + 1) for i, (c, s) in enumerate(res)]
        contexts[q.id] = pack_context(retrieved[q.id], settings.generation.context_budget_tokens)

    data = json.loads(RESULTS_RAW.read_text()) if RESULTS_RAW.exists() else {}
    data.setdefault("config", {"n_queries": N_QUERIES, "temperature": TEMPERATURE, "top_p": TOP_P,
                               "max_tokens": MAX_TOKENS, "prompt_version": "v2", "models": MODELS,
                               "judges": {k: list(v) for k, v in JUDGES.items()}})
    data.setdefault("generations", {})  # model -> query_id -> {text, ttft_ms, total_ms, cost...}
    data.setdefault("validations", {})  # judge -> model -> query_id -> validation
    data["generations"]["qwen/qwen3.8-27b"] = {qid: {"text": t, "source": "sampling-sweep-raw.json"}
                                               for qid, t in qwen_answers.items()}
    judge_stats = {j: {"calls": 0, "memo_hits": 0, "api_errors": 0, "unparseable": 0} for j in JUDGES}
    validators = {j: CitationValidator(entailment_fn=make_entailment_fn(j, judge_stats[j])) for j in JUDGES}

    def validate_set(judge: str, model: str) -> list[float]:
        done = data["validations"].setdefault(judge, {}).setdefault(model, {})
        for q in queries:
            if q.id not in done:
                done[q.id] = validate(validators[judge], data["generations"][model][q.id]["text"], contexts[q.id])
                save(data)
        return [done[q.id]["valid_citation_rate"] for q in queries]

    # Stage 0: harness check -- reproduce qwen's number with qwen's judge.
    rates = validate_set("qwen", "qwen/qwen3.8-27b")
    reproduced = statistics.mean(rates)
    print(f"[stage 0] qwen answers, qwen judge: {reproduced:.4f} (reference {QWEN_REFERENCE_RATE})")
    data["harness_check"] = {"reproduced": reproduced, "reference": QWEN_REFERENCE_RATE}
    save(data)
    if abs(reproduced - QWEN_REFERENCE_RATE) > 0.0005 and "--accept-drift" not in sys.argv:
        sys.exit("harness does not reproduce 0.7296 -- investigate before spending (rerun with --accept-drift to override)")

    # Stage 1: generation (paid). Never re-pays a generation already on disk.
    for model in MODELS:
        gens = data["generations"].setdefault(model, {})
        pipeline = GenerationPipeline(llm=FallbackLLMClient([OpenAIProvider(model=model, purpose=GEN_PURPOSE)]),
                                      validator=validators["nano"], temperature=TEMPERATURE, top_p=TOP_P,
                                      max_tokens=MAX_TOKENS)
        for q in queries:
            if q.id in gens:
                continue
            first: list[float] = []
            start = time.perf_counter()
            draft: DraftAnswer = pipeline.generate(Query(id=q.id, text=q.text), retrieved[q.id],
                                                   on_token=lambda _t: first or first.append(time.perf_counter()))
            total_ms = (time.perf_counter() - start) * 1000
            gens[q.id] = {"text": draft.text, "provider": draft.provider,
                          "visible_tokens": len(_o200k().encode(draft.text)),
                          "ttft_ms": (first[0] - start) * 1000 if first else None, "total_ms": total_ms,
                          **last_generation_cost(model)}
            save(data)
            print(f"  {model} {q.id}: {total_ms:.0f} ms, {gens[q.id]['output_tokens']} out, ${gens[q.id]['cost_usd']:.6f}")

    # Stage 2: validate every set under both judges.
    for judge in JUDGES:
        for model in ["qwen/qwen3.8-27b", *MODELS]:
            r = validate_set(judge, model)
            print(f"[{judge} judge] {model}: {statistics.mean(r):.4f}")
    data["judge_stats_this_run"] = judge_stats
    save(data)
    write_table(data, queries)
    print(f"raw: {RESULTS_RAW}\ntable: {RESULTS_MD}")


def write_table(data: dict, queries: list[GoldenQuery]) -> None:
    ids = [q.id for q in queries]
    lines = ["# Generation bake-off (OpenAI) -- generated table", "",
             "Prompt v2, temperature 0.0, max_tokens 400, 20 seeded dev positives (same as Phase 9 sweep). "
             "Analysis: `generation-bakeoff-openai.md`.", "",
             "| generator | judge | mean valid_citation_rate | claim status counts |", "|---|---|---|---|"]
    for judge, by_model in data["validations"].items():
        for model, by_q in by_model.items():
            counts: dict[str, int] = {}
            for qid in ids:
                for k, v in by_q[qid]["claim_status_counts"].items():
                    counts[k] = counts.get(k, 0) + v
            mean = statistics.mean(by_q[qid]["valid_citation_rate"] for qid in ids)
            lines.append(f"| {model} | {judge} | {mean:.4f} | {', '.join(f'{k}={v}' for k, v in sorted(counts.items()))} |")
    lines += ["", "| generator | mean TTFT ms | p50 total ms | max total ms | mean output tokens | answers at cap | total cost USD | cost/answer USD |",
              "|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        g = [data["generations"][model][qid] for qid in ids]
        cost = sum(x["cost_usd"] for x in g)
        lines.append(f"| {model} | {statistics.mean(x['ttft_ms'] for x in g):.0f} | "
                     f"{statistics.median(x['total_ms'] for x in g):.0f} | {max(x['total_ms'] for x in g):.0f} | "
                     f"{statistics.mean(x['output_tokens'] for x in g):.0f} | {sum(x['output_tokens'] >= MAX_TOKENS for x in g)} | "
                     f"{cost:.6f} | {cost / len(g):.6f} |")
    RESULTS_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
