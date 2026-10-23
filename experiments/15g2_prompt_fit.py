"""Phase 15 step 1b (design record 15g2): is the OpenAI models' citation gap
a prompt-fit problem? Prompt v2 vs v2c on gpt-4.1-mini and gpt-4.1-nano.

Two stages, run separately on purpose:
  `--stage dev`   10 seeded dev positives DISJOINT from the bake-off's 20:
                  v2 and v2c x mini and nano. A sanity gate (does v2c move
                  behaviour in the intended direction without degenerating?)
                  before scoring on the comparison set.
  `--stage eval`  the bake-off's 20 queries, v2c only, mini and nano. The v2
                  answers for the same 20 are reused from 15g1's raw file.

Protocol, judges and the non-fail-closed entailment wrapper are 15g1's
(imported, not copied). A degenerate "win" is possible: an answer with no
checkable claims scores valid_citation_rate 1.0, so a prompt that makes the
model say less looks better. Reported alongside the rate: supported claims
(content volume), answers with no checkable claim, refusal-language answers.

Spend: hard stop if this script's spend in one invocation exceeds
STOP_AT_USD (the approved estimate was ~$0.09).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import statistics
import sys
import time

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.generation.budgeter import pack_context
from aegis.generation.llm.fallback import FallbackLLMClient
from aegis.generation.llm.openai_provider import OpenAIProvider, default_ledger
from aegis.generation.pipeline import GenerationPipeline
from aegis.generation.prompts import get_prompt
from aegis.guardrails.citations import CitationValidator
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, GoldenQuery, Query, Retrieved

_spec = importlib.util.spec_from_file_location("bakeoff", REPO_ROOT / "experiments" / "15g1_generation_bakeoff.py")
bakeoff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bakeoff)

MODELS = ["gpt-4.1-mini", "gpt-4.1-nano"]
N_DEV = 10
DEV_SEED_OFFSET = 1  # settings.golden.random_seed + 1: a different draw, then the eval 20 removed
STOP_AT_USD = 0.15
GEN_PURPOSE = "promptfit_generation"
RESULTS_RAW = REPO_ROOT / "docs" / "results" / "prompt-fit-openai-raw.json"
REFUSAL = re.compile(r"(does not|doesn.t) (contain|provide|address|specify|cover|say)|cannot (provide|confirm|answer)", re.I)


def dev_queries(eval_ids: set[str]) -> list[GoldenQuery]:
    path = settings.paths.data_golden / "golden_set_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        positives = [q for q in (GoldenQuery(**json.loads(l)) for l in f) if not q.is_negative and q.id not in eval_ids]
    return random.Random(settings.golden.random_seed + DEV_SEED_OFFSET).sample(positives, N_DEV)


def summarize(gens: dict, vals: dict, ids: list[str]) -> dict:
    claims = [c for qid in ids for c in vals[qid]["claims"]]
    supported = sum(c["status"].startswith("supported") for c in claims)
    unsupported = sum(c["status"] == "unsupported" for c in claims)
    return {
        "rate": statistics.mean(vals[qid]["valid_citation_rate"] for qid in ids),
        "claims": len(claims), "supported": supported, "unsupported": unsupported,
        "missing": sum(c["status"] == "missing_citation" for c in claims),
        "wrong_cite_share": unsupported / max(1, supported + unsupported),
        "no_checkable_claim_answers": sum(all(c["status"] == "exempt" for c in vals[qid]["claims"]) for qid in ids),
        "refusal_answers": sum(bool(REFUSAL.search(gens[qid]["text"])) for qid in ids),
        "mean_output_tokens": statistics.mean(gens[qid]["output_tokens"] for qid in ids),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "eval"], required=True)
    stage = ap.parse_args().stage

    eval_q = bakeoff.load_queries()
    queries = dev_queries({q.id for q in eval_q}) if stage == "dev" else eval_q
    prompts = ["v2", "v2c"] if stage == "dev" else ["v2c"]

    chunks = [Chunk(**json.loads(l)) for l in open(settings.paths.data_processed / "chunks.jsonl", encoding="utf-8")]
    encoder = Encoder(model_name=settings.embedding.bge_large_model,
                      cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
    dense = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)
    retrieved, contexts = {}, {}
    for q in queries:
        res = dense.search(q.text, top_k=settings.generation.retrieval_top_k, exclude_doc_id=q.source_issue)
        retrieved[q.id] = [Retrieved(chunk=c, score=s, rank=i + 1) for i, (c, s) in enumerate(res)]
        contexts[q.id] = pack_context(retrieved[q.id], settings.generation.context_budget_tokens)

    data = json.loads(RESULTS_RAW.read_text()) if RESULTS_RAW.exists() else {}
    data.setdefault("query_ids", {})[stage] = [q.id for q in queries]
    gens = data.setdefault("generations", {}).setdefault(stage, {})
    vals = data.setdefault("validations", {}).setdefault(stage, {})
    if stage == "eval":  # v2 answers on the eval 20 already exist: reuse 15g1's
        prior = json.loads(bakeoff.RESULTS_RAW.read_text())
        for m in MODELS:
            gens.setdefault("v2", {})[m] = prior["generations"][m]
            for judge in bakeoff.JUDGES:
                vals.setdefault(judge, {}).setdefault("v2", {})[m] = prior["validations"][judge][m]
    stats = {j: {"calls": 0, "memo_hits": 0, "api_errors": 0, "unparseable": 0} for j in bakeoff.JUDGES}
    validators = {j: CitationValidator(entailment_fn=bakeoff.make_entailment_fn(j, stats[j])) for j in bakeoff.JUDGES}
    ledger = default_ledger()
    start_spend = ledger.spent_usd()

    def save() -> None:
        RESULTS_RAW.write_text(json.dumps(data, indent=2))

    for prompt in prompts:
        for model in MODELS:
            out = gens.setdefault(prompt, {}).setdefault(model, {})
            pipeline = GenerationPipeline(llm=FallbackLLMClient([OpenAIProvider(model=model, purpose=GEN_PURPOSE)]),
                                          validator=validators["nano"], prompt=get_prompt(prompt),
                                          temperature=bakeoff.TEMPERATURE, top_p=bakeoff.TOP_P,
                                          max_tokens=bakeoff.MAX_TOKENS)
            for q in queries:
                if q.id in out:
                    continue
                if ledger.spent_usd() - start_spend > STOP_AT_USD:
                    save()
                    sys.exit(f"spend this run exceeded ${STOP_AT_USD} -- stopping")
                t0 = time.perf_counter()
                draft = pipeline.generate(Query(id=q.id, text=q.text), retrieved[q.id])
                out[q.id] = {"text": draft.text, "total_ms": (time.perf_counter() - t0) * 1000,
                             **bakeoff.last_generation_cost(model, purpose=GEN_PURPOSE)}
                save()

    for judge in bakeoff.JUDGES:
        for prompt in gens:
            for model in MODELS:
                done = vals.setdefault(judge, {}).setdefault(prompt, {}).setdefault(model, {})
                for q in queries:
                    if q.id not in done:
                        done[q.id] = bakeoff.validate(validators[judge], gens[prompt][model][q.id]["text"], contexts[q.id])
                        save()

    ids = [q.id for q in queries]
    summary = {judge: {prompt: {m: summarize(gens[prompt][m], vals[judge][prompt][m], ids) for m in MODELS}
                       for prompt in gens} for judge in bakeoff.JUDGES}
    data.setdefault("summary", {})[stage] = summary
    data.setdefault("judge_stats", {})[stage] = stats
    data.setdefault("spend_usd", {})[stage] = ledger.spent_usd() - start_spend
    save()
    print(json.dumps({"stage": stage, "spend_usd": data["spend_usd"][stage], "judge_stats": stats, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
