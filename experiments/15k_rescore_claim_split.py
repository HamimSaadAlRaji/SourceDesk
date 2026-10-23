"""Design record 15k: re-score the decision-relevant stored answers under the
fixed claim splitter (paragraph + list-item boundaries, Open Question #18).

No generation -- the stored answers of the 15g bake-off (qwen, gpt-4.1-mini,
gpt-4o-mini, gpt-4.1-nano, gpt-5-mini, gpt-5-nano; prompt v2) and of 15h H2
(gpt-5-mini, prompt v3) are re-validated with the current `CitationValidator`
under both judges. Old numbers stay in their own raw files; this writes a new
one. Paired bootstrap CIs for the two comparisons that drove decisions:
gpt-5-mini vs qwen (generator, 15g3) and v3 vs v2 (served prompt, 15h).
"""

from __future__ import annotations

import importlib.util
import json
import random
import statistics
import sys

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.generation.budgeter import pack_context
from aegis.generation.llm.openai_provider import default_ledger
from aegis.guardrails.citations import CitationValidator
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, Retrieved

_spec = importlib.util.spec_from_file_location("bakeoff", REPO_ROOT / "experiments" / "15g1_generation_bakeoff.py")
bakeoff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bakeoff)

STOP_AT_USD = 0.03
REALSTACK_RAW = REPO_ROOT / "docs" / "results" / "security-realstack-raw.json"
RESULTS_RAW = REPO_ROOT / "docs" / "results" / "claim-split-rescore-raw.json"
SETS_V2 = ["qwen/qwen3.8-27b", "gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1-nano", "gpt-5-mini", "gpt-5-nano"]


def main() -> None:
    bake = json.loads(bakeoff.RESULTS_RAW.read_text())
    real = json.loads(REALSTACK_RAW.read_text())
    answers = {f"{m}|v2": {q: g["text"] for q, g in bake["generations"][m].items()} for m in SETS_V2}
    answers["gpt-5-mini|v3"] = {q: g["text"] for q, g in real["h2"]["generations"].items()}
    old = {j: {f"{m}|v2": bake["validations"][j][m] for m in SETS_V2} | {"gpt-5-mini|v3": real["h2"]["validations"][j]}
           for j in bakeoff.JUDGES}

    queries = bakeoff.load_queries()
    ids = [q.id for q in queries]
    chunks = [Chunk(**json.loads(line)) for line in open(settings.paths.data_processed / "chunks.jsonl", encoding="utf-8")]
    dense = DenseIndex(chunks, settings.qdrant.collection_bge_large,
                       Encoder(model_name=settings.embedding.bge_large_model,
                               cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu"))
    contexts = {}
    for q in queries:
        res = dense.search(q.text, top_k=settings.generation.retrieval_top_k, exclude_doc_id=q.source_issue)
        contexts[q.id] = pack_context([Retrieved(chunk=c, score=s, rank=i + 1) for i, (c, s) in enumerate(res)],
                                      settings.generation.context_budget_tokens)

    ledger = default_ledger()
    start = ledger.spent_usd()
    data = json.loads(RESULTS_RAW.read_text()) if RESULTS_RAW.exists() else {"validations": {}}
    stats = {j: {"calls": 0, "memo_hits": 0, "api_errors": 0, "unparseable": 0} for j in bakeoff.JUDGES}
    for judge in bakeoff.JUDGES:
        validator = CitationValidator(entailment_fn=bakeoff.make_entailment_fn(judge, stats[judge]))
        for name, by_q in answers.items():
            done = data["validations"].setdefault(judge, {}).setdefault(name, {})
            for qid in ids:
                if qid in done:
                    continue
                if ledger.spent_usd() - start > STOP_AT_USD:
                    RESULTS_RAW.write_text(json.dumps(data, indent=2))
                    sys.exit(f"spend exceeded ${STOP_AT_USD}")
                done[qid] = bakeoff.validate(validator, by_q[qid], contexts[qid])
                RESULTS_RAW.write_text(json.dumps(data, indent=2))

    rng = random.Random(42)
    summary: dict = {}
    for judge in bakeoff.JUDGES:
        new = data["validations"][judge]
        print(f"\n== {judge} judge: set | old rate -> new rate | new claims (sup/uns/miss) | wrong-cite share")
        for name in answers:
            claims = [c for qid in ids for c in new[name][qid]["claims"]]
            sup = sum(c["status"].startswith("supported") for c in claims)
            uns = sum(c["status"] == "unsupported" for c in claims)
            row = {"old": statistics.mean(old[judge][name][q]["valid_citation_rate"] for q in ids),
                   "new": statistics.mean(new[name][q]["valid_citation_rate"] for q in ids),
                   "claims_old": sum(len(old[judge][name][q]["claims"]) for q in ids), "claims_new": len(claims),
                   "supported": sup, "unsupported": uns,
                   "missing": sum(c["status"] == "missing_citation" for c in claims),
                   "wrong_cite_share": uns / max(1, sup + uns)}
            summary.setdefault(judge, {})[name] = row
            print(f"  {name:20} {row['old']:.4f} -> {row['new']:.4f} | claims {row['claims_old']} -> {row['claims_new']} "
                  f"({sup}/{uns}/{row['missing']}) | {row['wrong_cite_share']:.3f}")
        for a, b in [("gpt-5-mini|v2", "qwen/qwen3.8-27b|v2"), ("gpt-5-mini|v3", "gpt-5-mini|v2")]:
            diff = [new[a][q]["valid_citation_rate"] - new[b][q]["valid_citation_rate"] for q in ids]
            boot = sorted(statistics.mean(rng.choice(diff) for _ in ids) for _ in range(10000))
            comp = {"diff": statistics.mean(diff), "ci": [boot[250], boot[9750]],
                    "wins": sum(d > 0 for d in diff), "losses": sum(d < 0 for d in diff)}
            summary[judge][f"{a} - {b}"] = comp
            print(f"  {a} - {b}: {comp['diff']:+.4f} CI [{comp['ci'][0]:+.4f}, {comp['ci'][1]:+.4f}] W/L {comp['wins']}/{comp['losses']}")
    data["summary"] = summary
    data["judge_stats"] = stats
    data["spend_usd"] = ledger.spent_usd() - start
    RESULTS_RAW.write_text(json.dumps(data, indent=2))
    print(f"\nspend: ${data['spend_usd']:.6f}")


if __name__ == "__main__":
    main()
