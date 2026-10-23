"""15h H2 sensitivity check: is v3's valid_citation_rate gain an artifact of
claim granularity?

Found running H2: `split_sentences` doesn't split on newlines or list
bullets, so a bulleted answer becomes a few very long "claims" (v3 max 1,718
chars) that pass or fail as one unit -- and v3 answers carry about twice as
many bullet lines as v2's (36 vs 17). This re-scores BOTH prompt versions'
gpt-5-mini answers on the bake-off 20 with the splitter patched to also break
on line boundaries (each non-empty line, bullet markers kept), under both
judges. The served validator is NOT changed; this is a measurement-only patch.

Same stored answers, same contexts, same judges: only claim boundaries move.
"""

from __future__ import annotations

import importlib.util
import json
import random
import statistics

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.generation.budgeter import pack_context
from aegis.guardrails import citations
from aegis.guardrails.citations import CitationValidator
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, Retrieved

_spec = importlib.util.spec_from_file_location("bakeoff", REPO_ROOT / "experiments" / "15g1_generation_bakeoff.py")
bakeoff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bakeoff)

REALSTACK_RAW = REPO_ROOT / "docs" / "results" / "security-realstack-raw.json"
_original_split = citations.split_sentences


def line_aware_split(text: str) -> list[str]:
    return [s for line in text.splitlines() if line.strip() for s in _original_split(line)]


def main() -> None:
    realstack = json.loads(REALSTACK_RAW.read_text())
    bake = json.loads(bakeoff.RESULTS_RAW.read_text())
    answers = {"v2": {q: g["text"] for q, g in bake["generations"]["gpt-5-mini"].items()},
               "v3": {q: g["text"] for q, g in realstack["h2"]["generations"].items()}}

    queries = bakeoff.load_queries()
    chunks = [Chunk(**json.loads(line)) for line in open(settings.paths.data_processed / "chunks.jsonl", encoding="utf-8")]
    dense = DenseIndex(chunks, settings.qdrant.collection_bge_large,
                       Encoder(model_name=settings.embedding.bge_large_model,
                               cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu"))
    contexts = {}
    for q in queries:
        res = dense.search(q.text, top_k=settings.generation.retrieval_top_k, exclude_doc_id=q.source_issue)
        contexts[q.id] = pack_context([Retrieved(chunk=c, score=s, rank=i + 1) for i, (c, s) in enumerate(res)],
                                      settings.generation.context_budget_tokens)

    citations.split_sentences = line_aware_split  # measurement-only patch, this process only
    stats = {j: {"calls": 0, "memo_hits": 0, "api_errors": 0, "unparseable": 0} for j in bakeoff.JUDGES}
    out = realstack.setdefault("h2_line_split", {})
    rng = random.Random(42)
    for judge in bakeoff.JUDGES:
        validator = CitationValidator(entailment_fn=bakeoff.make_entailment_fn(judge, stats[judge]))
        res = out.setdefault(judge, {})
        for prompt in ("v2", "v3"):
            done = res.setdefault(prompt, {})
            for q in queries:
                if q.id not in done:
                    done[q.id] = bakeoff.validate(validator, answers[prompt][q.id], contexts[q.id])
                    REALSTACK_RAW.write_text(json.dumps(realstack, indent=2))
        ids = [q.id for q in queries]
        diff = [res["v3"][i]["valid_citation_rate"] - res["v2"][i]["valid_citation_rate"] for i in ids]
        boot = sorted(statistics.mean(rng.choice(diff) for _ in ids) for _ in range(10000))
        for prompt in ("v2", "v3"):
            claims = [c for i in ids for c in res[prompt][i]["claims"]]
            sup = sum(c["status"].startswith("supported") for c in claims)
            uns = sum(c["status"] == "unsupported" for c in claims)
            print(f"[{judge}] {prompt}: rate {statistics.mean(res[prompt][i]['valid_citation_rate'] for i in ids):.4f} "
                  f"claims {len(claims)} supported {sup} unsupported {uns} "
                  f"missing {sum(c['status'] == 'missing_citation' for c in claims)} wrong-share {uns / max(1, sup + uns):.3f}")
        print(f"[{judge}] v3 - v2: {statistics.mean(diff):+.4f} CI [{boot[250]:+.4f}, {boot[9750]:+.4f}] "
              f"W/L {sum(d > 0 for d in diff)}/{sum(d < 0 for d in diff)}")
    realstack["h2_line_split_judge_stats"] = stats
    REALSTACK_RAW.write_text(json.dumps(realstack, indent=2))


if __name__ == "__main__":
    main()
