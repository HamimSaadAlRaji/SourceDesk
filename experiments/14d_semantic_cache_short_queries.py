"""Phase 14 investigation (design record 14h): why did the semantic cache serve
"How do I make a query parameter required in FastAPI?" the cached answer to
"How do I declare a query parameter?" (design record 14e)?

Phase 12's threshold sweep (12a) compared short positives (golden-issue titles
vs. short paraphrases) against long negatives (pairs of *full* golden-issue
bodies). Length alone changes cosine similarity, so "0 false positives at 0.88"
was never measured on short questions -- the shape chat traffic has. This
reruns the sweep with like-for-like negatives: pairs of real, distinct golden
*titles* (the first line of each issue -- short questions real users wrote).

Zero LLM cost: bge-large on CPU through the same `Encoder` + embedding cache
the serving semantic cache uses.

    uv run python experiments/14d_semantic_cache_short_queries.py
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder

OUT = REPO_ROOT / "docs" / "results" / "semantic-cache-short-queries-raw.json"
THRESHOLDS = [0.85, 0.88, 0.90, 0.92, 0.94, 0.96]
N_HARDEST = 15

# The real 14e collision, plus the cached question it was served from. The
# cached entry's exact wording isn't recorded (the 14e session's caches were
# throwaway), so both recorded phrasings are measured.
COLLISION = [
    ("How do I make a query parameter required in FastAPI?", "How do I declare a query parameter?"),
    ("How do I make a query parameter required in FastAPI?", "how do I declare a query parameter"),
]


def _load_12a_paraphrases() -> list[tuple[str, str]]:
    spec = importlib.util.spec_from_file_location("sweep12a", REPO_ROOT / "experiments" / "12a_semantic_cache_threshold_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PARAPHRASE_PAIRS


def _titles() -> list[tuple[str, str]]:
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [(r["id"], r["text"].strip().splitlines()[0].strip()) for r in rows if not r["is_negative"]]


def main() -> None:
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")

    titles = _titles()
    vecs = encoder.encode([t for _, t in titles])
    sims = vecs @ vecs.T
    iu = np.triu_indices(len(titles), k=1)
    all_neg = sims[iu]
    order = np.argsort(-all_neg)
    hardest = [(float(all_neg[k]), titles[iu[0][k]], titles[iu[1][k]]) for k in order[:N_HARDEST]]

    paraphrases = _load_12a_paraphrases()
    pos = [float(encoder.encode([a])[0] @ encoder.encode([b])[0]) for a, b in paraphrases]
    collision = [(a, b, float(encoder.encode([a])[0] @ encoder.encode([b])[0])) for a, b in COLLISION]

    sweep = []
    for t in THRESHOLDS:
        sweep.append({
            "threshold": t,
            "hardest15_false_positives": sum(s >= t for s, _, _ in hardest),
            "all_title_pairs_above": int((all_neg >= t).sum()),
            "true_positives_of_15": sum(s >= t for s in pos),
        })

    out = {
        "n_titles": len(titles), "n_pairs": int(all_neg.size),
        "title_pair_similarity": {"p50": float(np.median(all_neg)), "p99": float(np.percentile(all_neg, 99)), "max": float(all_neg.max())},
        "hardest": [{"similarity": round(s, 4), "a": a, "b": b} for s, a, b in hardest],
        "paraphrase_similarities": [round(s, 4) for s in pos],
        "collision": [{"a": a, "b": b, "similarity": round(s, 4)} for a, b, s in collision],
        "sweep": sweep,
        "shipped_threshold": settings.cache.semantic_similarity_threshold,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"{out['n_titles']} real dev titles, {out['n_pairs']} distinct pairs; similarity {out['title_pair_similarity']}")
    print("\nthe 14e collision:")
    for c in out["collision"]:
        print(f"  {c['similarity']:.4f}  {c['a']!r} vs {c['b']!r}")
    print("\nhardest distinct-title pairs:")
    for h in out["hardest"]:
        print(f"  {h['similarity']:.4f}  {h['a'][1][:70]!r} | {h['b'][1][:70]!r}")
    print("\nthreshold | FP of hardest 15 | all distinct pairs >= t | TP of 15 paraphrases")
    for s in sweep:
        print(f"  {s['threshold']:.2f}    | {s['hardest15_false_positives']:>2}/15            | {s['all_title_pairs_above']:>4}                    | {s['true_positives_of_15']:>2}/15")
    print("raw ->", OUT.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
