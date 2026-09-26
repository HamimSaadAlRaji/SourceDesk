"""Phase 12 Unit 2: tune the semantic cache's similarity threshold on real
data (plan.md's own instruction -- "too loose serves the wrong answer to a
similar-but-different question, which is a hallucination vector you built
yourself"). Zero LLM cost -- local `Encoder` (bge-large) calls only.

Two real, measured populations, same asymmetric-cost logic as Phase 10's
tau sweep (a false positive here is strictly worse than a false negative --
a missed cache hit costs one extra real retrieval+generation call, a false
hit serves a confidently wrong answer):

- **Real negatives**: the N highest-similarity pairs among genuinely
  *different* real dev-positive golden queries -- the actual hardest real
  near-miss cases this corpus produces, not a guess.
- **Real-ish positives**: hand-written paraphrases of 15 real golden
  queries, covering a real range of phrasing looseness (a close rewrite to
  a much looser one). Not LLM-generated (today's quota is committed
  elsewhere and this needs zero LLM cost to stay unblocked) -- a real,
  disclosed limitation: these are one person's paraphrases, not a sample of
  real user phrasing variance, and the measured hit-rate number should be
  read with that in mind, not treated as a population estimate.
"""

from __future__ import annotations

import json

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder

RESULTS_MD = REPO_ROOT / "docs" / "results" / "semantic-cache-threshold.md"

# Real paraphrase pairs (original golden-query text, hand-written rephrasing) --
# 15 real golden queries, a deliberate spread from a close rewrite to a loose one.
PARAPHRASE_PAIRS = [
    ("How can I get upload files?", "how do i receive an uploaded file in fastapi?"),
    ("How is the request body processed with pydantic?", "how does fastapi use pydantic to parse the request body?"),
    ("Generated OpenAPI schema does not reflect pydantic dynamic models.", "the openapi schema is wrong for my dynamic pydantic models"),
    ("performance latency after updating to Fastapi > 0.82.0", "why did latency get worse after upgrading fastapi past 0.82.0"),
    ("Yield dependency not cleaning up immediately after the request", "my yield-based dependency cleanup is delayed after the response is sent"),
    ("how do i add routes with equal sign?", "how can i define a path that includes an equals sign character?"),
    ("Combining Form() with Arbitrary dicts", "can i mix Form() fields with an arbitrary dict in the same request?"),
    ("Generate openapi schema without running server", "how do i produce the openapi schema without starting the server?"),
    ('Will FastAPI support QUERY http method? "app.query"', "is there a plan to add support for the http QUERY method, like app.query?"),
    ("How can I return a 202 Accepted response for long running REST call?", "what's the way to send back a 202 status for a slow-running endpoint?"),
    ("using Depends() to mock function inside endpoint", "how can i mock a function used via Depends() inside a path operation?"),
    ("How to get token of authenticated user in websocket?", "how do i access the logged-in user's auth token from a websocket connection?"),
    ("StreamResponse of output of several commands", "how do i stream the combined output of multiple shell commands as a response?"),
    ("Comparing django with fastapi ?", "how does fastapi compare to django overall?"),
    ("What the framework does, and what are the use cases of it?", "what is fastapi actually used for and what problems does it solve?"),
]

N_HARDEST_NEGATIVES = 15
THRESHOLDS = [0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96]


def load_dev_positive_texts() -> list[tuple[str, str]]:
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f]
    return [(q["id"], q["text"]) for q in queries if not q["is_negative"]]


def hardest_real_negative_pairs(encoder: Encoder, n: int) -> list[tuple[float, str, str]]:
    ids_texts = load_dev_positive_texts()
    ids = [t[0] for t in ids_texts]
    texts = [t[1] for t in ids_texts]
    vecs = encoder.encode(texts)
    sims = vecs @ vecs.T

    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pairs.append((float(sims[i, j]), texts[i], texts[j]))
    pairs.sort(reverse=True, key=lambda p: p[0])
    return pairs[:n]


def main() -> None:
    cache = EmbeddingCache(settings.paths.embedding_cache)
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=cache, device="cpu")

    negatives = hardest_real_negative_pairs(encoder, N_HARDEST_NEGATIVES)
    print(f"{len(negatives)} real hardest distinct-query pairs (negatives)")

    positive_sims = []
    for orig, paraphrase in PARAPHRASE_PAIRS:
        v = encoder.encode([orig, paraphrase])
        positive_sims.append(float(v[0] @ v[1]))
    print(f"{len(positive_sims)} real paraphrase pairs (positives)")

    negative_sims = [p[0] for p in negatives]

    print("\nthreshold | false_positive_rate (negatives wrongly matched) | true_positive_rate (positives correctly matched)")
    sweep_rows = []
    for t in THRESHOLDS:
        fp = sum(1 for s in negative_sims if s >= t)
        tp = sum(1 for s in positive_sims if s >= t)
        fpr = fp / len(negative_sims)
        tpr = tp / len(positive_sims)
        print(f"{t:.2f}      | {fp}/{len(negative_sims)} ({fpr:.2%})                              | {tp}/{len(positive_sims)} ({tpr:.2%})")
        sweep_rows.append({"threshold": t, "fp": fp, "n_negatives": len(negative_sims), "tp": tp, "n_positives": len(positive_sims)})

    chosen = None
    for row in sorted(sweep_rows, key=lambda r: r["threshold"]):
        if row["fp"] == 0:
            chosen = row
            break

    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Semantic Cache Threshold — Real Sweep (Phase 12 Unit 2)",
        "",
        "Zero LLM cost. Two real, measured populations against `bge-large` (the same encoder used for "
        "retrieval): the 15 highest-similarity pairs among genuinely *different* real dev-positive golden "
        "queries (real negatives -- the actual hardest near-miss cases this corpus produces), and 15 "
        "hand-written paraphrases of real golden queries (positives -- a real, disclosed limitation: "
        "one person's paraphrases, not a sample of real user phrasing variance).",
        "",
        "## The real negatives (highest-similarity distinct-question pairs)",
        "",
        "| similarity | query A | query B |",
        "|---|---|---|",
    ]
    for sim, a, b in negatives[:10]:
        lines.append(f"| {sim:.4f} | {a[:70].replace(chr(10), ' ')} | {b[:70].replace(chr(10), ' ')} |")

    lines += [
        "",
        "## Threshold sweep",
        "",
        "| threshold | false positives (of 15 real negatives) | true positives (of 15 real paraphrases) |",
        "|---|---|---|",
    ]
    for row in sweep_rows:
        lines.append(f"| {row['threshold']:.2f} | {row['fp']}/{row['n_negatives']} | {row['tp']}/{row['n_positives']} |")

    lines += [
        "",
        f"## Chosen threshold: {chosen['threshold']:.2f}" if chosen else "## No threshold in the sweep cleared 0 false positives",
        "",
    ]
    if chosen:
        lines.append(
            f"Lowest threshold in the sweep with **zero** false positives on the real hardest-negative set "
            f"({chosen['fp']}/{chosen['n_negatives']}), catching {chosen['tp']}/{chosen['n_positives']} real "
            f"paraphrases. Same asymmetric-cost logic as Phase 10's tau: a false positive here serves a "
            f"confidently wrong cached answer (this project's own North Star failure mode), a false negative "
            f"just costs one extra real pipeline call -- so the threshold is chosen to eliminate the former "
            f"on real measured data, not to maximize the latter."
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {RESULTS_MD}")


if __name__ == "__main__":
    main()
