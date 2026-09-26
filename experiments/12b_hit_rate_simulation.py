"""Phase 12 Unit 3: hit-rate simulation against a realistic (Zipfian, not
uniform) query distribution -- plan.md's own framing: real support traffic
is dominated by a small set of frequently repeated questions, not a
uniform sample over every distinct question that could be asked.

Zero LLM cost. Real components throughout:
- The real `ExactMatchCache`/`SemanticCache` classes (Units 1-2), same
  `corpus_version` scoping and `CachedAnswer` payload shape they ship with.
- Real `bge-large` embeddings (same encoder as retrieval) deciding every
  semantic-tier hit/miss, at the real tuned threshold (`docs/results/
  semantic-cache-threshold.md`, 0.88) -- not a labeled/assumed outcome.
- Real per-query generation cost: bootstrapped (with replacement) from the
  4 real successful (latency_ms, completion_tokens) samples in
  `docs/results/qwen3-8-27b-reasoning-check-raw.json`
  (`experiments/11b_qwen_reasoning_check.py`). Deliberately *not* re-run
  fresh here -- spending N fresh generations to measure how much the cache
  saves on generations would burn exactly the quota headroom this cache
  exists to protect (the same discipline Phase 9/11's deferred-to-the-user
  reruns already established).

What's a disclosed modeling assumption, not measured data:
- The Zipf skew itself. This system has never carried real production
  traffic, so there is no real query log to fit a skew parameter against --
  swept across a few literature-typical values instead of asserting one.
- Which of the 150 real dev-positive intents can appear "paraphrased"
  in the simulated stream: only the 15 with a real hand-written paraphrase
  from Unit 2 (`experiments/12a_semantic_cache_threshold_sweep.py`,
  duplicated here rather than imported -- a leading-digit module name isn't
  importable as a normal dotted path, and these are already-published,
  fixed real data, not re-derived). This makes the simulation's semantic-
  tier hit rate a conservative lower bound: real users would presumably
  rephrase *any* of the 150 intents, not just these 15.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from aegis.cache.exact_match import CachedAnswer, ExactMatchCache
from aegis.cache.semantic import SemanticCache
from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder

RESULTS_MD = REPO_ROOT / "docs" / "results" / "cache-hit-rate-simulation.md"

CORPUS_VERSION = "sim-corpus-v1"
# Checkpoints, not just a single N: with only 150 real unique intents, the
# ratio of requests to unique intents (traffic volume, i.e. how "mature"
# the cache is) turned out in a first real run to dominate the exact-hit
# rate far more than skew does -- reporting cumulative hit rate at several
# points along one real request stream shows that curve directly instead
# of hiding it behind one arbitrary N.
CHECKPOINTS = [150, 450, 1500, 4500, 9000]
N_REQUESTS = CHECKPOINTS[-1]
ZIPF_SKEWS = [0.8, 1.2, 1.6]
PARAPHRASE_REPEAT_PROB = 0.5

# Real 4/6 successful (latency_ms, completion_tokens) pairs -- see module
# docstring. The 2 missing rows hit a real RateLimitError, not a
# measurement gap; excluded, not imputed.
MISS_COST_SAMPLES = [
    (879.19, 140),
    (865.41, 237),
    (7857.12, 188),
    (653.33, 157),
]

# Real dense-retrieval latency added on top of every cache miss (`docs/
# results/ann-recall-vs-latency.md`, m=8/ef_construct=50/ef_search=50).
# Reranking is off by default (config.py RerankSettings.enabled=False,
# measured to hurt this golden set -- docs/results/rerank-tradeoff.md), so
# it isn't part of the real miss-path cost.
RETRIEVAL_LATENCY_MS = 6.54

# Real 15 (original title, paraphrase) pairs from Unit 2
# (experiments/12a_semantic_cache_threshold_sweep.py) -- duplicated, not
# re-derived; see module docstring.
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


class _MemoizedEncoder:
    """Wraps the real Encoder, memoizing by exact text. A deterministic
    encoder in eval mode gives identical vectors for identical strings, so
    memoizing only skips re-computing what real byte-identical repeats
    would compute anyway -- at most 165 distinct real strings (150 titles
    + 15 paraphrases) ever get a real encode call in this run, regardless
    of how many of the N_REQUESTS simulated queries reuse one."""

    def __init__(self, encoder: Encoder):
        self._encoder = encoder
        self._cache: dict[str, np.ndarray] = {}
        self.real_encode_calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        misses = [t for t in texts if t not in self._cache]
        if misses:
            vecs = self._encoder.encode(misses)
            self.real_encode_calls += len(misses)
            for t, v in zip(misses, vecs):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])


def load_intents() -> list[tuple[str, str]]:
    """(id, title) for every real dev-positive golden query. `text` is
    title + full issue body (used for retrieval eval); a live chat query
    looks like the title alone, so that's what this simulation's stream
    of "user queries" is built from."""
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [(r["id"], r["text"].split("\n")[0].strip()) for r in rows if not r["is_negative"]]


def zipf_pmf(n: int, skew: float) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = ranks ** (-skew)
    return weights / weights.sum()


def run_simulation(
    skew: float,
    intents: list[tuple[str, str]],
    paraphrase_by_id: dict[str, str],
    rng: np.random.Generator,
    exact_cache: ExactMatchCache,
    semantic_cache: SemanticCache,
) -> dict:
    n = len(intents)
    pmf = zipf_pmf(n, skew)
    # Fixed random rank assignment (independent of golden-set file order):
    # rank_of_intent[i] is intent i's Zipf rank (0 = most popular).
    rank_of_intent = rng.permutation(n)
    probs = pmf[rank_of_intent]

    draws = rng.choice(n, size=N_REQUESTS, p=probs)

    tiers = {"exact": 0, "semantic": 0, "miss": 0}
    latency_with_cache_ms = 0.0
    latency_without_cache_ms = 0.0
    tokens_with_cache = 0
    tokens_without_cache = 0
    exact_hit_timings_ms: list[float] = []
    semantic_hit_timings_ms: list[float] = []
    checkpoint_set = set(CHECKPOINTS)
    checkpoints: list[dict] = []

    for request_num, draw_idx in enumerate(draws, start=1):
        intent_id, title = intents[draw_idx]
        use_paraphrase = intent_id in paraphrase_by_id and rng.random() < PARAPHRASE_REPEAT_PROB
        query_text = paraphrase_by_id[intent_id] if use_paraphrase else title

        # Every request's "if this had been a cold miss" cost is sampled
        # once, up front, so the with-cache and without-cache totals are
        # computed on the same per-request cost basis.
        sample_idx = rng.integers(0, len(MISS_COST_SAMPLES))
        miss_latency_ms, miss_tokens = MISS_COST_SAMPLES[sample_idx]
        miss_latency_ms += RETRIEVAL_LATENCY_MS
        latency_without_cache_ms += miss_latency_ms
        tokens_without_cache += miss_tokens

        t0 = time.perf_counter()
        exact_hit = exact_cache.get(query_text, CORPUS_VERSION)
        t_exact = (time.perf_counter() - t0) * 1000

        if exact_hit is not None:
            tiers["exact"] += 1
            exact_hit_timings_ms.append(t_exact)
            latency_with_cache_ms += t_exact
        else:
            t0 = time.perf_counter()
            semantic_hit = semantic_cache.get(query_text, CORPUS_VERSION)
            t_semantic = (time.perf_counter() - t0) * 1000

            if semantic_hit is not None:
                tiers["semantic"] += 1
                semantic_hit_timings_ms.append(t_semantic)
                latency_with_cache_ms += t_exact + t_semantic
            else:
                tiers["miss"] += 1
                latency_with_cache_ms += t_exact + t_semantic + miss_latency_ms
                tokens_with_cache += miss_tokens

                placeholder = CachedAnswer(
                    text=f"[simulated placeholder answer for {intent_id} -- payload content carries no metric weight here]",
                    citations=[],
                    provider="groq",
                )
                exact_cache.set(query_text, CORPUS_VERSION, placeholder)
                semantic_cache.set(query_text, CORPUS_VERSION, placeholder)

        if request_num in checkpoint_set:
            checkpoints.append({
                "skew": skew,
                "n_requests": request_num,
                "n_unique_intents": n,
                "tier_counts": dict(tiers),
                "hit_rate_exact": tiers["exact"] / request_num,
                "hit_rate_semantic": tiers["semantic"] / request_num,
                "hit_rate_total": (tiers["exact"] + tiers["semantic"]) / request_num,
                "latency_with_cache_ms": latency_with_cache_ms,
                "latency_without_cache_ms": latency_without_cache_ms,
                "latency_reduction_pct": 1 - (latency_with_cache_ms / latency_without_cache_ms),
                "tokens_with_cache": tokens_with_cache,
                "tokens_without_cache": tokens_without_cache,
                "token_reduction_pct": 1 - (tokens_with_cache / tokens_without_cache),
                "mean_exact_hit_latency_ms": float(np.mean(exact_hit_timings_ms)) if exact_hit_timings_ms else None,
                "mean_semantic_hit_latency_ms": float(np.mean(semantic_hit_timings_ms)) if semantic_hit_timings_ms else None,
            })

    return checkpoints


def main() -> None:
    intents = load_intents()
    print(f"{len(intents)} real dev-positive intents")

    title_to_id = {title: intent_id for intent_id, title in intents}
    paraphrase_by_id: dict[str, str] = {}
    for orig_title, paraphrase in PARAPHRASE_PAIRS:
        intent_id = title_to_id.get(orig_title)
        if intent_id is None:
            raise ValueError(f"paraphrase original not found in golden set: {orig_title!r}")
        paraphrase_by_id[intent_id] = paraphrase
    print(f"{len(paraphrase_by_id)}/{len(intents)} intents have a real known paraphrase")

    embedding_cache = EmbeddingCache(settings.paths.embedding_cache)
    real_encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=embedding_cache, device="cpu")
    memoized_encoder = _MemoizedEncoder(real_encoder)

    all_checkpoints: list[dict] = []
    final_by_skew: dict[float, dict] = {}
    for skew in ZIPF_SKEWS:
        rng = np.random.default_rng(settings.golden.random_seed)
        with_tmp = REPO_ROOT / "data" / "cache" / f"_sim_exact_{skew}.sqlite"
        with_tmp.unlink(missing_ok=True)
        sem_tmp = REPO_ROOT / "data" / "cache" / f"_sim_semantic_{skew}.sqlite"
        sem_tmp.unlink(missing_ok=True)
        exact_cache = ExactMatchCache(with_tmp)
        semantic_cache = SemanticCache(sem_tmp, memoized_encoder, threshold=settings.cache.semantic_similarity_threshold)

        checkpoints = run_simulation(skew, intents, paraphrase_by_id, rng, exact_cache, semantic_cache)
        all_checkpoints.extend(checkpoints)
        final_by_skew[skew] = checkpoints[-1]
        for c in checkpoints:
            print(
                f"skew={skew} n={c['n_requests']}: hit_rate={c['hit_rate_total']:.1%} "
                f"(exact={c['hit_rate_exact']:.1%}, semantic={c['hit_rate_semantic']:.1%}) "
                f"latency_reduction={c['latency_reduction_pct']:.1%} "
                f"token_reduction={c['token_reduction_pct']:.1%}"
            )
        with_tmp.unlink(missing_ok=True)
        sem_tmp.unlink(missing_ok=True)

    print(f"\n{memoized_encoder.real_encode_calls} real bge-large encode calls total (memoized by exact text)")

    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cache Hit-Rate Simulation — Zipfian Traffic (Phase 12 Unit 3)",
        "",
        f"Up to {N_REQUESTS} simulated requests per skew value, drawn over the {len(intents)} real "
        f"dev-positive golden intents (`golden_set_v1.jsonl`), real `ExactMatchCache`/`SemanticCache` "
        f"classes (threshold={settings.cache.semantic_similarity_threshold}), real `bge-large` embeddings "
        f"({memoized_encoder.real_encode_calls} real encode calls total, memoized by exact text). Zero LLM "
        "calls -- cache-miss cost bootstrapped from 4 real (latency_ms, completion_tokens) samples "
        "(`docs/results/qwen3-8-27b-reasoning-check-raw.json`) plus real dense-retrieval latency "
        f"({RETRIEVAL_LATENCY_MS} ms, `docs/results/ann-recall-vs-latency.md`).",
        "",
        "No real production traffic log exists for this system (never deployed) -- the Zipf skew is a "
        "disclosed modeling assumption, swept rather than asserted. Only "
        f"{len(paraphrase_by_id)}/{len(intents)} intents have a real known paraphrase (Unit 2's hand-written "
        "set), so the semantic-tier hit rate here is a conservative lower bound, not an estimate of the "
        "full population.",
        "",
        "## Hit rate vs. traffic volume (requests per unique intent), by skew",
        "",
        "Cumulative hit rate at each checkpoint within one real request stream per skew -- reported this "
        f"way, not just at one final N, because a first real run showed traffic volume (N over only "
        f"{len(intents)} real unique intents) drives the exact-hit rate far more than skew shape does once "
        "N is large relative to the intent pool; see 'what surprised me' in the design record.",
        "",
        "| skew | N | requests/intent | hit rate (total) | exact | semantic | latency reduction | token reduction |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in all_checkpoints:
        lines.append(
            f"| {c['skew']} | {c['n_requests']} | {c['n_requests'] / c['n_unique_intents']:.1f} | "
            f"{c['hit_rate_total']:.1%} | {c['hit_rate_exact']:.1%} | {c['hit_rate_semantic']:.1%} | "
            f"{c['latency_reduction_pct']:.1%} | {c['token_reduction_pct']:.1%} |"
        )

    lines += ["", "## Detail at final checkpoint per skew", ""]
    for skew in ZIPF_SKEWS:
        r = final_by_skew[skew]
        lines += [
            f"### skew = {skew}",
            "",
            f"- Tier counts (of {r['n_requests']}): exact={r['tier_counts']['exact']}, "
            f"semantic={r['tier_counts']['semantic']}, miss={r['tier_counts']['miss']}",
            f"- Real mean exact-hit latency: {r['mean_exact_hit_latency_ms']:.3f} ms" if r['mean_exact_hit_latency_ms'] else "- No exact hits",
            f"- Real mean semantic-hit latency: {r['mean_semantic_hit_latency_ms']:.3f} ms" if r['mean_semantic_hit_latency_ms'] else "- No semantic hits",
            f"- Total latency: {r['latency_with_cache_ms']/1000:.1f}s with caching vs. "
            f"{r['latency_without_cache_ms']/1000:.1f}s without ({r['latency_reduction_pct']:.1%} reduction)",
            f"- Total completion tokens: {r['tokens_with_cache']} with caching vs. "
            f"{r['tokens_without_cache']} without ({r['token_reduction_pct']:.1%} reduction)",
            "",
        ]

    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {RESULTS_MD}")


if __name__ == "__main__":
    main()
