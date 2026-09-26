"""Every tunable, in one place.

Scattered constants make ablation studies impossible, and this project is
mostly ablation studies (plan.md Phase 0, CLAUDE.md §6). New tunables get
added here as each phase introduces them — never as magic numbers in logic.

Each settings group reads from `.env` via its own `env_prefix`, e.g.
`GROQ_API_KEY` -> `Settings().groq.api_key`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class PathSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AEGIS_PATH_", env_file=".env", extra="ignore")

    data_raw: Path = REPO_ROOT / "data" / "raw"
    data_processed: Path = REPO_ROOT / "data" / "processed"
    data_golden: Path = REPO_ROOT / "data" / "golden"
    embedding_cache: Path = REPO_ROOT / "data" / "cache" / "embeddings.sqlite"

    # Phase 12 Unit 4 -- one manifest per collection records exactly which
    # chunk_id/content_hash pairs are indexed, so the next reindex can diff
    # against it (aegis.ingest.delta) instead of re-embedding everything.
    index_manifest_dir: Path = REPO_ROOT / "data" / "processed" / "manifests"


class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QDRANT_", env_file=".env", extra="ignore")

    host: str = "localhost"
    port: int = 6333
    collection_baseline: str = "aegis_baseline"

    # Phase 3: one collection per encoder candidate (dev workhorse + the two
    # real bake-off models) rather than multiple named vectors sharing one
    # collection — simpler to reason about, and Phase 6's ablation swaps
    # between them by config anyway.
    collection_bge_small: str = "aegis_chunks_bge_small"
    collection_bge_large: str = "aegis_chunks_bge_large"
    collection_bge_m3: str = "aegis_chunks_bge_m3"

    # HNSW params. Qdrant's own defaults until the Phase 3 recall/latency
    # sweep picks a justified point — see docs/results/ann-recall-vs-latency.md.
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    hnsw_ef_search: int = 100


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", env_file=".env", extra="ignore")

    # Phase 0 only: fast, local, ONNX (fastembed) — no MPS setup needed for a
    # throwaway baseline. Phase 3 bakes off bge-large-en-v1.5 vs BGE-M3 for real.
    baseline_model: str = "BAAI/bge-small-en-v1.5"

    # Phase 3: dev workhorse (fast iteration, used for the HNSW sweep) plus
    # the two real bake-off candidates. All three loaded via
    # sentence-transformers (not fastembed) so the same wrapper/cache path
    # serves all of Phase 3 — full comparison lands in Phase 6.
    dev_model: str = "BAAI/bge-small-en-v1.5"
    bge_large_model: str = "BAAI/bge-large-en-v1.5"
    bge_m3_model: str = "BAAI/bge-m3"

    device: str = "mps"
    batch_size: int = 32


class ChunkingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHUNKING_", env_file=".env", extra="ignore")

    # Naive fixed-size chunking, on purpose — this is the Air Canada
    # architecture the baseline is built to fail. Phase 2 replaces it.
    baseline_chunk_chars: int = 500

    # Phase 2: shared ceiling across all four strategies, so the ablation grid
    # (Phase 6) compares chunking *logic*, not different size budgets. 400,
    # not the bge-large/BGE-M3 512-token limit itself, to leave headroom for
    # the contextual-enrichment prefix (title > heading_path > version) added
    # on top without silently truncating the encoder's input.
    max_tokens: int = 400
    overlap_tokens: int = 50

    # Semantic chunker: a sentence-to-sentence cosine-distance percentile
    # above which a boundary is cut. Reuses the same bge-small encoder as the
    # Phase 0 baseline — a real bake-off model would be overkill for drawing
    # split points, and this component is discarded/replaced if Phase 6 shows
    # it doesn't pay for itself (plan.md cut-line #1).
    semantic_breakpoint_percentile: float = 0.95
    semantic_min_sentences_per_group: int = 1

    # Ablation axis (plan.md Phase 2 exit criteria) — off by default so every
    # strategy's own measurement isn't silently altered by enrichment text.
    contextual_enrichment: bool = False


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", env_file=".env", extra="ignore")

    baseline_top_k: int = 3

    # Phase 5: bm25s defaults (Lucene's own defaults, per bm25s's `method`
    # parameter). Starting point for the Phase 5/6 k1/b sweep -- not yet
    # tuned against the golden set.
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Phase 5: RRF's rank-damping constant (plan.md's `k` in
    # `score(d) = sum(weight_r / (k + rank_r(d)))`). 60 is the value from
    # the original Cormack et al. RRF paper and the common default;
    # Phase 6 tunes it on the training slice.
    rrf_k: int = 60
    rrf_dense_weight: float = 1.0
    rrf_sparse_weight: float = 1.0

    # Phase 5: how many candidates each side (dense, sparse) contributes to
    # a hybrid fusion before truncating to the caller's requested `top_k`.
    # Must exceed top_k or fusion has no breadth to find cross-method
    # overlap in (retrieval/fusion.py's whole premise). Phase 7's reranker
    # pool-size sweep is a related but separate question (how many fused
    # results to *rerank*, not how many each retriever contributes here).
    hybrid_candidate_pool_size: int = 50


class RerankSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RERANK_", env_file=".env", extra="ignore")

    # Cross-encoder (plan.md Phase 7, step 1) -- scores (query, chunk_text)
    # jointly instead of independently encoding each side, catching
    # interactions a bi-encoder's cosine similarity can't (design record
    # 07 has the attention-pattern mechanism). Loaded via
    # sentence-transformers.CrossEncoder, same MPS/batching treatment as
    # aegis.embedding.encoder.Encoder.
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "mps"
    batch_size: int = 32

    # Candidate pool sizes swept (plan.md Phase 7, step 2) -- the axis
    # being measured: how many bi-encoder candidates to rerank before
    # truncating to the system's real top_k. Reranking more costs O(n)
    # cross-encoder calls; the sweep finds where NDCG@5 plateaus against
    # that cost.
    pool_sizes: list[int] = [10, 20, 50, 100]
    final_top_k: int = 10

    # Off by default -- measured (docs/results/rerank-tradeoff.md): every
    # swept pool size makes NDCG@5 *worse* than the no-rerank baseline
    # (-9.0% at pool=10, worsening monotonically to -55.8% at pool=100),
    # and every pool size's own p95 latency alone already exceeds the full
    # pipeline's 3s budget. Root cause is in the golden set's labeling
    # methodology, not the reranker (design record 07 §6) -- kept as a
    # tested, toggleable component (RerankedRetriever.enabled) rather than
    # removed, in case Phase 11's judge-agreement measurement changes this.
    enabled: bool = False

    # Not a reranking-only budget -- Phase 13's full end-to-end p95 target
    # (plan.md exit criterion). Used here only as the bar the pool-size
    # choice is weighed against, since Phase 13's other stages (generation,
    # etc.) don't exist yet to measure the real remaining headroom.
    pipeline_latency_budget_ms: int = 3000


class GoldenSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GOLDEN_", env_file=".env", extra="ignore")

    # 6 strata x 30 = 180 (plan.md exit criterion: 180 total, >= 20/stratum).
    # Holding out 5/stratum leaves 25/stratum in the dev split, still >= 20 --
    # the invariant Phase 6's per-stratum reporting actually needs.
    target_per_stratum: int = 30
    holdout_per_stratum: int = 5
    n_negatives: int = 40

    # Not every harvested candidate survives human labeling -- some
    # "resolutions" are non-answers the low-signal filter didn't catch
    # (a meta-discussion, a redirect with unusual phrasing). Harvest more
    # than the target and let the labeling CLI stop accepting once a
    # stratum hits `target_per_stratum`, rather than chasing a perfect
    # algorithmic pre-filter (same bounded-residual-limitation call as
    # Phase 1/2's canned-reply and unclosed-fence handling).
    harvest_buffer_multiplier: float = 2.0

    # "Permissive config" (plan.md step 2): dense retrieval over one of the
    # two real bake-off encoders -- not bge-small, which Phase 3 only ever
    # used as the fast dev workhorse. Split by doc_type rather than one
    # flat top-20: measured on the real corpus (see docs/explainers/
    # 04-building-a-golden-set.md) that a single unfiltered top-20 is
    # 83.6% issue chunks and misses documentation entirely for 30.9% of
    # candidates -- an artifact of the shared GitHub issue-template
    # checklist boilerplate dominating similarity scores, not genuine
    # topical closeness (the same boilerplate-pollution failure mode
    # Phases 1-3 each independently hit). A doc_type-filtered search for
    # the same query surfaces the actually-correct doc chunk instead.
    permissive_model_key: str = "bge_large"
    permissive_top_k_docs: int = 12
    permissive_top_k_issues: int = 8

    random_seed: int = 20260802


class GroqSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GROQ_", env_file=".env", extra="ignore")

    # `llama-3.3-70b-versatile` (the original pick) and `llama-3.1-8b-instant`
    # (RouterSettings/QuerySettings/CitationSettings' original small-task
    # model) were both retired from Groq's catalog entirely between Phase 9
    # and this check -- confirmed via a live `client.models.list()` call and
    # real 404s (Open Question #15). Investigated replacements directly
    # rather than guessing: `openai/gpt-oss-120b` is reasoning-first with a
    # real, non-fatal but real tax (25-45% of completion_tokens across 6 real
    # answer-length dev-positive generations) and a real mean latency of
    # 16.7s (1.2-29s) -- nearly as slow as Gemini's already-rejected 21.4s,
    # undermining the entire "Groq primary because it's fast" premise.
    # `groq/compound-mini` (verified for the small-task roles in the same
    # session that found the two retirements) was *also* 404 by the time
    # this session re-checked the catalog five weeks later -- Groq's catalog
    # is churning fast enough that "verified last session" doesn't carry
    # forward. `qwen/qwen3.8-27b` (a renamed successor to `qwen3.6-27b`,
    # which was itself reasoning-first and rejected last session) measured
    # clean instead: 0 reasoning tokens across the same 6 real generations,
    # mean latency 8.7s -- slower than the dead 70B model's bake-off 1.5s,
    # but ~2.5x faster than Gemini. One real free-tier constraint found and
    # fixed: a 1000 output-tokens/minute cap causes a real 429 at the old
    # max_tokens=1024 (confirmed: 2/6 real calls failed at 1024, both
    # succeeded once retried at 1000). Picked as a single model for *all*
    # five Groq-backed roles (this settings group plus RouterSettings,
    # QuerySettings.rewrite/expansion/hyde_model, CitationSettings.
    # entailment_model) rather than pairing two independently-verified
    # models, given how quickly one of the two already died again.
    api_key: str = ""
    model: str = "qwen/qwen3.8-27b"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1000


class GeminiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GEMINI_", env_file=".env", extra="ignore")

    # Phase 9, step 1: the fallback provider (plan.md "Groq primary, Gemini
    # secondary"). Verified against a real call, not just `client.models.
    # list()` -- `gemini-2.5-flash`/`gemini-2.5-flash-lite` both appear in
    # the list endpoint but return a real 404 "no longer available to new
    # users" for this API key. `gemini-3.5-flash` is the pinned model that
    # actually answers for this key (checked 2026-08-16); `gemini-flash-
    # latest` was also confirmed working but is an alias, not pinned --
    # rejected for reproducibility (a later re-run could silently resolve
    # to a different underlying model).
    api_key: str = ""
    model: str = "gemini-3.5-flash"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024


class RouterSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ROUTER_", env_file=".env", extra="ignore")

    # Phase 8, step 1: "a *small* LLM classifier" -- originally deliberately
    # not the same model used for generation. `llama-3.1-8b-instant` was
    # retired from Groq's catalog (see GroqSettings' comment, Open Question
    # #15); its real replacement, `groq/compound-mini`, died too by the time
    # this session re-checked. Reusing `qwen/qwen3.8-27b` (GroqSettings.model)
    # here instead of hunting for a second small model -- this task's real
    # output is ~8-12 tokens (`{"route": "..."}`), far under the model's
    # measured 1000 output-tokens/minute cap, so the "why pay 70B latency for
    # a cheap task" concern doesn't reappear at this call size.
    model: str = "qwen/qwen3.8-27b"
    temperature: float = 0.0
    # 20 was too tight -- measured: Groq's JSON-mode grammar constraint
    # occasionally emits a few tokens of formatting/whitespace before the
    # closing brace and hit `json_validate_failed` (truncated mid-document)
    # on real golden-set queries during the Phase 8 router eval. 50 gives
    # headroom; the real output is `{"route": "..."}`, ~8-12 tokens.
    max_tokens: int = 50


class QuerySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QUERY_", env_file=".env", extra="ignore")

    # Same "small model" reasoning as RouterSettings -- rewriting and
    # expansion are short generation tasks, not the main answer. Same
    # retirement/replacement chain as GroqSettings/RouterSettings (Open
    # Question #15) -- `qwen/qwen3.8-27b` for all three roles here, since a
    # second independently-verified small model (`groq/compound-mini`)
    # didn't survive to this session either.
    rewrite_model: str = "qwen/qwen3.8-27b"
    expansion_model: str = "qwen/qwen3.8-27b"
    # HyDE's hypothetical answer is embedded, not shown to the user, but
    # its *content* directly determines retrieval quality (plan.md step 6)
    # -- kept on the larger/primary model since a thin/wrong hypothetical
    # answer would corrupt every downstream dense score, not just look
    # worse. Same model as GroqSettings.model now that one model covers
    # both roles (see GroqSettings' comment).
    hyde_model: str = "qwen/qwen3.8-27b"
    temperature: float = 0.0

    n_expansions: int = 3

    # Measured (docs/results/query-understanding-ablation.md, 150-query dev
    # set + 24-item synthetic multi-turn set): only rewriting pays for
    # itself. Expansion and HyDE both regress hard (NDCG@5 -49.4% and
    # -23.6%) and are off, same "off until proven" convention as
    # RerankSettings.enabled and ChunkingSettings.contextual_enrichment.
    rewrite_enabled: bool = True
    expansion_enabled: bool = False
    hyde_enabled: bool = False
    iterative_retrieval_enabled: bool = False

    # Plan.md step 8: "capped at one" retry.
    iterative_retrieval_max_retries: int = 1
    # Provisional proxy for "low confidence" -- Phase 10 hasn't built the
    # real confidence scorer yet. Top-1 fused score below this triggers one
    # reformulate-and-retry. Calibrated as the 25th percentile of the real
    # baseline top-1 score distribution over the 150-query dev set
    # (docs/results/query-understanding-ablation.md) -- not an arbitrary
    # default; replaced by Phase 10's scorer once it exists (same
    # provisional-then-superseded pattern as Phase 3 persisting
    # heading-aware chunks before Phase 6 picked it).
    low_confidence_score_threshold: float = 0.7747


class CitationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CITATION_", env_file=".env", extra="ignore")

    # Tier 1 (plan.md Phase 9, step 4): cheap lexical overlap between a claim
    # sentence and its cited chunk's text, same "fraction of tokens found in
    # the other text" technique Phase 4's golden-set auto-labeling already
    # validated on this corpus (`eval/auto_label.py`, threshold 0.15 there).
    # Set higher here as a starting point -- claims are short, specific
    # sentences, not a whole query, so a real match should share a much
    # larger fraction of its tokens with the source than a query does with
    # a broad topical chunk. Checked against real generations in the Phase 9
    # design record, not assumed correct.
    lexical_overlap_threshold: float = 0.4

    # Tier 2: LLM entailment, only for claims tier 1 couldn't confidently
    # pass -- same "small model for a cheap classification task" convention
    # as RouterSettings/QuerySettings, and same retirement/replacement chain
    # (Open Question #15) -- `qwen/qwen3.8-27b`.
    entailment_model: str = "qwen/qwen3.8-27b"
    entailment_temperature: float = 0.0


class ConfidenceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONFIDENCE_", env_file=".env", extra="ignore")

    # Phase 10, step 1: only rank 1 and rank 2 are ever read (reranker_top_score,
    # score_gap), so only a small pool needs the cross-encoder at all. Real
    # per-pair cost on this machine's CPU (measured on real corpus-length
    # chunks, not synthetic text -- see design record 10 §5): ~65-1400ms,
    # mean ~320ms across a real 6-query probe. At pool_size=3, a full
    # 180-query dev sweep (150 positives + 30 negatives) projects to several
    # minutes, comfortably inside CLAUDE.md's 10-minute local-compute budget;
    # reranking the full top-10 risked crossing it for signal this scorer
    # doesn't need (rank 3+ is never read).
    rerank_pool_size: int = 3

    # plan.md Phase 10 step 1 starts from "a transparent weighted
    # combination" -- tried equal (0.25 each) first, per the real
    # 180-query dev sweep (experiments/10a_tau_sweep.py). Measured result:
    # equal weighting scored *worse* than reranker_top_score alone (0.8333
    # vs 0.9333 recall at precision>=0.5) because score_gap and
    # distribution_shape are both close to uncorrelated with the real
    # should-escalate label (|rank-correlation| 0.16 and 0.02) and dilute
    # the one strong signal (reranker_top_score, |rank-correlation| 0.58).
    # Reweighted by measured strength instead of shipping the uniform
    # default -- confirmed with the user (AskUserQuestion) given it's a
    # real deviation from plan.md's literal starting point. All 4 signals
    # stay in the combination (semantic_agreement, |rank-correlation| 0.22,
    # earns a real share too) -- see design record 10 §5.
    weight_reranker_top_score: float = 0.60
    weight_score_gap: float = 0.15
    weight_semantic_agreement: float = 0.20
    weight_distribution_shape: float = 0.05

    # Min-max normalization bounds per raw signal, so four differently-scaled
    # numbers (a cross-encoder logit-like score, a small score gap, a cosine
    # similarity, a z-score-like ratio) can be weighted-averaged meaningfully.
    # 1st/99th percentile of the real 180-query dev sweep, not raw min/max --
    # distribution_shape's raw max was 76,856.55 (one query, near-zero
    # variance among its non-top-1 dense scores blew up the ratio;
    # traces to the same near-duplicate-chunk corpus property as Open
    # Question #6), which under raw min-max would have compressed every
    # other query's real range into the bottom ~0.1% of [0, 1]. Winsorizing
    # at p1/p99 keeps the bounds representative of the typical case instead
    # of one outlier -- same "measure first, hardcode the real number"
    # convention as QuerySettings.low_confidence_score_threshold.
    reranker_top_score_lo: float = 0.0001
    reranker_top_score_hi: float = 0.9936
    score_gap_lo: float = 0.0000
    score_gap_hi: float = 0.6943
    semantic_agreement_lo: float = 0.6420
    semantic_agreement_hi: float = 0.9976
    distribution_shape_lo: float = 1.4188
    distribution_shape_hi: float = 27.5129

    # Escalation threshold (plan.md Phase 10, steps 2 & 4): chosen from the
    # real precision/recall curve over the 180-query dev set (150 positive
    # + 30 negative), presented to the user as 3 real candidate operating
    # points with the full confusion matrix at each (AskUserQuestion) given
    # plan.md's own framing of this as "the decision that matters most."
    # At tau=0.50: catches 27/30 negatives (90% recall, only 3 slip through
    # -- all unanswerable/out_of_corpus, not billing_legal or
    # adversarial_injection, both 100% caught) while still directly
    # answering 126/150 (84%) of legitimate questions. Chosen over a
    # deflection-preserving tau=0.15 (70% recall, 9/30 slip through) and a
    # high-recall tau=0.65 (97% recall, but only 50% deflection) as the
    # point that most reduces the asymmetric-cost Air Canada quadrant
    # without going degenerate. See docs/results/confusion-matrix.md.
    tau: float = 0.50


class GenerationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GENERATION_", env_file=".env", extra="ignore")

    # Real per-chunk token_count stats measured on the persisted 19,236-chunk
    # corpus (heading-aware, enrichment off -- Phase 6's winning config):
    # p50=79, p90=363, p95=390 tokens (see design record 09 §5). Retrieval's
    # own default top_k is 10 (HybridRetriever/RerankSettings.final_top_k),
    # so 10 chunks at p90 size is ~3,630 tokens -- 4096 leaves headroom for
    # that case, while the median case (~790 tokens for 10 chunks) rarely
    # triggers pruning at all. Not the model's real context window (131,072
    # for `qwen/qwen3.8-27b`, per `client.models.list()` -- GroqSettings.model)
    # -- a deliberate cost/latency budget, the same "why not put everything
    # in the prompt" lesson Phase 0 measured directly, applied to a real
    # number instead of "the whole window."
    context_budget_tokens: int = 4096
    retrieval_top_k: int = 10

    # Order matters: fallback tries providers left to right, only advancing
    # past a provider if it fails *before* yielding its first token (see
    # `generation/llm/fallback.py` -- a mid-stream failure propagates rather
    # than silently splicing in a second provider's answer after the first
    # provider's partial output).
    provider_order: list[str] = ["groq", "gemini"]

    # v2 promoted over v1 on real, matched-pair evidence (N=6, same 6 real
    # queries, both versions, Groq): mean valid_citation_rate 0.4199 -> 0.5641
    # (+34.3% relative), v2 winning or tying on 6/6 comparisons, 0 losses.
    # See docs/results/prompt-v1-vs-v2.md and design record 09 -- N=6 is
    # modest, but the direction is consistent and mechanistically explained
    # (v2's explicit first-sentence citation rule fixes exactly the pattern
    # it targeted, confirmed by inspecting real before/after answer pairs).
    prompt_version: str = "v2"


class JudgeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JUDGE_", env_file=".env", extra="ignore")

    # Second judge (2026-09-25 decision, progress.md Decision Log -- inter-
    # judge agreement replaces judge-vs-hand-label since the user declined
    # to hand-label). Deliberately NOT `GroqSettings.model`
    # (`qwen/qwen3.8-27b`, the generator) -- an architecturally distinct
    # model so neither judge scores its own generation. Reasoning-first
    # (real tax measured this phase's prerequisite work: 25-45% of
    # completion_tokens, mean latency 16.7s, `docs/results/groq-model-
    # replacement.md`) -- acceptable here since judge calls sit off the
    # request-latency path, unlike generation. The first judge reuses
    # `GeminiSettings.model` directly (no separate field) -- it's already
    # "the" Gemini model this project uses everywhere and was locked in
    # last session specifically as the starting judge model.
    groq_model: str = "openai/gpt-oss-120b"
    temperature: float = 0.0

    # RAGAS's own scoring LLM (Faithfulness/ResponseRelevancy's internal
    # statement-generation + NLI-entailment calls) needs its own model and
    # max_tokens, decoupled from GroqSettings -- found running Unit 4's real
    # batch: `qwen/qwen3.8-27b` (GroqSettings.model) hits its own real,
    # measured 1000 output-tokens/minute ceiling (Open Question #15) on
    # these occasionally-longer RAGAS-internal generations, producing a
    # real `LLMDidNotFinishException`. `openai/gpt-oss-120b` doesn't hit
    # that specific error -- reused from `groq_model` above rather than
    # verifying a third model, since it's already this phase's second judge.
    #
    # 2048 wasn't always enough (one real dev-positive query still hit
    # `LLMDidNotFinishException` at 2048); 4096 cleared it.
    #
    # Originally also `openai/gpt-oss-120b` (same as `groq_model` above),
    # until running Unit 4's real N=30 batch hit a real, hard wall: that
    # model has its own 200,000 tokens/day (TPD) cap, shared account-wide
    # per-model (not per-role) -- Faithfulness (2 calls/query) +
    # ResponseRelevancy (3 calls/query) + the judge role (1 call/query) all
    # drew from the same bucket, and it hit 197,634/200,000 (98.8%,
    # accumulated across this whole session's real testing) after
    # essentially one query's worth of RAGAS calls, with a real ~18-minute
    # partial-refill message from Groq. Split onto `openai/gpt-oss-20b`
    # instead -- confirmed via a real call immediately after the 120b wall
    # that its quota bucket is genuinely independent (worked instantly),
    # and confirmed it can complete a real `Faithfulness` call at
    # max_tokens=4096 (real score 0.625) despite a *higher* measured
    # reasoning-token tax than 120b (204/295 = 69% on one real call vs.
    # 120b's 25-45%). `groq_model` (120b) now serves the judge role only,
    # which needed model-diversity from the generator anyway (design record
    # 11c) -- this split isn't a workaround, it also removes the earlier
    # same-model shared-bucket collision entirely, not just this instance
    # of it.
    ragas_llm_model: str = "openai/gpt-oss-20b"
    ragas_llm_max_tokens: int = 4096


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVAL_", env_file=".env", extra="ignore")

    # plan.md Phase 11 exit criteria -- the two numeric bars Unit 6's
    # scorecard checks the real measured numbers against. Not ablation
    # tunables (nothing sweeps these), but real thresholds used in
    # comparison logic, so CLAUDE.md SS6 puts them here rather than as
    # magic numbers in scorecard.py.
    ragas_faithfulness_target: float = 0.90
    judge_kappa_target: float = 0.6

    generation_batch_raw_path: Path = REPO_ROOT / "docs" / "results" / "judge-agreement-batch-raw.json"


class CacheSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CACHE_", env_file=".env", extra="ignore")

    exact_match_db_path: Path = REPO_ROOT / "data" / "cache" / "exact_match.sqlite"

    # Real sweep against 15 real hardest-similarity distinct-query pairs
    # (negatives) vs. 15 hand-written real-query paraphrases (positives),
    # bge-large encoder -- same asymmetric-cost logic as Phase 10's tau:
    # the lowest threshold in the sweep with zero false positives on the
    # real negative set (0.88), not the one maximizing hit rate. A false
    # positive here serves a confidently wrong cached answer; a false
    # negative just costs one extra real pipeline call.
    # docs/results/semantic-cache-threshold.md
    semantic_similarity_threshold: float = 0.88
    semantic_cache_db_path: Path = REPO_ROOT / "data" / "cache" / "semantic_cache.sqlite"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    paths: PathSettings = PathSettings()
    qdrant: QdrantSettings = QdrantSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    rerank: RerankSettings = RerankSettings()
    golden: GoldenSettings = GoldenSettings()
    groq: GroqSettings = GroqSettings()
    gemini: GeminiSettings = GeminiSettings()
    router: RouterSettings = RouterSettings()
    query: QuerySettings = QuerySettings()
    generation: GenerationSettings = GenerationSettings()
    citation: CitationSettings = CitationSettings()
    confidence: ConfidenceSettings = ConfidenceSettings()
    judge: JudgeSettings = JudgeSettings()
    eval: EvalSettings = EvalSettings()
    cache: CacheSettings = CacheSettings()


settings = Settings()
