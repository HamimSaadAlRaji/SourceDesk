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

    api_key: str = ""
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_tokens: int = 1024


class RouterSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ROUTER_", env_file=".env", extra="ignore")

    # Phase 8, step 1: "a *small* LLM classifier" -- deliberately not the
    # same 70B model used for generation (Phase 9 hasn't even bake-off'd
    # that yet). Classification into 3 labels is a cheap task; paying
    # 70B-model latency for it on every query would eat into the 3s
    # end-to-end budget (RerankSettings.pipeline_latency_budget_ms) for no
    # accuracy this task needs.
    model: str = "llama-3.1-8b-instant"
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
    # expansion are short generation tasks, not the main answer.
    rewrite_model: str = "llama-3.1-8b-instant"
    expansion_model: str = "llama-3.1-8b-instant"
    # HyDE's hypothetical answer is embedded, not shown to the user, but
    # its *content* directly determines retrieval quality (plan.md step 6)
    # -- kept on the larger model since a thin/wrong hypothetical answer
    # would corrupt every downstream dense score, not just look worse.
    hyde_model: str = "llama-3.3-70b-versatile"
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
    router: RouterSettings = RouterSettings()
    query: QuerySettings = QuerySettings()


settings = Settings()
