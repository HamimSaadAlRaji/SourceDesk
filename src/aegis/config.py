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


class GroqSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GROQ_", env_file=".env", extra="ignore")

    api_key: str = ""
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_tokens: int = 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    paths: PathSettings = PathSettings()
    qdrant: QdrantSettings = QdrantSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    groq: GroqSettings = GroqSettings()


settings = Settings()
