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


class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QDRANT_", env_file=".env", extra="ignore")

    host: str = "localhost"
    port: int = 6333
    collection_baseline: str = "aegis_baseline"


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", env_file=".env", extra="ignore")

    # Phase 0 only: fast, local, ONNX (fastembed) — no MPS setup needed for a
    # throwaway baseline. Phase 3 bakes off bge-large-en-v1.5 vs BGE-M3 for real.
    baseline_model: str = "BAAI/bge-small-en-v1.5"


class ChunkingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHUNKING_", env_file=".env", extra="ignore")

    # Naive fixed-size chunking, on purpose — this is the Air Canada
    # architecture the baseline is built to fail. Phase 2 replaces it.
    baseline_chunk_chars: int = 500


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
