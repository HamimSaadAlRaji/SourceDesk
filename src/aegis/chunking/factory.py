"""Config-driven strategy selection — the "one interface, config-swappable"
exit criterion. Phase 6's ablation grid sweeps `strategy` as one axis; this
is the single call site that lets it do so without touching chunker code.
"""

from __future__ import annotations

from aegis.chunking.base import Chunker
from aegis.chunking.fixed import FixedChunker
from aegis.chunking.heading_aware import HeadingAwareChunker
from aegis.chunking.recursive import RecursiveChunker
from aegis.chunking.semantic import SemanticChunker
from aegis.config import settings

_STRATEGIES: dict[str, type] = {
    "fixed": FixedChunker,
    "recursive": RecursiveChunker,
    "heading_aware": HeadingAwareChunker,
    "semantic": SemanticChunker,
}


def build_chunker(strategy: str, *, contextual_enrichment: bool | None = None) -> Chunker:
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown chunking strategy {strategy!r}; expected one of {sorted(_STRATEGIES)}")

    cfg = settings.chunking
    enrichment = cfg.contextual_enrichment if contextual_enrichment is None else contextual_enrichment
    return _STRATEGIES[strategy](
        max_tokens=cfg.max_tokens,
        overlap_tokens=cfg.overlap_tokens,
        contextual_enrichment=enrichment,
    )
