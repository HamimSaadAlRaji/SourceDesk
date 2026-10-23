"""Semantic chunking — embed sentences, cut where consecutive-sentence
cosine distance spikes above a percentile threshold.

The candidate plan.md's own cut-line (§7) flags as least likely to earn its
keep on structured technical docs: it has no notion of Markdown headings, so
every chunk it produces has an empty `heading_path` — a real, measured cost
against heading-aware, not a hypothetical one. Built anyway (full Phase 2
build, not the cut-down path) so Phase 6's ablation grid has real numbers to
compare rather than an assumption.

The sentence splitter is a period/question/exclamation regex, not an NLP
sentence tokenizer (spaCy/nltk) — a heavier dependency isn't justified per
CLAUDE.md's build-vs-import test for a strategy this project already expects
might not pay for itself; see the design record for where it misfires
(abbreviations, numbered steps).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np

from aegis.chunking.base import make_embedding_text
from aegis.chunking.merge import greedy_merge
from aegis.chunking.recursive import recursive_split_pieces
from aegis.chunking.segments import split_code_segments
from aegis.chunking.tokens import count_tokens
from aegis.config import settings
from aegis.ingest.hashing import content_hash
from aegis.types import Chunk, Document

EmbedFn = Callable[[list[str]], list[list[float]]]

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'`])')
_MODEL_CACHE: dict[str, Any] = {}  # fastembed is imported lazily, so its type isn't nameable here


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def _default_embed(sentences: list[str]) -> list[list[float]]:
    model_name = settings.embedding.baseline_model
    if model_name not in _MODEL_CACHE:
        from fastembed import TextEmbedding

        _MODEL_CACHE[model_name] = TextEmbedding(model_name=model_name)
    return [vec.tolist() for vec in _MODEL_CACHE[model_name].embed(sentences)]


def _find_breakpoints(embeddings: list[list[float]], percentile: float) -> set[int]:
    """Sentence indices where a cut happens *before* that sentence."""
    if len(embeddings) < 2:
        return set()
    vecs = np.array(embeddings, dtype=float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    unit = vecs / norms
    similarities = np.sum(unit[:-1] * unit[1:], axis=1)
    distances = 1.0 - similarities
    threshold = float(np.percentile(distances, percentile * 100))
    return {i + 1 for i, d in enumerate(distances) if d > threshold}


def _group_sentences(sentences: list[str], breakpoints: set[int], min_group: int) -> list[str]:
    groups: list[list[str]] = []
    current: list[str] = []
    for i, sentence in enumerate(sentences):
        if i in breakpoints and len(current) >= min_group:
            groups.append(current)
            current = []
        current.append(sentence)
    if current:
        groups.append(current)
    return [" ".join(g) for g in groups]


class SemanticChunker:
    name = "semantic"

    def __init__(
        self,
        max_tokens: int,
        overlap_tokens: int,
        contextual_enrichment: bool = False,
        breakpoint_percentile: float | None = None,
        min_sentences_per_group: int | None = None,
        embed_fn: EmbedFn | None = None,
    ):
        if overlap_tokens >= max_tokens:
            raise ValueError(f"overlap_tokens ({overlap_tokens}) must be < max_tokens ({max_tokens})")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.contextual_enrichment = contextual_enrichment
        self.breakpoint_percentile = (
            breakpoint_percentile
            if breakpoint_percentile is not None
            else settings.chunking.semantic_breakpoint_percentile
        )
        self.min_sentences_per_group = (
            min_sentences_per_group
            if min_sentences_per_group is not None
            else settings.chunking.semantic_min_sentences_per_group
        )
        self._embed_fn = embed_fn or _default_embed

    def _semantic_groups(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return sentences
        embeddings = self._embed_fn(sentences)
        breakpoints = _find_breakpoints(embeddings, self.breakpoint_percentile)
        return _group_sentences(sentences, breakpoints, self.min_sentences_per_group)

    def chunk(self, document: Document) -> list[Chunk]:
        pieces: list[str] = []
        for seg in split_code_segments(document.text):
            if seg.kind == "code":
                pieces.append(seg.content)
                continue
            for group in self._semantic_groups(seg.content):
                if count_tokens(group) <= self.max_tokens:
                    pieces.append(group)
                else:
                    pieces.extend(recursive_split_pieces(group, self.max_tokens))

        if not pieces:
            return []

        merged = greedy_merge(pieces, self.max_tokens, self.overlap_tokens)
        chunks: list[Chunk] = []
        for index, text in enumerate(merged):
            embedding_text = make_embedding_text(text, document, [], self.contextual_enrichment)
            chunks.append(
                Chunk(
                    id=f"{document.id}::{self.name}::{index}",
                    doc_id=document.id,
                    text=text,
                    heading_path=[],
                    doc_version=document.doc_version,
                    doc_type=document.doc_type,
                    product_area=document.product_area,
                    content_hash=content_hash(text),
                    token_count=count_tokens(text),
                    embedding_text=embedding_text,
                )
            )
        return chunks
