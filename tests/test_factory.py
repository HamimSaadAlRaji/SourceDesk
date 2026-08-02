import pytest

from aegis.chunking.factory import build_chunker
from aegis.chunking.fixed import FixedChunker
from aegis.chunking.heading_aware import HeadingAwareChunker
from aegis.chunking.recursive import RecursiveChunker
from aegis.chunking.semantic import SemanticChunker
from aegis.config import settings


def test_build_chunker_returns_correct_class_per_strategy():
    assert isinstance(build_chunker("fixed"), FixedChunker)
    assert isinstance(build_chunker("recursive"), RecursiveChunker)
    assert isinstance(build_chunker("heading_aware"), HeadingAwareChunker)
    assert isinstance(build_chunker("semantic"), SemanticChunker)


def test_build_chunker_uses_shared_config_values():
    chunker = build_chunker("fixed")
    assert chunker.max_tokens == settings.chunking.max_tokens
    assert chunker.overlap_tokens == settings.chunking.overlap_tokens


def test_build_chunker_defaults_enrichment_from_settings():
    chunker = build_chunker("fixed")
    assert chunker.contextual_enrichment == settings.chunking.contextual_enrichment


def test_build_chunker_enrichment_override():
    chunker = build_chunker("fixed", contextual_enrichment=True)
    assert chunker.contextual_enrichment is True


def test_build_chunker_unknown_strategy_raises():
    with pytest.raises(ValueError):
        build_chunker("not_a_real_strategy")
