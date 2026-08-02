from aegis.config import settings
from aegis.types import Chunk, DocType


def test_settings_load():
    assert settings.retrieval.baseline_top_k == 3


def test_chunk_roundtrip():
    chunk = Chunk(
        id="c1",
        doc_id="d1",
        text="hello",
        doc_type=DocType.DOCUMENTATION,
        content_hash="abc",
        token_count=1,
        embedding_text="hello",
    )
    assert chunk.doc_type is DocType.DOCUMENTATION
