"""Content hashing — the foundation of Phase 12's delta indexing.

Cheap to add now, expensive to retrofit: without a hash per document,
detecting "this doc changed since last ingestion" means re-embedding
everything, every time.
"""

import hashlib


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
