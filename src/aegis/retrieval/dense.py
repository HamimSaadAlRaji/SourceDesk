"""Dense retrieval over Qdrant, with metadata prefiltering (plan.md Phase 5,
step 3).

I/O-bound wrapper over already-built Phase 3 components (`Encoder`, the
Qdrant client) — integration-tested by actually running it against the
live collection, per CLAUDE.md §6, the same treatment as
`aegis.embedding.run` and `aegis.eval.retrieve_candidates`.

Filtering is native Qdrant pre-filtering, not a post-hoc filter on returned
results: Phase 3 measured post-filtering losing 60-83% recall on this
corpus even for `product_area=general` (26% of the corpus) — see
`docs/results/prefilter-vs-postfilter.md`. Doing it any other way here
would silently reintroduce a bug that's already been measured and fixed.

Returns `list[tuple[Chunk, float]]` — the same shape as
`aegis.retrieval.sparse.SparseIndex.search` — so `hybrid.py` can fuse
dense and sparse results without knowing which is which (Phase 5 exit
criterion: "All retrieval modes behind one interface").
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from aegis.embedding.encoder import Encoder
from aegis.embedding.vector_store import DENSE_VECTOR_NAME, get_client
from aegis.types import Chunk, DocType


class DenseIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        collection_name: str,
        encoder: Encoder,
        client: QdrantClient | None = None,
    ) -> None:
        # Qdrant's payload is a display/filter cache, not the source of
        # truth for chunk content (it doesn't carry `content_hash` or
        # `embedding_text` — see docs/explainers/05-lexical-dense-and-
        # fusion.md §6). Real `Chunk` objects are looked up by ID from
        # what the caller already has in memory, the same corpus that was
        # indexed, so dense results carry full fidelity like sparse ones.
        self._chunks_by_id = {chunk.id: chunk for chunk in chunks}
        self.collection_name = collection_name
        self.encoder = encoder
        self.client = client or get_client()

    def warm_up(self, text: str) -> None:
        """Load the query encoder's model (see `AnswerPipeline.warm_up`)."""
        warm = getattr(self.encoder, "warm_up", None)
        if warm is not None:
            warm(text)

    def search(
        self,
        query: str,
        top_k: int = 10,
        product_area: str | None = None,
        doc_type: DocType | None = None,
        exclude_doc_id: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        query_vector = self.encoder.encode([query])[0]

        must: list[FieldCondition] = []
        if product_area is not None:
            must.append(FieldCondition(key="product_area", match=MatchValue(value=product_area)))
        if doc_type is not None:
            must.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type.value)))
        # Native must_not, not a post-hoc filter -- same pre-filter-over-
        # post-filter reasoning as product_area/doc_type above. Needed for
        # eval: a golden query's own source issue (aegis.eval.
        # retrieve_candidates uses the identical must_not pattern at
        # labeling time) is a trivial near-verbatim match that would
        # otherwise waste a top-k slot on a chunk ground truth deliberately
        # excludes.
        must_not = [FieldCondition(key="doc_id", match=MatchValue(value=exclude_doc_id))] if exclude_doc_id else []
        query_filter = Filter(must=must, must_not=must_not) if (must or must_not) else None

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            query_filter=query_filter,
        ).points

        results: list[tuple[Chunk, float]] = []
        for hit in hits:
            chunk = self._chunks_by_id.get(hit.payload["chunk_id"])
            if chunk is not None:
                results.append((chunk, float(hit.score)))
        return results
