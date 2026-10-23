"""Retrieval-layer partitioning (Phase 15, design record 15b).

A partition is fixed when a pipeline is built, never chosen per request:
`PartitionedRetriever.search` has no filter parameter at all, so no query
text, history, or request field can widen it. Two layers:

1. Push-down: the partition's doc_types go to the base retriever as a
   pre-filter (Qdrant `MatchAny` on the dense leg, a bm25s weight mask on the
   sparse leg), so recall inside the partition is unaffected.
2. Fail closed: every returned chunk is checked against the partition. A base
   retriever that ignored the filter -- exactly the hybrid BM25 bug found in
   design record 15 (A1) -- makes the request fail instead of leaking.

Layer 2 is what makes the guarantee independent of any one index's
correctness; layer 1 is what keeps layer 2 from ever firing in practice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from aegis.config import settings
from aegis.observability.logging import get_logger, log_event
from aegis.types import Chunk, DocType

_log = get_logger("tenancy")


class PartitionViolation(RuntimeError):
    """A base retriever returned a chunk outside the pipeline's partition."""


@dataclass(frozen=True)
class Partition:
    name: str
    doc_types: frozenset[DocType]


def resolve_partition(name: str) -> Partition:
    try:
        values = settings.tenancy.partitions[name]
    except KeyError:
        raise ValueError(f"unknown partition {name!r}; known: {sorted(settings.tenancy.partitions)}") from None
    return Partition(name=name, doc_types=frozenset(DocType(v) for v in values))


class _Retriever(Protocol):
    def search(self, query: str, top_k: int = 10, **kwargs) -> list[tuple[Chunk, float]]: ...


class PartitionedRetriever:
    def __init__(self, base: _Retriever, partition: Partition) -> None:
        self.base = base
        self.partition = partition

    def warm_up(self, text: str) -> None:
        warm = getattr(self.base, "warm_up", None)
        if warm is not None:
            warm(text)

    def search(self, query: str, top_k: int = 10, exclude_doc_id: str | None = None) -> list[tuple[Chunk, float]]:
        kwargs: dict = {"doc_types": self.partition.doc_types}
        if exclude_doc_id is not None:
            kwargs["exclude_doc_id"] = exclude_doc_id
        results = self.base.search(query, top_k=top_k, **kwargs)
        leaked = [c.id for c, _ in results if c.doc_type not in self.partition.doc_types]
        if leaked:
            log_event(_log, logging.ERROR, "tenancy.partition_violation",
                      partition=self.partition.name, leaked_chunk_ids=leaked[:10], n_leaked=len(leaked))
            raise PartitionViolation(f"{len(leaked)} chunk(s) outside partition {self.partition.name!r}")
        return results
