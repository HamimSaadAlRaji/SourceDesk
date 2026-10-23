"""Phase 0 naive RAG baseline — deliberately bad, on purpose.

500-char fixed chunks, dense-only retrieval, top-3, stuffed into a prompt,
no citations, no abstention. This is the Air Canada architecture: a
confident answer with no way to check it, and no way for the system itself
to know when it's wrong. Every later phase exists to fix one piece of this.

Not reused past Phase 0 — Phase 1 replaces the ingestion with structure-aware
parsing, Phase 2 replaces the chunker, Phase 9-10 add citations and
abstention. This file stays as a record of what "bad" looked like.
"""

from __future__ import annotations

from pathlib import Path

from fastembed import TextEmbedding
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from aegis.config import settings

FASTAPI_DOCS = settings.paths.data_raw / "fastapi" / "docs" / "en" / "docs"
DOC_SUBDIRS = ["tutorial", "advanced"]


def load_raw_text() -> list[tuple[str, str]]:
    """(doc_id, raw_text) pairs. No markdown parsing, no heading awareness —
    that structure-preservation is exactly what Phase 1 adds and Phase 0
    doesn't have."""
    docs = []
    for subdir in DOC_SUBDIRS:
        for path in sorted((FASTAPI_DOCS / subdir).rglob("*.md")):
            doc_id = str(path.relative_to(FASTAPI_DOCS))
            docs.append((doc_id, path.read_text(encoding="utf-8")))
    return docs


def fixed_chunks(text: str, size: int) -> list[str]:
    """Naive fixed-size character windows, no overlap, no boundary
    awareness. Will slice through code blocks and mid-sentence on purpose."""
    return [text[i : i + size] for i in range(0, len(text), size) if text[i : i + size].strip()]


def build_index(client: QdrantClient, embedder: TextEmbedding) -> list[str]:
    """Chunk every doc, embed, upsert into Qdrant. Returns chunk texts in
    point-id order for later lookup."""
    docs = load_raw_text()
    chunk_texts: list[str] = []
    payloads: list[dict] = []
    for doc_id, text in docs:
        for chunk in fixed_chunks(text, settings.chunking.baseline_chunk_chars):
            chunk_texts.append(chunk)
            payloads.append({"doc_id": doc_id, "text": chunk})

    vectors = list(embedder.embed(chunk_texts))
    dim = len(vectors[0])

    client.recreate_collection(
        collection_name=settings.qdrant.collection_baseline,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    client.upload_points(
        collection_name=settings.qdrant.collection_baseline,
        points=[
            PointStruct(id=i, vector=vec.tolist(), payload=payload)
            for i, (vec, payload) in enumerate(zip(vectors, payloads))
        ],
    )
    print(f"Indexed {len(chunk_texts)} chunks from {len(docs)} docs into Qdrant.")
    return chunk_texts


NAIVE_SYSTEM_PROMPT = """You are a helpful assistant answering questions about FastAPI. \
Use the context below to answer the user's question directly and confidently."""


def answer(query: str, client: QdrantClient, embedder: TextEmbedding, groq: Groq) -> tuple[str, list[dict]]:
    """Dense-only top-k, stuffed into a prompt. No citations, no confidence
    check, no abstention path — whatever the LLM says goes straight to the
    user."""
    query_vec = next(embedder.embed([query]))
    hits = client.query_points(
        collection_name=settings.qdrant.collection_baseline,
        query=query_vec.tolist(),
        limit=settings.retrieval.baseline_top_k,
    ).points

    context = "\n\n---\n\n".join(h.payload["text"] for h in hits)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    completion = groq.chat.completions.create(
        model=settings.groq.model,
        temperature=settings.groq.temperature,
        max_tokens=settings.groq.max_tokens,
        messages=[
            {"role": "system", "content": NAIVE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    retrieved = [{"doc_id": h.payload["doc_id"], "score": h.score, "text": h.payload["text"]} for h in hits]
    return completion.choices[0].message.content, retrieved


QUESTIONS = [
    "Does FastAPI have built-in rate limiting, and if so what's the default limit?",
    "What is the default request timeout in FastAPI?",
    "How do you define a path parameter with a regex constraint in FastAPI?",
    "What decorator should I use today to run code when the app shuts down?",
    "What's the maximum number of path operations allowed in a single FastAPI app?",
]


def main() -> None:
    import json

    from aegis.embedding.vector_store import get_client

    client = get_client()
    embedder = TextEmbedding(model_name=settings.embedding.baseline_model)
    groq = Groq(api_key=settings.groq.api_key)

    build_index(client, embedder)

    transcript = []
    for q in QUESTIONS:
        ans, retrieved = answer(q, client, embedder, groq)
        transcript.append({"question": q, "answer": ans, "retrieved": retrieved})
        print(f"\nQ: {q}\nA: {ans}\n")

    out_path = Path(__file__).resolve().parents[2] / "docs" / "results" / "00-baseline-transcript.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(transcript, indent=2))
    print(f"\nTranscript saved to {out_path}")

    return transcript


if __name__ == "__main__":
    main()
