"""Phase 15 unit 15f: how much of a chunk's content leaks from its stored
vector alone? (Design record 15f, "vector reconstruction".) Zero LLM calls;
bge-large on CPU (~22 s benchmarked for the vocabulary).

Threat model: an attacker who can read Qdrant but whose dump has NO payload
text -- only the 1,024-d vectors -- and who knows (or guesses) the public
embedding model. Crude attack, deliberately: embed a vocabulary, rank words
by cosine to each stolen vector, take the top 10. Not a trained inversion
model (vec2text-style attacks recover far more); this is a lower bound.

Two scorings, both reported: raw cosine, and hubness-corrected cosine (each
word's mean similarity over 2,000 other stolen vectors subtracted, CSLS-style)
-- added after the raw scoring lost to the frequency baseline because a few
words ("apirouter", "openapi") top almost every vector (design record 15f §6).

Metric: precision@10 = fraction of the top-10 guessed words that occur in
the hidden chunk. Baselines: 10 random vocabulary words; the 10 most
frequent vocabulary words (what an attacker gets for free without vectors).

    uv run python experiments/15f_vector_leakage.py
"""

from __future__ import annotations

import json
import random
import re
import statistics
import time
from collections import Counter

import numpy as np

from aegis.config import REPO_ROOT, settings
from aegis.embedding.encoder import Encoder
from aegis.embedding.vector_store import DENSE_VECTOR_NAME, get_client
from aegis.factory import load_chunks

OUT = REPO_ROOT / "docs" / "results" / "vector-leakage-raw.json"
VOCAB_SIZE, N_CHUNKS, K, SEED = 3000, 100, 10, 15
_WORD = re.compile(r"[a-z][a-z_]{3,}")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def main() -> None:
    t0 = time.perf_counter()
    rnd = random.Random(SEED)
    chunks = load_chunks()
    df = Counter(w for c in chunks for w in _words(c.text))
    n = len(chunks)
    # Content words: in >= 20 chunks (not typos/ids), in <= 20% of chunks (not "this", "that").
    band = [w for w, d in df.most_common() if 20 <= d <= 0.2 * n]
    vocab = band[:VOCAB_SIZE]
    enc = Encoder(model_name=settings.embedding.bge_large_model, cache=None, device="cpu")
    V = np.asarray(enc.encode(vocab), dtype=np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    sample = rnd.sample([c for c in chunks if len(_words(c.text) & set(vocab)) >= 5], N_CHUNKS)
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    # The Phase 3 collection predates uuid5 point ids, so look points up by the
    # chunk_id payload field. Only chunk_id is read from the payload -- the
    # attacker model has vectors, not text.
    points, _ = get_client().scroll(
        settings.qdrant.collection_bge_large, limit=N_CHUNKS * 2, with_vectors=True, with_payload=["chunk_id"],
        scroll_filter=Filter(must=[FieldCondition(key="chunk_id", match=MatchAny(any=[c.id for c in sample]))]),
    )
    by_id = {p.payload["chunk_id"]: np.asarray(p.vector[DENSE_VECTOR_NAME] if isinstance(p.vector, dict) else p.vector)
             for p in points}
    # Hubness correction: a word's average closeness to *all* chunk vectors.
    bg, _ = get_client().scroll(settings.qdrant.collection_bge_large, limit=2000, with_vectors=True, with_payload=False)
    B = np.asarray([p.vector[DENSE_VECTOR_NAME] if isinstance(p.vector, dict) else p.vector for p in bg], dtype=np.float32)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    hub = (V @ B.T).mean(axis=1)
    frequent = vocab[:K]
    rows = []
    for c in sample:
        v = by_id[c.id]
        v = v / np.linalg.norm(v)
        sims = V @ v
        top = [vocab[i] for i in np.argsort(-sims)[:K]]
        top_corrected = [vocab[i] for i in np.argsort(-(sims - hub))[:K]]
        truth = _words(c.text)
        rows.append({"chunk_id": c.id, "doc_type": c.doc_type.value,
                     "p_at_10_vector": sum(w in truth for w in top) / K,
                     "p_at_10_vector_hub_corrected": sum(w in truth for w in top_corrected) / K,
                     "top_words_hub_corrected": top_corrected,
                     "p_at_10_random": sum(w in truth for w in rnd.sample(vocab, K)) / K,
                     "p_at_10_frequent": sum(w in truth for w in frequent) / K,
                     "top_words": top})
    mean = lambda key: round(statistics.mean(r[key] for r in rows), 4)
    out = {"vocab_size": len(vocab), "n_chunks": len(rows), "k": K, "seed": SEED,
           "precision_at_10": {"from_vector": mean("p_at_10_vector"),
                               "from_vector_hub_corrected": mean("p_at_10_vector_hub_corrected"),
                               "random_words": mean("p_at_10_random"),
                               "most_frequent_words": mean("p_at_10_frequent")},
           "chunks_with_half_or_more_correct": sum(r["p_at_10_vector"] >= 0.5 for r in rows),
           "chunks_with_half_or_more_correct_hub_corrected": sum(r["p_at_10_vector_hub_corrected"] >= 0.5 for r in rows),
           "examples": rows[:5], "elapsed_s": round(time.perf_counter() - t0, 1)}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=2))
    for r in rows[:3]:
        print(r["chunk_id"], r["p_at_10_vector_hub_corrected"], r["top_words_hub_corrected"])


if __name__ == "__main__":
    main()
