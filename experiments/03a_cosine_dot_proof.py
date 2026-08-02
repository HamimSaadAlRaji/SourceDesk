"""20-minute numpy check (plan.md Phase 3): verify on real corpus vectors,
not toy examples, that cosine similarity reduces to a plain dot product once
embeddings are L2-normalized. Knowing this from a slide and proving it on
your own data are different states — this is the proof, with real numbers.
"""

from __future__ import annotations

import numpy as np

from aegis.config import settings

VECTORS_PATH = settings.paths.data_processed / "vectors_bge_small.npy"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    vectors = np.load(VECTORS_PATH)
    print(f"Loaded {vectors.shape[0]} real corpus vectors, dim={vectors.shape[1]} ({VECTORS_PATH.name})")

    norms = np.linalg.norm(vectors, axis=1)
    print(
        f"Norms: min={norms.min():.6f} max={norms.max():.6f} mean={norms.mean():.6f} "
        f"(sentence-transformers normalize_embeddings=True — should be ~1.0)"
    )

    rng = np.random.default_rng(20260730)
    n_pairs = 2000
    idx_a = rng.integers(0, vectors.shape[0], n_pairs)
    idx_b = rng.integers(0, vectors.shape[0], n_pairs)

    max_abs_diff = 0.0
    for i, j in zip(idx_a, idx_b):
        a, b = vectors[i], vectors[j]
        cos = cosine_similarity(a, b)
        dot = float(np.dot(a, b))
        max_abs_diff = max(max_abs_diff, abs(cos - dot))

    print(f"\n{n_pairs} random real chunk pairs: cosine-formula vs. plain dot product")
    print(f"Max absolute difference: {max_abs_diff:.2e}")
    print(f"Conclusion: {'EQUAL within float32 precision' if max_abs_diff < 1e-4 else 'NOT equal — investigate'}")

    # One concrete example, printed in full.
    a, b = vectors[idx_a[0]], vectors[idx_b[0]]
    print(f"\nWorked example (chunk indices {idx_a[0]} and {idx_b[0]}):")
    print(f"  ||a|| = {np.linalg.norm(a):.6f}, ||b|| = {np.linalg.norm(b):.6f}")
    print(f"  cosine_similarity(a, b) = dot(a,b) / (||a|| * ||b||) = {cosine_similarity(a, b):.8f}")
    print(f"  dot(a, b)                                            = {float(np.dot(a, b)):.8f}")


if __name__ == "__main__":
    main()
