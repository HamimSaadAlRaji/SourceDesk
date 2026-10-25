"""Run this on Kaggle (GPU notebook), not locally.

Companion to `experiments/16e_export_rerank_pairs.py`. It scores every exported
(query, chunk) pair with the project's cross-encoder and writes the score cache the
CI escalation check reads (`ci/caches/rerank_scores.json`). Self-contained: no `aegis`
import, no corpus, no Qdrant -- only pip-installable packages.

--- How to run (this is a MANUAL handoff: nothing here runs automatically) ---
1. https://www.kaggle.com/code -> New Notebook.
2. Settings (right sidebar) -> Accelerator -> GPU T4 x2 (or any GPU).
3. Add Input -> Upload -> `rerank_pairs_export.json`
   (from `ci-fixtures/` in this repo, produced by 16e). Note the mounted path Kaggle
   shows (usually `/kaggle/input/<dataset-slug>/rerank_pairs_export.json`) and update
   EXPORT_PATH below if it differs.
4. Paste this whole file into one cell and run it. Takes a few minutes on a T4.
5. Download `rerank_scores.json` from the notebook's Output panel and put it at
   `ci/caches/rerank_scores.json` in the repo. Also copy back the printed
   "max |kaggle - local|" line: it is the check that GPU scores match this machine's.
"""

from __future__ import annotations

import json
import time

EXPORT_PATH = "/kaggle/input/rerank-pairs/rerank_pairs_export.json"
OUTPUT_PATH = "/kaggle/working/rerank_scores.json"
# 16 forces internal chunking: Phase 7 hit a CUDA OOM on a T4 scoring long chunks in one big batch.
BATCH_SIZE = 16


def main() -> None:
    import sentence_transformers
    import torch
    from sentence_transformers import CrossEncoder

    with open(EXPORT_PATH, encoding="utf-8") as f:
        export = json.load(f)
    model_name, pairs, reference = export["model_name"], export["pairs"], export["reference"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(pairs)} pairs, model {model_name}, device {device}")
    if device == "cpu":
        print("WARNING: no GPU; this will be slow. Enable an accelerator in Settings.")

    model = CrossEncoder(model_name, device=device)
    t0 = time.time()
    scores = [float(s) for s in model.predict([(p["query"], p["chunk_text"]) for p in pairs],
                                              batch_size=BATCH_SIZE, show_progress_bar=True)]
    print(f"scored {len(pairs)} pairs in {time.time() - t0:.0f}s")
    by_key = {p["key"]: s for p, s in zip(pairs, scores)}

    # Determinism: rescore a few pairs; GPU kernels should agree with themselves to float noise.
    probe = pairs[:8]
    again = [float(s) for s in model.predict([(p["query"], p["chunk_text"]) for p in probe], batch_size=BATCH_SIZE)]
    print("max |run1 - run2| on 8 pairs:", max(abs(by_key[p["key"]] - a) for p, a in zip(probe, again)))

    diffs = [abs(by_key[k] - v) for k, v in reference.items() if k in by_key]
    print(f"max |kaggle - local| over {len(diffs)} reference pairs: {max(diffs):.6f}   <-- copy this back")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {"model_name": model_name, "device": device, "torch": torch.__version__,
                     "sentence_transformers": sentence_transformers.__version__, "batch_size": str(BATCH_SIZE),
                     "n_pairs": str(len(pairs))},
            "scores": by_key,
        }, f, indent=1, sort_keys=True)
    print("wrote", OUTPUT_PATH)


main()
