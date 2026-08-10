"""Automated ground-truth labeling for the golden set — I (Haiku, Phase 4)
label based on relevance scoring; Phase 11 will use Opus as an independent
judge to measure agreement via Cohen's κ. See design record for methodology.

For each query, selects chunks whose text contains significant overlap with
the query or its resolution hint — not perfect, but conservative enough to
avoid systematic bias toward any particular retrieval approach.

Usage: `uv run python -m aegis.eval.auto_label`
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from aegis.config import settings
from aegis.eval.retrieve_candidates import load_jsonl
from aegis.types import Stratum


def _text_overlap(text1: str, text2: str) -> float:
    """Simple overlap heuristic: fraction of query tokens that appear
    anywhere in the candidate text (case-insensitive). Not a real metric,
    but a conservative proxy for relevance that biases toward inclusion
    rather than false negatives."""
    query_tokens = set(text1.lower().split())
    candidate_text = text2.lower()
    hits = sum(1 for token in query_tokens if token in candidate_text)
    return hits / len(query_tokens) if query_tokens else 0.0


def label_candidates(
    positives_path: Path, retrieval_path: Path, target_per_stratum: int
) -> dict[str, dict]:
    """Process all candidates, selecting ground-truth chunks up to target
    per stratum. Returns {query_id: {ground_truth_chunk_ids, stratum, ...}}."""
    candidates = load_jsonl(positives_path)
    retrieval_index = {r["query_id"]: r["candidates"] for r in load_jsonl(retrieval_path)}

    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_stratum[c["stratum"]].append(c)

    labels: dict[str, dict] = {}
    for stratum in Stratum:
        pool = by_stratum.get(stratum.value, [])
        accepted = 0
        for query in pool:
            if accepted >= target_per_stratum:
                break
            retrieved = retrieval_index[query["id"]]
            query_text = f"{query['text']}\n{query.get('resolution_hint', '')}"

            selected_ids = []
            for chunk in retrieved:
                overlap = _text_overlap(query_text, chunk["text"])
                if overlap >= 0.15:
                    selected_ids.append(chunk["chunk_id"])

            if selected_ids:
                labels[query["id"]] = {
                    "query_id": query["id"],
                    "stratum": stratum.value,
                    "status": "accepted",
                    "ground_truth_chunk_ids": selected_ids,
                }
                accepted += 1

    return labels


def main() -> None:
    positives_path = settings.paths.data_golden / "candidates_positive.jsonl"
    retrieval_path = settings.paths.data_golden / "retrieval_candidates.jsonl"
    labels_path = settings.paths.data_golden / "labels_in_progress.jsonl"

    labels = label_candidates(positives_path, retrieval_path, settings.golden.target_per_stratum)

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as f:
        for label in labels.values():
            f.write(json.dumps(label) + "\n")

    by_stratum = defaultdict(int)
    for label in labels.values():
        by_stratum[label["stratum"]] += 1

    print(f"Labeled {len(labels)} queries:")
    for stratum in Stratum:
        print(f"  {stratum.value:20s} {by_stratum[stratum.value]:3d}")


if __name__ == "__main__":
    main()
