"""Freeze the golden set (plan.md Phase 4 step 5) — combine positives +
negatives + labels, apply the holdout split, hash, write immutable files.

Once this runs, the golden set is locked: silently editing it invalidates
every comparison Phase 6+ makes. The hash in progress.md is the
proof-of-freeze, comparable across sessions.

Usage: `uv run python -m aegis.eval.freeze_golden_set`
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from aegis.config import settings
from aegis.eval.retrieve_candidates import load_jsonl
from aegis.eval.negatives import build_negative_queries
from aegis.types import GoldenQuery, Stratum


def _hash_file(path: Path) -> str:
    """SHA256 hash of a JSONL file, sorted by lines to ensure determinism
    regardless of internal ordering or write order."""
    lines = sorted(path.read_text(encoding="utf-8").strip().split("\n"))
    content = "\n".join(lines)
    return hashlib.sha256(content.encode()).hexdigest()


def main() -> None:
    positives_path = settings.paths.data_golden / "candidates_positive.jsonl"
    labels_path = settings.paths.data_golden / "labels_in_progress.jsonl"
    frozen_path = settings.paths.data_golden / "golden_set_v1.jsonl"
    holdout_path = settings.paths.data_golden / "golden_set_holdout_v1.jsonl"

    positives_by_id = {q["id"]: q for q in load_jsonl(positives_path)}
    labels = {r["query_id"]: r for r in load_jsonl(labels_path) if r["status"] == "accepted"}
    negatives = build_negative_queries()

    # Combine labeled positives with negatives
    queries_to_split: list[GoldenQuery] = []
    by_stratum: dict[str, list[dict]] = defaultdict(list)

    for query_id, label in labels.items():
        q = GoldenQuery(**positives_by_id[query_id])
        q.ground_truth_chunk_ids = label["ground_truth_chunk_ids"]
        queries_to_split.append(q)
        by_stratum[q.stratum.value].append(q)

    queries_to_split.extend(negatives)

    # Holdout split: 5 per stratum for positives (25 total), plus 10 negatives
    # (25% of the 40-query negative set) for balance. Deterministic via the
    # random seed.
    import random

    rng = random.Random(settings.golden.random_seed)
    dev_split = []
    holdout_split = []

    for stratum in Stratum:
        pool = by_stratum.get(stratum.value, [])
        rng.shuffle(pool)
        holdout_target = settings.golden.holdout_per_stratum
        holdout_from_stratum = pool[:holdout_target]
        dev_from_stratum = pool[holdout_target:]
        for q in holdout_from_stratum:
            q.split = "holdout"
            holdout_split.append(q)
        for q in dev_from_stratum:
            q.split = "dev"
            dev_split.append(q)

    # Holdout 10 negatives randomly
    negatives = [q for q in queries_to_split if q.is_negative]
    rng.shuffle(negatives)
    for q in negatives[:10]:
        q.split = "holdout"
        holdout_split.append(q)
    for q in negatives[10:]:
        q.split = "dev"
        dev_split.append(q)

    # Write frozen files (dev split only; holdout is sealed).
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    with open(frozen_path, "w", encoding="utf-8") as f:
        for q in dev_split:
            f.write(q.model_dump_json() + "\n")

    with open(holdout_path, "w", encoding="utf-8") as f:
        for q in holdout_split:
            f.write(q.model_dump_json() + "\n")

    # Hash and report
    dev_hash = _hash_file(frozen_path)
    holdout_hash = _hash_file(holdout_path)
    print(f"Froze {len(dev_split)} dev split queries, SHA256: {dev_hash[:16]}...")
    print(f"Sealed {len(holdout_split)} holdout queries, SHA256: {holdout_hash[:16]}...")
    print(f"\nFull hashes for progress.md:")
    print(f"dev:     {dev_hash}")
    print(f"holdout: {holdout_hash}")


if __name__ == "__main__":
    main()
