"""Orchestrates plan.md Phase 4 steps 1+3+4: harvest positive candidates
from resolved issues, stratify, sample, and combine with the hand-authored
negative set. Writes two intermediate files under `data/golden/` — split so
the downstream retrieval + labeling steps (which only apply to positives)
don't have to filter negatives back out on every run.

Usage: `uv run python -m aegis.eval.build_candidates`
"""

from __future__ import annotations

from aegis.config import settings
from aegis.eval.harvest import harvest_candidates
from aegis.eval.negatives import build_negative_queries
from aegis.types import GoldenQuery, Stratum


def _record_to_golden_query(record: dict) -> GoldenQuery:
    text = f"{record['title']}\n\n{record['body']}".strip()
    return GoldenQuery(
        id=f"golden::{record['issue_id']}",
        text=text,
        is_negative=False,
        stratum=record["stratum"],
        source_issue=record["issue_id"],
        resolution_hint=record["resolution"].strip(),
    )


def write_jsonl(path, queries: list[GoldenQuery]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(q.model_dump_json() + "\n")


def main() -> None:
    documents_path = settings.paths.data_processed / "documents.jsonl"
    buffer_size = int(settings.golden.target_per_stratum * settings.golden.harvest_buffer_multiplier)
    sampled = harvest_candidates(
        documents_path,
        target_per_stratum=buffer_size,
        seed=settings.golden.random_seed,
    )

    positives = []
    for stratum in Stratum:
        records = sampled[stratum]
        print(f"{stratum.value:20s} {len(records)} sampled")
        positives.extend(_record_to_golden_query(r) for r in records)

    negatives = build_negative_queries()

    positives_path = settings.paths.data_golden / "candidates_positive.jsonl"
    negatives_path = settings.paths.data_golden / "candidates_negative.jsonl"
    write_jsonl(positives_path, positives)
    write_jsonl(negatives_path, negatives)

    print(f"\n{len(positives)} positive candidates -> {positives_path}")
    print(f"{len(negatives)} negatives -> {negatives_path}")


if __name__ == "__main__":
    main()
