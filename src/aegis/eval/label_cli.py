"""Resumable terminal tool for the human ground-truth labeling step (plan.md
Phase 4 step 2: "you confirm which chunks actually contain the answer").

Deliberately no LLM assistance anywhere in this file — Phase 11 later
measures an LLM-judge's agreement against these exact labels (Cohen's
kappa), and that comparison is only meaningful against a golden set with
zero LLM fingerprints on it.

For each candidate query (already stratified and retrieved), shows the
query, the maintainer's resolution as a non-authoritative hint, and 20
permissively-retrieved candidate chunks (12 documentation + 8 issue, see
GoldenSettings). You either select the chunk(s) that actually answer the
query, skip the query outright (no good answer in this pool, or the issue
turned out not to be a real question), or fall back to a keyword search
over the full chunk corpus when the true chunk isn't in the retrieved 20.

Every decision is appended immediately to `labels_in_progress.jsonl`, so
`Ctrl+C` or 'q' loses nothing — rerunning the script picks up exactly where
you left off. A stratum stops prompting once it has `target_per_stratum`
accepted queries; the rest of its buffer is left unlabeled on purpose.

Usage: `uv run python -m aegis.eval.label_cli`
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from aegis.config import settings
from aegis.eval.retrieve_candidates import load_jsonl
from aegis.types import Stratum

_PREVIEW_LEN = 200


def _preview(text: str, length: int = _PREVIEW_LEN) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= length else flat[:length] + "..."


def load_existing_labels(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["query_id"]: r for r in load_jsonl(path)}


def append_label(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_all_chunks(path: Path) -> list[dict]:
    return load_jsonl(path)


def _print_candidate(query: dict, retrieved: list[dict], accepted_so_far: int, target: int) -> None:
    print("\n" + "=" * 88)
    print(f"[{query['stratum']}] accepted {accepted_so_far}/{target}  |  {query['id']}  |  source {query['source_issue']}")
    print("-" * 88)
    print("QUERY:")
    print(_preview(query["text"], 600))
    print("\nRESOLUTION HINT (not authoritative — a maintainer's reply, may itself be wrong or off-topic):")
    print(_preview(query["resolution_hint"], 300))
    print("-" * 88)
    for i, c in enumerate(retrieved, start=1):
        heading = " > ".join(c["heading_path"]) if c["heading_path"] else c["doc_id"]
        print(f"{i:2d}. [{c['doc_type']:13s} score={c['score']:.3f}] {heading}")
        print(f"    {_preview(c['text'])}")


def _manual_search(chunks: list[dict], query: str) -> list[dict]:
    needle = query.lower()
    matches = [c for c in chunks if needle in c["text"].lower() or needle in " ".join(c["heading_path"]).lower()]
    return matches[:15]


def _prompt(query: dict, retrieved: list[dict], chunks: list[dict], accepted_so_far: int, target: int) -> dict | str:
    """Returns {'ground_truth_chunk_ids': [...]} to accept, or 'skip'/'quit'."""
    _print_candidate(query, retrieved, accepted_so_far, target)
    while True:
        raw = input("\n> chunk #s to accept | 's' skip | 'm' manual search | 'q' save & quit: ").strip().lower()
        if raw == "q":
            return "quit"
        if raw == "s":
            return "skip"
        if raw == "m":
            keyword = input("  search keyword: ").strip()
            if not keyword:
                continue
            found = _manual_search(chunks, keyword)
            if not found:
                print("  no matches.")
                continue
            for i, c in enumerate(found, start=1):
                heading = " > ".join(c["heading_path"]) if c["heading_path"] else c["doc_id"]
                print(f"  {i:2d}. [{c['doc_type']:13s}] {heading}")
                print(f"      {_preview(c['text'])}")
            picked = input("  chunk #s to accept from these results (blank to cancel): ").strip()
            indices = [int(x) for x in re.findall(r"\d+", picked)]
            valid = [found[i - 1]["id"] for i in indices if 1 <= i <= len(found)]
            if valid:
                return {"ground_truth_chunk_ids": valid}
            continue
        indices = [int(x) for x in re.findall(r"\d+", raw)]
        valid = [retrieved[i - 1]["chunk_id"] for i in indices if 1 <= i <= len(retrieved)]
        if valid:
            return {"ground_truth_chunk_ids": valid}
        print("  no valid selection understood — try again.")


def main() -> None:
    positives_path = settings.paths.data_golden / "candidates_positive.jsonl"
    retrieval_path = settings.paths.data_golden / "retrieval_candidates.jsonl"
    chunks_path = settings.paths.data_processed / "chunks.jsonl"
    labels_path = settings.paths.data_golden / "labels_in_progress.jsonl"

    candidates = load_jsonl(positives_path)
    retrieval_index = {r["query_id"]: r["candidates"] for r in load_jsonl(retrieval_path)}
    chunks = load_all_chunks(chunks_path)
    existing = load_existing_labels(labels_path)

    target = settings.golden.target_per_stratum
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_stratum[c["stratum"]].append(c)

    accepted_counts: Counter[str] = Counter()
    for rec in existing.values():
        if rec["status"] == "accepted":
            accepted_counts[rec["stratum"]] += 1

    print(f"Resuming: {len(existing)} candidates already decided ({sum(accepted_counts.values())} accepted).")

    quit_requested = False
    for stratum in Stratum:
        if quit_requested:
            break
        pool = by_stratum.get(stratum.value, [])
        for query in pool:
            if accepted_counts[stratum.value] >= target:
                break
            if query["id"] in existing:
                continue

            result = _prompt(query, retrieval_index[query["id"]], chunks, accepted_counts[stratum.value], target)
            if result == "quit":
                quit_requested = True
                break
            if result == "skip":
                append_label(labels_path, {"query_id": query["id"], "stratum": stratum.value, "status": "skipped", "ground_truth_chunk_ids": []})
                existing[query["id"]] = {"status": "skipped"}
                continue
            append_label(
                labels_path,
                {
                    "query_id": query["id"],
                    "stratum": stratum.value,
                    "status": "accepted",
                    "ground_truth_chunk_ids": result["ground_truth_chunk_ids"],
                },
            )
            existing[query["id"]] = {"status": "accepted"}
            accepted_counts[stratum.value] += 1

    print("\n" + "=" * 88)
    print("Session summary:")
    for stratum in Stratum:
        status = "OK" if accepted_counts[stratum.value] >= target else "SHORT"
        print(f"  {stratum.value:20s} {accepted_counts[stratum.value]:3d}/{target}  [{status}]")
    print("Rerun this command to resume where you left off.")


if __name__ == "__main__":
    main()
