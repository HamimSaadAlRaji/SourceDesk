"""Feedback -> review queue -> golden-set candidates (plan.md Phase 14 step 5).

A vote is not a label. 👎 says something went wrong, not what the right
answer or the right chunks are. So a person reviews each voted interaction
here and decides whether it becomes a positive candidate (with a stratum), a
negative candidate (with a category), or nothing. The export is a *candidate*
file next to the frozen golden set, never an edit to it.
`golden_set_v1.jsonl` stays frozen; a v2 would go through the same Phase 4
labeling (permissive retrieval pool -> human-picked chunks) and be frozen
under a new version.

Two choices protect future measurements:
- The system's own cited chunks are not exported as suggested labels.
  Phase 12 (design record 12e) found this golden set's labels are already
  biased toward the index that surfaced the candidates. Seeding new labels
  from the answering system's citations would make that worse, scoring the
  system against its own output.
- Each candidate is checked against every existing golden query, holdout
  included (normalized exact text), and flagged with `duplicate_of`.
  Promoting a holdout query into dev would quietly leak it.

Terminal tool, like `label_cli`: no review UI on the unauthenticated HTTP
surface (Phase 15 owns auth). Deliberately no LLM in the loop, for the same
reason as label_cli: Phase 11 measures an LLM judge against these labels.

    uv run python -m aegis.eval.review_queue            # review pending items
    uv run python -m aegis.eval.review_queue --export   # write the candidate file
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from pathlib import Path

from aegis.api.store import FeedbackRow, InteractionStore
from aegis.cache.exact_match import normalize_query
from aegis.config import settings
from aegis.eval.negatives import EXPECTED_BEHAVIOR
from aegis.eval.strata import classify_stratum
from aegis.types import GoldenQuery, NegativeCategory, Stratum

CANDIDATES_PATH = settings.paths.data_golden / "candidates_feedback.jsonl"
GOLDEN_PATHS = (
    settings.paths.data_golden / "golden_set_v1.jsonl",
    settings.paths.data_golden / "golden_set_holdout_v1.jsonl",
)
_PREVIEW = 600


def _standalone_text(query_text: str, rewritten_query: str | None) -> str:
    # A follow-up ("what about making it required?") means nothing without its
    # history; the rewritten text is what the system actually searched for.
    return rewritten_query or query_text


def _golden_index(paths: Iterable[Path]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    g = json.loads(line)
                    index.setdefault(normalize_query(g["text"]), {"id": g["id"], "split": g.get("split", "dev")})
    return index


def export_candidates(
    store: InteractionStore, path: Path = CANDIDATES_PATH, golden_paths: Iterable[Path] = GOLDEN_PATHS
) -> int:
    """Rewrite the candidate file from every promoted review. Deterministic
    (trace_id order) and idempotent, so it is safe to rerun after each session.
    Returns the number of candidates written."""
    golden = _golden_index(golden_paths)
    rows: list[dict] = []
    for r in store.reviewed():
        if r["decision"] == "reject":
            continue
        text = _standalone_text(r["query_text"], r["rewritten_query"])
        negative = r["decision"] == "negative"
        category = NegativeCategory(r["negative_category"]) if negative else None
        # Validates the core fields against the real golden-set schema.
        core = GoldenQuery(
            id=f"feedback::{r['trace_id']}",
            text=text,
            is_negative=negative,
            stratum=None if negative else Stratum(r["stratum"]),
            negative_category=category,
            expected_behavior=EXPECTED_BEHAVIOR[category] if category else None,
            split="candidate",
        )
        row = core.model_dump(mode="json", exclude_none=True)
        row["feedback"] = {
            "trace_id": r["trace_id"],
            "rating": r["rating"],
            "comment": r["comment"],
            "original_query": r["query_text"],
            "answer_shown": r["answer_text"],
            "escalated": bool(r["escalated"]),
            "cache_status": r["cache_status"],
            "asked_at": r["asked_at"],
            "review_note": r["note"],
            "reviewed_at": r["reviewed_at"],
        }
        dup = golden.get(normalize_query(text))
        if dup is not None:
            row["duplicate_of"] = dup
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _show(item: FeedbackRow, position: int, total: int, print_fn: Callable[[str], None]) -> None:
    print_fn("=" * 88)
    print_fn(f"[{position}/{total}] vote: {'👎 down' if item.rating == 'down' else '👍 up'}   trace: {item.trace_id}")
    if item.comment:
        print_fn(f"comment: {item.comment}")
    print_fn(f"asked: {item.query_text[:_PREVIEW]}")
    if item.rewritten_query:
        print_fn(f"searched for (rewritten): {item.rewritten_query[:_PREVIEW]}")
    print_fn(f"{'ESCALATED' if item.escalated else 'answer'} ({item.cache_status}): {item.answer_text[:_PREVIEW]}")


def _choose(prompt: str, options: list[str], default: str | None, input_fn: Callable[[str], str],
            print_fn: Callable[[str], None]) -> str:
    while True:
        raw = input_fn(prompt).strip().lower()
        if not raw and default is not None:
            return default
        if raw in options:
            return raw
        print_fn(f"  choose one of: {', '.join(options)}")


def review_session(
    store: InteractionStore,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> int:
    """Walk the pending queue. Every decision is written immediately, so quitting
    loses nothing. Returns the number of decisions recorded."""
    pending = store.pending_review()
    strata = [s.value for s in Stratum]
    categories = [c.value for c in NegativeCategory]
    decided = 0
    print_fn(f"{len(pending)} voted interaction(s) awaiting review.")
    for n, item in enumerate(pending, start=1):
        _show(item, n, len(pending), print_fn)
        action = _choose(
            "> p positive candidate | n negative candidate | r reject | s skip | q quit: ",
            ["p", "n", "r", "s", "q"], None, input_fn, print_fn,
        )
        if action == "q":
            break
        if action == "s":
            continue
        if action == "p":
            # Phase 4's harvest heuristic, shown but never accepted by default: on
            # query text alone (no issue resolution, as here) it agrees with the
            # frozen strata on 138/180 positives, so Enter-to-accept would
            # rubber-stamp a wrong stratum about 1 time in 4 (design record 14g).
            suggested = classify_stratum(_standalone_text(item.query_text, item.rewritten_query), "", "").value
            stratum = _choose(f"  stratum [{'/'.join(strata)}] (heuristic guess: {suggested}): ", strata, None,
                              input_fn, print_fn)
            store.record_review(item.trace_id, "positive", stratum=stratum)
        elif action == "n":
            category = _choose(f"  category [{'/'.join(categories)}]: ", categories, None, input_fn, print_fn)
            store.record_review(item.trace_id, "negative", negative_category=category)
        else:
            store.record_review(item.trace_id, "reject")
        decided += 1
    return decided


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--export", action="store_true", help="write the candidate file instead of reviewing")
    args = parser.parse_args()
    store = InteractionStore(settings.api.interactions_db_path)
    if args.export:
        n = export_candidates(store)
        dups = sum(1 for line in CANDIDATES_PATH.read_text().splitlines() if '"duplicate_of"' in line)
        print(f"{n} candidate(s) -> {CANDIDATES_PATH} ({dups} flagged as duplicates of existing golden queries)")
    else:
        print(f"{review_session(store)} decision(s) recorded. Run with --export to write candidates.")


if __name__ == "__main__":
    main()
