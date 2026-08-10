"""Harvest golden-set query candidates from closed issues with a maintainer
resolution (plan.md Phase 4 step 1) — free labels: the question is the
issue title/body, the ground truth is already written by a maintainer.

`issue_to_document` (ingest phase) joined title + body + resolution into one
`Document.text` field for retrieval/embedding. Harvesting needs them apart
again — the resolution is a labeling *hint*, never itself the query or a
ground-truth chunk, so it must not leak into the text a human labeler is
shown as "the question".
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from aegis.eval.strata import classify_stratum
from aegis.types import DocType, Stratum

_RESOLUTION_SPLIT_RE = re.compile(r"\n\n\*\*Resolution \(@[^)]+\):\*\*\n\n", re.DOTALL)
_TITLE_PREFIX_RE = re.compile(r"^# .*?\n\n", re.DOTALL)

# Not every "resolution comment" (Phase 1's definition: most recent comment
# from a maintainer) is an actual answer -- some are pure redirects/closes
# with no technical content ("please open a new discussion", "closing this",
# a bare "thanks!"). Measured on the real corpus: 95/2,204 (4.3%) match this
# filter. Kept narrow (short length + an explicit redirect/close phrase, or
# extremely short regardless of phrase) so real terse-but-correct answers --
# common from this maintainer -- aren't discarded; the remainder is the
# human labeler's job to skip during confirmation, not this filter's.
_LOW_SIGNAL_PATTERNS = [
    r"please (open|create) a new (discussion|issue)",
    r"please (try|check|update) (again|with) (the )?(latest|newer|recent) version",
    r"not appreciated",
    r"closing this",
    r"duplicate of",
    r"please provide a (minimal|complete) (reproducible )?example",
    r"\bstale\b",
    r"not planned",
    r"wontfix|won.t fix",
    r"can.t reproduce|cannot reproduce",
]
_LOW_SIGNAL_RE = re.compile("|".join(_LOW_SIGNAL_PATTERNS), re.IGNORECASE)
_LOW_SIGNAL_MAX_WORDS_WITH_PHRASE = 30
_LOW_SIGNAL_MAX_WORDS_ANY = 6


def is_low_signal_resolution(resolution: str) -> bool:
    words = resolution.split()
    if len(words) <= _LOW_SIGNAL_MAX_WORDS_ANY:
        return True
    return len(words) <= _LOW_SIGNAL_MAX_WORDS_WITH_PHRASE and bool(_LOW_SIGNAL_RE.search(resolution))


def load_resolved_issues(documents_path: str | Path) -> list[dict]:
    """Parse `documents.jsonl`, keep issue docs with a resolution comment,
    split back into (issue_id, title, body, resolution)."""
    records = []
    with open(documents_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if doc["doc_type"] != DocType.ISSUE.value or not doc.get("resolution_comment_id"):
                continue
            parts = _RESOLUTION_SPLIT_RE.split(doc["text"], maxsplit=1)
            if len(parts) != 2:
                continue
            body_with_title, resolution = parts
            body = _TITLE_PREFIX_RE.sub("", body_with_title, count=1)
            records.append(
                {
                    "issue_id": doc["id"],
                    "title": doc["title"],
                    "body": body,
                    "resolution": resolution,
                }
            )
    return records


def harvest_candidates(documents_path: str | Path, target_per_stratum: int, seed: int) -> dict[Stratum, list[dict]]:
    """Classify every resolved issue, then randomly sample up to
    `target_per_stratum` per bucket (seeded, so re-running is reproducible —
    CLAUDE.md §6). A stratum with fewer real candidates than the target
    returns everything it has; the shortfall is a measurement, not silently
    padded."""
    records = [r for r in load_resolved_issues(documents_path) if not is_low_signal_resolution(r["resolution"])]
    by_stratum: dict[Stratum, list[dict]] = defaultdict(list)
    for record in records:
        stratum = classify_stratum(record["title"], record["body"], record["resolution"])
        record["stratum"] = stratum
        by_stratum[stratum].append(record)

    rng = random.Random(seed)
    sampled: dict[Stratum, list[dict]] = {}
    for stratum in Stratum:
        pool = list(by_stratum.get(stratum, []))
        rng.shuffle(pool)
        sampled[stratum] = pool[:target_per_stratum]
    return sampled
