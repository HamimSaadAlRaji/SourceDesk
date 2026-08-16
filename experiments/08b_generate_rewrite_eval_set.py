"""Builds the synthetic multi-turn eval set for query rewriting (plan.md
Phase 8, step 4). This corpus (FastAPI docs + closed GitHub issues) has no
real multi-turn conversation data, so there's no natural source of
(history, pronoun-laden follow-up) pairs the way the golden set itself came
free from real issues.

Decided with the user (2026-08-12): an LLM generates the synthetic
conversation turns rather than a hand-authored set. For each sampled golden
query, the LLM is given the real question and asked to split it into a
short two-turn exchange -- a context-setting first turn plus a pronoun/
ellipsis-laden follow-up -- that captures the same underlying question.
Ground truth stays anchored to the *original* golden query's
`ground_truth_chunk_ids`, so scoring the follow-up (with vs without
rewriting) against that ground truth is a real, traceable measurement, not
an invented one.

Sample: 4 queries per stratum x 6 strata = 24, seeded from
`settings.golden.random_seed` for reproducibility. Output text is truncated
to 600 chars before generation -- grounds the LLM in the real question
without needing to digest an entire GitHub issue dump (some golden queries
run past 16,000 chars, see progress.md's length distribution check).
"""

from __future__ import annotations

import json
import random

from aegis.config import REPO_ROOT, settings
from aegis.types import GoldenQuery, Stratum

OUT_PATH = REPO_ROOT / "data" / "golden" / "query_rewrite_eval_v1.jsonl"
N_PER_STRATUM = 4
MAX_INPUT_CHARS = 600

SYSTEM_PROMPT = """You turn a single real user question about FastAPI into a \
short two-turn conversation:

- turn1: a brief message (1 sentence) that sets context by naming the \
general topic or feature, but withholds the specific detail the real \
question asks about.
- turn2: a short follow-up (1 sentence) that asks for that specific detail \
using a pronoun ("it", "that", "this") or ellipsis (an omitted subject) so \
it only makes sense combined with turn1. turn2 together with turn1 must \
capture the same underlying question as the original -- don't add or drop \
requirements.

Respond with JSON only: {"turn1": "...", "turn2": "..."}"""


def load_sample() -> list[GoldenQuery]:
    path = settings.paths.data_golden / "golden_set_v1.jsonl"
    rows = [GoldenQuery(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]
    positives = [q for q in rows if not q.is_negative]

    rng = random.Random(settings.golden.random_seed)
    sample = []
    for stratum in Stratum:
        pool = sorted((q for q in positives if q.stratum == stratum), key=lambda q: q.id)
        sample.extend(rng.sample(pool, min(N_PER_STRATUM, len(pool))))
    return sample


def generate_turns(client, query_text: str) -> dict | None:
    from groq import APIError

    try:
        completion = client.chat.completions.create(
            model=settings.query.rewrite_model,
            temperature=0.7,  # some variety in phrasing is fine/wanted here -- this is eval-fixture generation, not the rewriter itself
            max_tokens=400,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query_text[:MAX_INPUT_CHARS]},
            ],
        )
    except APIError as e:
        print(f"  generation failed: {e}")
        return None

    raw = completion.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
        turn1, turn2 = parsed["turn1"].strip(), parsed["turn2"].strip()
    except (json.JSONDecodeError, KeyError, AttributeError):
        print(f"  malformed output: {raw!r}")
        return None
    if not turn1 or not turn2:
        return None
    return {"turn1": turn1, "turn2": turn2}


def main() -> None:
    from groq import Groq

    client = Groq(api_key=settings.groq.api_key)
    sample = load_sample()
    print(f"Sampled {len(sample)} golden queries ({N_PER_STRATUM}/stratum)")

    rows = []
    for q in sample:
        turns = generate_turns(client, q.text)
        if turns is None:
            print(f"  SKIP {q.id} (generation failed)")
            continue
        rows.append(
            {
                "id": f"rewrite_eval::{q.id}",
                "original_query_id": q.id,
                "stratum": q.stratum.value,
                "original_text_preview": q.text[:200],
                "turn1": turns["turn1"],
                "turn2": turns["turn2"],
                "ground_truth_chunk_ids": q.ground_truth_chunk_ids,
                "source_issue": q.source_issue,
            }
        )
        print(f"  {q.id} ({q.stratum.value}):\n    turn1: {turns['turn1']}\n    turn2: {turns['turn2']}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(rows)} synthetic conversations to {OUT_PATH}")


if __name__ == "__main__":
    main()
