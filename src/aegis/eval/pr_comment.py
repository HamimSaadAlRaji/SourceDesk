"""Build the sticky PR comment for the evaluation gate (plan.md Phase 16 step 5,
design record 16f).

Standard library only, on purpose: the `report` job runs this by path on a bare
runner, without installing the project's dependencies, so a dependency problem
can never stop the failing gate's numbers from reaching the PR.

    python3 src/aegis/eval/pr_comment.py --sha $SHA --run-url $URL \\
        --retrieval-result success --router-result skipped --generation-result skipped \\
        --escalation-result success --escalation-md results-esc/escalation.md \\
        --retrieval-md results-retrieval/retrieval.md --router-md results-llm/router.md \\
        --generation-md results-gen/generation.md \\
        --spend-db results-llm/ci_spend.sqlite --spend-db results-gen/ci_spend.sqlite --out comment.md

The comment says what happened, not what is hoped: a gate that was skipped,
cancelled or died before writing its table is reported as exactly that.
"""

from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

MARKER = "<!-- aegis-eval-gate -->"  # the workflow finds and updates the comment by this line
_GATES = (("retrieval", "Retrieval"), ("router", "Router"), ("generation", "Generation"), ("escalation", "Escalation"))
_SECTION_LIMIT = 25_000  # GitHub rejects comments over 65,536 characters
_BODY_LIMIT = 60_000


def read_spend(db_path: Path) -> float | None:
    """Total cost recorded in a run's spend ledger, or None if there is no ledger."""
    if not db_path.exists():
        return None
    with closing(sqlite3.connect(db_path)) as con:
        row = con.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM spend").fetchone()
    return float(row[0])


def total_spend(db_paths: list[Path]) -> float | None:
    """Sum over every ledger that exists (each LLM job keeps its own); None if none do."""
    found = [spent for path in db_paths if (spent := read_spend(path)) is not None]
    return sum(found) if found else None


def _headline(jobs: dict[str, str]) -> str:
    names = dict(_GATES)
    failed = [names[k] for k, _ in _GATES if jobs.get(k) == "failure"]
    cancelled = [names[k] for k, _ in _GATES if jobs.get(k) == "cancelled"]
    not_run = [k for k, _ in _GATES if jobs.get(k) in (None, "skipped")]
    if failed:
        return f"## ❌ Evaluation gate failed: {', '.join(failed)}"
    if cancelled:
        return f"## ⚠️ Evaluation gate incomplete: {', '.join(cancelled)} cancelled"
    if not_run:
        plural = "gate" if len(not_run) == 1 else "gates"
        return f"## ✅ No gate failed ({len(not_run)} {plural} not run)"
    return "## ✅ All evaluation gates passed"


def _section(key: str, label: str, result: str | None, markdown: str | None) -> str:
    if markdown:
        text = markdown.strip()
        if len(text) > _SECTION_LIMIT:
            text = text[:_SECTION_LIMIT] + "\n\n_… truncated; see the run's job summary and artifacts._"
        return text
    if result in (None, "skipped"):
        return f"### {label}\n\n_Not run (skipped for this event; e.g. a fork PR has no secrets for the LLM gate)._"
    return f"### {label}\n\n_Gate {result}; no result was produced (see the run log)._"


def build_comment(
    *, sha: str, run_url: str, jobs: dict[str, str], sections: dict[str, str | None], spend_usd: float | None
) -> str:
    parts = [f"{MARKER}\n{_headline(jobs)}"]
    for key, label in _GATES:
        parts.append(_section(key, label, jobs.get(key), sections.get(key)))
    footer = f"Commit `{sha[:7]}` · [workflow run]({run_url})"
    if spend_usd is not None:
        footer += f" · OpenAI spend this run: ${spend_usd:.4f}"
    parts.append(f"<sub>{footer}</sub>")
    body = "\n\n".join(parts)
    if len(body) > _BODY_LIMIT:
        body = body[:_BODY_LIMIT] + "\n\n_… comment truncated._"
    return body + "\n"


def _read(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--retrieval-result", required=True)
    parser.add_argument("--router-result", required=True)
    parser.add_argument("--generation-result", required=True)
    parser.add_argument("--escalation-result", required=True)
    parser.add_argument("--retrieval-md")
    parser.add_argument("--router-md")
    parser.add_argument("--generation-md")
    parser.add_argument("--escalation-md")
    parser.add_argument("--spend-db", action="append", default=[], help="repeatable: one ledger per LLM job")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    body = build_comment(
        sha=args.sha, run_url=args.run_url,
        jobs={"retrieval": args.retrieval_result, "router": args.router_result, "generation": args.generation_result,
              "escalation": args.escalation_result},
        sections={"retrieval": _read(args.retrieval_md), "router": _read(args.router_md),
                  "generation": _read(args.generation_md), "escalation": _read(args.escalation_md)},
        spend_usd=total_spend([Path(p) for p in args.spend_db]),
    )
    args.out.write_text(body, encoding="utf-8")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
