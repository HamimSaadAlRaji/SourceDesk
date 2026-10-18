"""Phase 11 rubric-fix follow-up (design record 11f): re-score Unit 4's real,
already-generated answers with the fixed `JUDGE_PROMPT_TEMPLATE`, without
re-running generation or RAGAS -- neither changed, so re-paying their real
LLM cost would waste quota to re-measure something that didn't move. Only
the two judges are re-run.

Writes to a *separate* raw file from Unit 4's shipped
`judge-agreement-batch-raw.json` -- that file is the real, already-published
negative result and stays untouched so it remains reproducible and directly
comparable to whatever this rubric produces.

Field-level resumable, same convention as `11e_judge_validation_batch.py`:
`judge_groq`/`judge_gemini` are checkpointed independently per row, so a
partial run (e.g. Gemini still quota-exhausted, Groq open) doesn't lose the
Groq-side scores already collected.
"""

from __future__ import annotations

import json
import time

from aegis.config import REPO_ROOT, settings
from aegis.eval.agreement import cohen_kappa, raw_agreement_rate
from aegis.eval.judge import build_gemini_judge, build_groq_judge

SOURCE_RAW = REPO_ROOT / "docs" / "results" / "judge-agreement-batch-raw.json"
RESULTS_RAW = REPO_ROOT / "docs" / "results" / "judge-agreement-batch-raw-rubric-fix.json"
# gemini-3.6-flash free tier: 5 requests/minute (429 GenerateRequestsPerMinute,
# 2026-10-18 run -- ~13 of that run's failures were this, not the daily cap).
GEMINI_PACE_S = 13.0


def load_source_rows() -> list[dict]:
    with open(SOURCE_RAW, encoding="utf-8") as f:
        rows = json.load(f)
    return [r for r in rows if "error" not in r.get("generation", {"error": 1})]


def load_query_text_by_id() -> dict[str, str]:
    path = settings.paths.data_golden / "golden_set_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        return {(q := json.loads(line))["id"]: q["text"] for line in f}


def load_checkpoint() -> dict[str, dict]:
    if not RESULTS_RAW.exists():
        return {}
    with open(RESULTS_RAW, encoding="utf-8") as f:
        rows = json.load(f)
    return {row["query_id"]: row for row in rows}


def save_checkpoint(rows_by_id: dict[str, dict]) -> None:
    RESULTS_RAW.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_RAW.write_text(json.dumps(list(rows_by_id.values()), indent=2, sort_keys=False))


def main() -> None:
    source_rows = load_source_rows()
    query_text_by_id = load_query_text_by_id()
    print(f"{len(source_rows)} real generation-ok rows from Unit 4's batch")

    groq_judge = build_groq_judge()
    gemini_judge = build_gemini_judge()

    rows_by_id = load_checkpoint()

    for i, src in enumerate(source_rows, 1):
        qid = src["query_id"]
        row = rows_by_id.setdefault(qid, {"query_id": qid, "generation": src["generation"], "ragas": src.get("ragas", {})})
        print(f"\n[{i}/{len(source_rows)}] {qid}")

        query_text = query_text_by_id[qid]
        answer_text = src["generation"]["answer_text"]
        context_block = src["generation"]["context_block"]

        for judge, key in ((groq_judge, "judge_groq"), (gemini_judge, "judge_gemini")):
            if key in row and "error" not in row[key]:
                print(f"  {key}: cached (score={row[key]['score']})")
                continue
            start = time.perf_counter()
            try:
                judgment = judge.judge(query=query_text, answer=answer_text, context_block=context_block)
                row[key] = {"score": judgment.score, "reasoning": judgment.reasoning, "latency_ms": (time.perf_counter() - start) * 1000}
                print(f"  {key}={judgment.score}")
            except Exception as exc:  # noqa: BLE001
                row[key] = {"error": repr(exc)}
                print(f"  {key} FAILED: {exc!r}")
            save_checkpoint(rows_by_id)
            if key == "judge_gemini":
                time.sleep(GEMINI_PACE_S)

    # --- Summary ---
    rows = list(rows_by_id.values())
    print(f"\n{'=' * 60}\nSummary over {len(rows)} rows (rubric-fix re-score)\n{'=' * 60}")

    groq_scores = [r["judge_groq"]["score"] for r in rows if "score" in r.get("judge_groq", {})]
    gemini_scores = [r["judge_gemini"]["score"] for r in rows if "score" in r.get("judge_gemini", {})]
    print(f"judge_groq: n={len(groq_scores)}" + (f" mean={sum(groq_scores) / len(groq_scores):.2f}" if groq_scores else ""))
    print(f"judge_gemini: n={len(gemini_scores)}" + (f" mean={sum(gemini_scores) / len(gemini_scores):.2f}" if gemini_scores else ""))

    paired = [r for r in rows if "score" in r.get("judge_groq", {}) and "score" in r.get("judge_gemini", {})]
    print(f"\nboth judges succeeded: {len(paired)}/{len(rows)}")
    if len(paired) >= 2:
        pg = [r["judge_groq"]["score"] for r in paired]
        pm = [r["judge_gemini"]["score"] for r in paired]
        print(f"raw_agreement_rate={raw_agreement_rate(pg, pm):.4f}")
        print(f"cohen_kappa={cohen_kappa(pg, pm):.4f}")
        print(f"cohen_kappa (quadratic)={cohen_kappa(pg, pm, weights='quadratic'):.4f}")
    else:
        print("fewer than 2 paired rows -- kappa not measurable yet")

    print(f"\nRaw results: {RESULTS_RAW}")


if __name__ == "__main__":
    main()
