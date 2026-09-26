"""Phase 11 faithfulness root-cause pass (queued since design record 11d,
docs/results/judge-agreement.md's "Not yet done"): why is real RAGAS
`faithfulness` (mean 0.7977, n=20/29, Unit 4) below plan.md's 0.90 target?

Zero new LLM calls -- pure analysis of data already on disk
(`judge-agreement-batch-raw.json`): correlate faithfulness against the
citation validator's independent `valid_citation_rate` (a second, cheaper
groundedness signal from the same real answers), then manually inspect the
lowest-scoring real cases against their real retrieved context to
distinguish "the answer is genuinely unsupported" from "the metric can't
represent this answer type."
"""

from __future__ import annotations

import json
import re

from aegis.config import REPO_ROOT

RAW = REPO_ROOT / "docs" / "results" / "judge-agreement-batch-raw.json"


def _correlation(xs: list[float], ys: list[float]) -> float:
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    sx = (sum((x - mx) ** 2 for x in xs) / len(xs)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys) / len(ys)) ** 0.5
    return cov / (sx * sy) if sx * sy > 0 else float("nan")


def _chunk(context_block: str, n: int) -> str:
    m = re.search(rf"\[{n}\] .*?(?=\n\[\d+\]|\Z)", context_block, re.S)
    return m.group(0) if m else "(not found)"


def main() -> None:
    rows = {r["query_id"]: r for r in json.load(open(RAW, encoding="utf-8"))}
    paired = [
        (qid, r["ragas"]["faithfulness"]["score"], r["generation"]["valid_citation_rate"])
        for qid, r in rows.items()
        if "error" not in r.get("generation", {"error": 1}) and "score" in r.get("ragas", {}).get("faithfulness", {})
    ]
    print(f"{len(paired)} rows with both faithfulness and valid_citation_rate")
    faith = [p[1] for p in paired]
    cite = [p[2] for p in paired]
    print(f"mean faithfulness={sum(faith) / len(faith):.4f}  mean valid_citation_rate={sum(cite) / len(cite):.4f}")
    print(f"pearson correlation(faithfulness, valid_citation_rate)={_correlation(faith, cite):.4f}")
    print()
    for qid, f, c in sorted(paired, key=lambda x: x[1])[:5]:
        print(f"  {qid}: faithfulness={f:.3f} valid_citation_rate={c:.3f}")

    print("\n" + "=" * 60)
    print("Case 1: golden::issue::219 (faithfulness=0.077, lowest of all 20)")
    print("=" * 60)
    r = rows["golden::issue::219"]
    print(r["generation"]["answer_text"][:400])
    print("\n--- cited chunk [2] (the answer's primary citation) ---")
    print(_chunk(r["generation"]["context_block"], 2)[:600])

    print("\n" + "=" * 60)
    print("Case 2: golden::issue::15764 (faithfulness=0.250, an honest refusal)")
    print("=" * 60)
    r = rows["golden::issue::15764"]
    print(r["generation"]["answer_text"])
    print("\n--- cited chunk [1] (supports the answer's one positive claim) ---")
    print(_chunk(r["generation"]["context_block"], 1)[:400])


if __name__ == "__main__":
    main()
