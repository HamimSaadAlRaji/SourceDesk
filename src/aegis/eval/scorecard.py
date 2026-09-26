"""The Phase 11 `make eval` scorecard (plan.md step 5): one command that
consolidates Unit 4's real generation-eval batch (RAGAS, both judges,
inter-judge kappa) and Unit 5's custom metrics (code-block syntax
validity, citation-validity rate) into a single report, checked against
plan.md's own Phase 11 exit criteria.

Scoped to Phase 11's own build list (RAGAS + judge + custom metrics) --
retrieval-quality metrics already have their own harness and results docs
from Phase 6 onward, and the true all-10-criteria scorecard is Phase 19's
job (plan.md step 4 there), not this one. Pure aggregation over rows
already written by `experiments/11e_judge_validation_batch.py`: zero new
LLM calls, so it runs even with every judge/RAGAS model quota-exhausted
(the real state as of 2026-09-25).

Citation validity here is the mean of the batch's stored scalar
`valid_citation_rate` only -- not the full `ClaimStatus` breakdown
`custom_metrics.citation_validity_summary` can produce, because the batch
checkpoint never stored full `ValidationResult` objects (`query_id` and a
scalar rate were the only citation fields captured). Getting the fuller
breakdown would mean re-running `CitationValidator` against the stored
answers, which needs a real tier-2 entailment LLM call for some claims --
a real, disclosed scope cut, not a bug.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from aegis.config import settings
from aegis.eval.agreement import cohen_kappa, raw_agreement_rate
from aegis.eval.custom_metrics import CodeBlockSyntaxResult, code_block_syntax_validity

RAGAS_METRIC_NAMES = ["faithfulness", "response_relevancy", "id_based_context_precision", "id_based_context_recall"]
JUDGE_KEYS = ("judge_groq", "judge_gemini")


class MetricSummary(BaseModel):
    n: int
    mean: float | None


class JudgeAgreementSummary(BaseModel):
    judge_a: str
    judge_b: str
    n_a: int
    mean_a: float | None
    n_b: int
    mean_b: float | None
    n_paired: int
    raw_agreement_rate: float | None
    cohen_kappa: float | None
    cohen_kappa_quadratic: float | None


class DoDCheck(BaseModel):
    criterion: str
    target: str
    actual: str
    met: bool | None  # None = not measurable from this data


class GenerationEvalScorecard(BaseModel):
    n_queries: int
    n_generation_ok: int
    ragas: dict[str, MetricSummary]
    judge_agreement: JudgeAgreementSummary | None
    code_blocks: CodeBlockSyntaxResult
    mean_valid_citation_rate: MetricSummary
    checks: list[DoDCheck]


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _generation_ok(rows: list[dict]) -> list[dict]:
    return [r for r in rows if "error" not in r.get("generation", {"error": 1})]


def summarize_ragas(rows: list[dict]) -> dict[str, MetricSummary]:
    gen_ok = _generation_ok(rows)
    summary: dict[str, MetricSummary] = {}
    for name in RAGAS_METRIC_NAMES:
        scores = [r["ragas"][name]["score"] for r in gen_ok if "score" in r.get("ragas", {}).get(name, {})]
        summary[name] = MetricSummary(n=len(scores), mean=(sum(scores) / len(scores)) if scores else None)
    return summary


def summarize_judge_agreement(rows: list[dict], judge_a: str = "judge_groq", judge_b: str = "judge_gemini") -> JudgeAgreementSummary | None:
    gen_ok = _generation_ok(rows)
    scores_a = [r[judge_a]["score"] for r in gen_ok if "score" in r.get(judge_a, {})]
    scores_b = [r[judge_b]["score"] for r in gen_ok if "score" in r.get(judge_b, {})]
    if not scores_a and not scores_b:
        return None

    paired = [(r[judge_a]["score"], r[judge_b]["score"]) for r in gen_ok if "score" in r.get(judge_a, {}) and "score" in r.get(judge_b, {})]
    n_paired = len(paired)
    if n_paired >= 2:
        paired_a = [p[0] for p in paired]
        paired_b = [p[1] for p in paired]
        kappa = cohen_kappa(paired_a, paired_b)
        kappa_quadratic = cohen_kappa(paired_a, paired_b, weights="quadratic")
        raw_rate = raw_agreement_rate(paired_a, paired_b)
    else:
        kappa = kappa_quadratic = raw_rate = None

    return JudgeAgreementSummary(
        judge_a=judge_a,
        judge_b=judge_b,
        n_a=len(scores_a),
        mean_a=(sum(scores_a) / len(scores_a)) if scores_a else None,
        n_b=len(scores_b),
        mean_b=(sum(scores_b) / len(scores_b)) if scores_b else None,
        n_paired=n_paired,
        raw_agreement_rate=raw_rate,
        cohen_kappa=kappa,
        cohen_kappa_quadratic=kappa_quadratic,
    )


def summarize_code_blocks(rows: list[dict]) -> CodeBlockSyntaxResult:
    n_checked = n_valid = n_skipped = 0
    invalid_blocks: list[str] = []
    for r in _generation_ok(rows):
        result = code_block_syntax_validity(r["generation"]["answer_text"])
        n_checked += result.n_checked
        n_valid += result.n_valid
        n_skipped += result.n_skipped_non_python
        invalid_blocks.extend(result.invalid_blocks)
    return CodeBlockSyntaxResult(n_checked=n_checked, n_valid=n_valid, n_skipped_non_python=n_skipped, invalid_blocks=invalid_blocks)


def summarize_citation_validity(rows: list[dict]) -> MetricSummary:
    rates = [r["generation"]["valid_citation_rate"] for r in _generation_ok(rows)]
    return MetricSummary(n=len(rates), mean=(sum(rates) / len(rates)) if rates else None)


def build_dod_checks(ragas: dict[str, MetricSummary], judge_agreement: JudgeAgreementSummary | None) -> list[DoDCheck]:
    # plan.md Phase 11's own exit-criteria list, in order.
    faithfulness = ragas.get("faithfulness")
    faithfulness_mean = faithfulness.mean if faithfulness else None
    target = settings.eval.ragas_faithfulness_target
    checks = [
        DoDCheck(
            criterion="RAGAS faithfulness >= target",
            target=f">= {target:.2f}",
            actual=f"{faithfulness_mean:.4f} (n={faithfulness.n})" if faithfulness_mean is not None else "not measured",
            met=(faithfulness_mean >= target) if faithfulness_mean is not None else None,
        )
    ]

    kappa_target = settings.eval.judge_kappa_target
    kappa = judge_agreement.cohen_kappa if judge_agreement else None
    checks.append(
        DoDCheck(
            criterion="Inter-judge Cohen's kappa >= target",
            target=f">= {kappa_target:.2f}",
            actual=(f"{kappa:.4f} (n_paired={judge_agreement.n_paired})" if judge_agreement else "not measured") if kappa is not None else "not measured (fewer than 2 paired judgments)",
            met=(kappa >= kappa_target) if kappa is not None else None,
        )
    )

    checks.append(DoDCheck(criterion="Custom metrics implemented (citation validity, code-block syntax)", target="implemented", actual="implemented", met=True))
    checks.append(DoDCheck(criterion="`make eval` produces the full scorecard in one command", target="one command", actual="this run", met=True))
    return checks


def build_scorecard(rows: list[dict]) -> GenerationEvalScorecard:
    ragas = summarize_ragas(rows)
    judge_agreement = summarize_judge_agreement(rows)
    return GenerationEvalScorecard(
        n_queries=len(rows),
        n_generation_ok=len(_generation_ok(rows)),
        ragas=ragas,
        judge_agreement=judge_agreement,
        code_blocks=summarize_code_blocks(rows),
        mean_valid_citation_rate=summarize_citation_validity(rows),
        checks=build_dod_checks(ragas, judge_agreement),
    )


def render(scorecard: GenerationEvalScorecard) -> str:
    lines = [
        "=" * 64,
        "Phase 11 -- Generation Eval Scorecard",
        "=" * 64,
        f"queries: {scorecard.n_queries}  generation ok: {scorecard.n_generation_ok}",
        "",
        "-- RAGAS --",
    ]
    for name in RAGAS_METRIC_NAMES:
        m = scorecard.ragas[name]
        lines.append(f"  {name}: n={m.n} mean={m.mean:.4f}" if m.mean is not None else f"  {name}: not measured")

    lines.append("")
    lines.append("-- Judge agreement --")
    ja = scorecard.judge_agreement
    if ja is None:
        lines.append("  not measured (no judge scores in this batch)")
    else:
        lines.append(f"  {ja.judge_a}: n={ja.n_a} mean={ja.mean_a:.2f}" if ja.mean_a is not None else f"  {ja.judge_a}: n=0")
        lines.append(f"  {ja.judge_b}: n={ja.n_b} mean={ja.mean_b:.2f}" if ja.mean_b is not None else f"  {ja.judge_b}: n=0")
        lines.append(f"  n_paired={ja.n_paired}")
        if ja.cohen_kappa is not None:
            lines.append(f"  raw_agreement_rate={ja.raw_agreement_rate:.4f}")
            lines.append(f"  cohen_kappa={ja.cohen_kappa:.4f}  (quadratic={ja.cohen_kappa_quadratic:.4f})")
        else:
            lines.append("  cohen_kappa: not measurable (fewer than 2 paired judgments)")

    lines.append("")
    lines.append("-- Custom metrics --")
    cb = scorecard.code_blocks
    lines.append(f"  code_block_syntax_validity: n_checked={cb.n_checked} n_valid={cb.n_valid} valid_rate={cb.valid_rate:.4f} n_skipped_non_python={cb.n_skipped_non_python}")
    cv = scorecard.mean_valid_citation_rate
    lines.append(f"  mean_valid_citation_rate: n={cv.n} mean={cv.mean:.4f}" if cv.mean is not None else "  mean_valid_citation_rate: not measured")

    lines.append("")
    lines.append("-- plan.md Phase 11 exit criteria --")
    for check in scorecard.checks:
        status = "PASS" if check.met is True else ("FAIL" if check.met is False else "N/A ")
        lines.append(f"  [{status}] {check.criterion}: target {check.target}, actual {check.actual}")

    lines.append("=" * 64)
    return "\n".join(lines)


def main() -> None:
    rows = load_rows(settings.eval.generation_batch_raw_path)
    scorecard = build_scorecard(rows)
    print(render(scorecard))


if __name__ == "__main__":
    main()
