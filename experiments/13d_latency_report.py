"""Phase 13 Unit 3 -- turn the raw per-query stage timings from
`13c_latency_attribution.py` into `docs/results/latency-budget.md`. Pure
aggregation, zero LLM calls: safe to re-run any time against the checkpoint.

Writes the BASELINE section only. Unit 4 (optimization against the p95 < 3s
budget) appends its own before/after section so the baseline stays frozen.
"""

from __future__ import annotations

import json
import sys

from aegis.config import REPO_ROOT, settings
from aegis.observability.latency import stage_latency_summary

RAW_PACED = REPO_ROOT / "docs" / "results" / "latency-attribution-paced-raw.json"
RAW_WARMED = REPO_ROOT / "docs" / "results" / "latency-attribution-warmed-raw.json"
RAW_BURST = REPO_ROOT / "docs" / "results" / "latency-attribution-burst-raw.json"
OUT_PATH = REPO_ROOT / "docs" / "results" / "latency-budget.md"
BUDGET_P95_MS = 3000.0  # plan.md Phase 13: p95 < 3 s
STAGE_ORDER = ("cache_lookup", "routing", "retrieval", "confidence", "generation", "validation", "escalation")
PRE_GENERATION = ("cache_lookup", "routing", "retrieval", "confidence")


def fmt(x: float) -> str:
    return f"{x:,.0f}"


def table(summary: dict, total_label: str | None = None, totals=None) -> list[str]:
    lines = ["| Stage | n | mean | p50 | p95 | p99 | max |", "|---|---|---|---|---|---|---|"]
    for stage in STAGE_ORDER:
        if stage in summary:
            s = summary[stage]
            lines.append(f"| {stage} | {s.n} | {fmt(s.mean)} | {fmt(s.p50)} | {fmt(s.p95)} | {fmt(s.p99)} | {fmt(s.max)} |")
    if total_label and totals is not None:
        s = totals
        lines.append(f"| **{total_label}** | {s.n} | {fmt(s.mean)} | {fmt(s.p50)} | {fmt(s.p95)} | {fmt(s.p99)} | {fmt(s.max)} |")
    return lines


def analyze(path):
    rows = json.loads(path.read_text())
    ok = [r for r in rows if not r.get("error")]
    errors = [r for r in rows if r.get("error")]
    cold = [r for r in ok if r.get("cold_start")]
    warm = [r for r in ok if not r.get("cold_start")]
    generated = [r for r in warm if not r["escalate"]]
    escalated = [r for r in warm if r["escalate"]]

    def totals(rs, stages=None):
        per = [{"total": sum(v for k, v in r["stage_latencies_ms"].items() if stages is None or k in stages)} for r in rs]
        return stage_latency_summary(per)["total"]

    providers: dict[str, int] = {}
    for r in generated:
        providers[r["provider"] or "none"] = providers.get(r["provider"] or "none", 0) + 1

    return {
        "rows": rows, "ok": ok, "errors": errors, "cold": cold, "warm": warm,
        "generated": generated, "escalated": escalated, "providers": providers,
        "gen_summary": stage_latency_summary([r["stage_latencies_ms"] for r in generated]),
        "esc_summary": stage_latency_summary([r["stage_latencies_ms"] for r in escalated]),
        "gen_total": totals(generated) if generated else None,
        "esc_total": totals(escalated) if escalated else None,
        "pre_total": totals(warm, PRE_GENERATION) if warm else None,
    }


def verdict(stats) -> str:
    return "within" if stats.p95 < BUDGET_P95_MS else "OVER"


def main() -> int:
    a = analyze(RAW_PACED)
    gen_summary, esc_summary = a["gen_summary"], a["esc_summary"]
    gen_total, esc_total, pre = a["gen_total"], a["esc_total"], a["pre_total"]
    pace = a["rows"][0].get("pace_seconds", 0) if a["rows"] else 0

    lines = [
        "# Latency Budget — Baseline (Phase 13, Unit 3)",
        "",
        "Per-stage wall-clock latency of the real `AnswerPipeline` (real shipped dense retrieval, real cross-encoder",
        "confidence scoring, real Groq router + generation calls), measured by the pipeline itself",
        "(`PipelineResult.stage_latencies_ms`), not read back from Langfuse. All numbers in ms.",
        "Reproduce: `AEGIS_LAT_N_POS=15 AEGIS_LAT_N_NEG=5 AEGIS_LAT_PACE_SECONDS=70 AEGIS_LAT_OUT=latency-attribution-paced-raw.json",
        "uv run python experiments/13c_latency_attribution.py`, then `uv run python experiments/13d_latency_report.py`.",
        "",
        "## Sample (paced baseline)",
        "",
        f"- {len(a['rows'])} queries attempted ({len(a['ok'])} succeeded, {len(a['errors'])} errored), dev split, seed {settings.golden.random_seed}.",
        f"- **{pace:.0f} s idle between queries**, so Groq's free-tier per-minute token caps have fully reset before each request. This measures the pipeline, not rate-limit backoff -- see the burst section for why that distinction matters.",
        "- Cold start (first query in process, lazy model loads): reported separately, excluded from percentiles.",
        f"- Warm: {len(a['warm'])} queries -- {len(a['generated'])} generate path, {len(a['escalated'])} escalated.",
        f"- Generation provider mix: {a['providers']}.",
        "- Caches emptied before the run: every query is a cache miss. Cache-hit latency is Phase 12's number.",
        f"- **Small-sample caveat:** n={len(a['generated'])} on the generate path. p95/p99 are effectively the max of a handful of points -- they bound the tail, they do not measure it.",
        "",
    ]
    if a["errors"]:
        lines += ["### Errored queries (excluded from all statistics)", ""]
        lines += [f"- `{r['query_id']}`: {r['error']}" for r in a["errors"]] + [""]

    lines += ["## Cold start (first query in process)", ""]
    for c in a["cold"]:
        lines += ["| Stage | ms |", "|---|---|"]
        for stage in STAGE_ORDER:
            if stage in c["stage_latencies_ms"]:
                lines.append(f"| {stage} | {fmt(c['stage_latencies_ms'][stage])} |")
        lines += [f"| **wall total** | {fmt(c['wall_ms'])} |", ""]

    if gen_total:
        lines += ["## Generate path (warm, paced)", "", *table(gen_summary, "sum of stages", gen_total), ""]
    if esc_total:
        lines += ["## Escalation path (warm, paced)", "", *table(esc_summary, "sum of stages", esc_total), ""]

    lines += ["## Against the p95 < 3 s budget", "", "| Path | n | p50 | p95 | Budget | Verdict |", "|---|---|---|---|---|---|"]
    if pre:
        lines.append(f"| Pre-generation (cache + routing + retrieval + confidence), all warm | {pre.n} | {fmt(pre.p50)} | {fmt(pre.p95)} | {fmt(BUDGET_P95_MS)} | {verdict(pre)} |")
    if gen_total:
        lines.append(f"| Full generate path | {gen_total.n} | {fmt(gen_total.p50)} | {fmt(gen_total.p95)} | {fmt(BUDGET_P95_MS)} | {verdict(gen_total)} |")
    if esc_total:
        lines.append(f"| Escalation path | {esc_total.n} | {fmt(esc_total.p50)} | {fmt(esc_total.p95)} | {fmt(BUDGET_P95_MS)} | {verdict(esc_total)} |")

    if gen_summary:
        lines += ["", "## Share of generate-path p50 by stage (paced)", ""]
        total_p50 = sum(gen_summary[s].p50 for s in gen_summary)
        for stage in STAGE_ORDER:
            if stage in gen_summary:
                lines.append(f"- {stage}: {gen_summary[stage].p50 / total_p50 * 100:.1f}% ({fmt(gen_summary[stage].p50)} ms)")

    if RAW_BURST.exists():
        b = analyze(RAW_BURST)
        lines += [
            "",
            "## Burst behavior (unpaced, back-to-back) -- why the baseline is paced",
            "",
            f"The first run sent {len(b['rows'])} queries back to back with no idle time. Same code, same corpus, same",
            "models -- and the numbers were wildly different. The likely cause is Groq's free-tier per-minute token",
            "caps (Phase 9: 7,000 ITPM on this model) triggering retry-after waits inside the LLM calls, which would",
            "land in whichever stage made the throttled call. **Supported by the effect, not by direct observation of",
            "the mechanism:** spacing queries 70 s apart removed the stalls (routing p50 5,353 -> 251 ms), but 429s and",
            "retries were not logged, so this was not confirmed at the HTTP level. Generate path, warm:",
            "",
            "| Stage | paced p50 | burst p50 | paced p95 | burst p95 |",
            "|---|---|---|---|---|",
        ]
        for stage in STAGE_ORDER:
            if stage in b["gen_summary"] and stage in gen_summary:
                lines.append(f"| {stage} | {fmt(gen_summary[stage].p50)} | {fmt(b['gen_summary'][stage].p50)} | {fmt(gen_summary[stage].p95)} | {fmt(b['gen_summary'][stage].p95)} |")
        if gen_total and b["gen_total"]:
            lines.append(f"| **sum of stages** | {fmt(gen_total.p50)} | {fmt(b['gen_total'].p50)} | {fmt(gen_total.p95)} | {fmt(b['gen_total'].p95)} |")
        lines += [
            "",
            "**Unexplained:** `confidence` is local CPU work (cross-encoder + encoder, no LLM call) yet was ~45% slower "
            "in the burst run (p50 3,656 vs 2,520 ms). Rate limiting cannot explain that. Machine load during the burst "
            "run is a possibility (other processes were active), but it was not measured or isolated.",
            "",
            f"Burst run: n={len(b['generated'])} generate-path queries, {len(b['escalated'])} escalated, "
            "raw data in `latency-attribution-burst-raw.json`. Burst rows are real measurements of what a user sees "
            "when traffic exceeds the free-tier cap; they are not a measurement of the pipeline's own latency.",
        ]
    lines.append("")

    if RAW_WARMED.exists():
        w = analyze(RAW_WARMED)
        first = w["cold"][0] if w["cold"] else None
        lines += [
            "", "---", "",
            "# Unit 4 -- optimization against the budget", "",
            "One code change was made: `AnswerPipeline.warm_up()` (real retrieval + real confidence scoring at startup,",
            "no LLM calls, no cache/trace side effects). Measured with `AEGIS_LAT_WARMUP=1`, same 70 s pacing, same",
            "sampling as the baseline. **Warm-up cost at startup: 30.3 s** (encoder load 14.2 s + reranker load 16.0 s) -- moved, not removed.",
            "", "## First request: before vs. after warm-up", "",
            "| Stage | before (cold) | after warm-up |", "|---|---|---|",
        ]
        before = a["cold"][0]["stage_latencies_ms"] if a["cold"] else {}
        for stage in STAGE_ORDER:
            if first and stage in first["stage_latencies_ms"] and stage in before:
                lines.append(f"| {stage} | {fmt(before[stage])} | {fmt(first['stage_latencies_ms'][stage])} |")
        if first and a["cold"]:
            lines.append(f"| **wall total** | {fmt(a['cold'][0]['wall_ms'])} | {fmt(first['wall_ms'])} |")
        lines += ["", "## Steady state: re-baseline vs. Unit 3 baseline (generate path, warm)", "",
                  "| Stage | Unit 3 p50 | re-run p50 | Unit 3 p95 | re-run p95 |", "|---|---|---|---|---|"]
        for stage in STAGE_ORDER:
            if stage in gen_summary and stage in w["gen_summary"]:
                lines.append(f"| {stage} | {fmt(gen_summary[stage].p50)} | {fmt(w['gen_summary'][stage].p50)} | {fmt(gen_summary[stage].p95)} | {fmt(w['gen_summary'][stage].p95)} |")
        if gen_total and w["gen_total"]:
            lines.append(f"| **sum of stages** | {fmt(gen_total.p50)} | {fmt(w['gen_total'].p50)} | {fmt(gen_total.p95)} | {fmt(w['gen_total'].p95)} |")
        lines += ["", "| Path | Unit 3 p95 | re-run p95 | Budget | Verdict (re-run) |", "|---|---|---|---|---|"]
        for label, x, y in (("Pre-generation", pre, w["pre_total"]), ("Full generate path", gen_total, w["gen_total"]), ("Escalation path", esc_total, w["esc_total"])):
            if x and y:
                lines.append(f"| {label} | {fmt(x.p95)} | {fmt(y.p95)} | {fmt(BUDGET_P95_MS)} | {verdict(y)} |")
        lines += [
            "", "**Steady state did not improve.** The re-run is within noise of the baseline (some stages slightly worse).",
            "An unrelated Docker stack had been running during the baseline and was gone for the re-run (background load was never",
            "measured, so its effect on the baseline is unknown). Removing it did not reduce the confidence stage's cost, so it does",
            "not appear to explain that cost. **The p95 < 3 s budget is not met.** (Escalation-path n=7: its \"within\" is not robust.)",
            "", "## The unresolved gap: `confidence` costs ~2.4 s in the pipeline but far less in isolation", "",
            "The `confidence` stage is always the real `ConfidenceScorer` (cross-encoder on MPS, pool of 3 pairs). What else ran",
            "in the process varies by row (stated in each). Separate experiments, small n; 13h/13i reuse the same queries across arms:", "",
            "| Condition | n | p50 | p95 | Source |", "|---|---|---|---|---|",
            "| Scorer alone, back-to-back | 15 | 560 | 1,470 | 13f |",
            "| Scorer alone, 70 s idle before each call | 5 | 1,031 | 3,202 | 13f |",
            "| Inside pipeline, no-op tracer, back-to-back | 12 | 502 | 1,344 | 13g |",
            "| Inside pipeline, real Langfuse tracer, back-to-back | 12 | 926 | 6,070 | 13g |",
            "| Full real pipeline (real router + real generation + real validation + real tracer), 70 s idle (Unit 3 baseline) | 12 | 2,520 | 3,233 | 13c |",
            "| Same, re-run with warm-up | 12 | " + fmt(w["gen_summary"]["confidence"].p50) + " | " + fmt(w["gen_summary"]["confidence"].p95) + " | 13c |",
            "| 13h: real Groq router + real tracer, fake generation, 10 s gap (same 7 queries) | 7 | 638 | 1,207 | 13h |",
            "| 13h: same, 0 s gap | 7 | 469 | 969 | 13h |",
            "| 13h: same, 70 s gap (3 usable rows; router quota then ran out) | 3 | 1,497 | 1,656 | 13h |",
            "| 13i: fixed router + real tracer, fake generation, 70 s gap, plain (paired, same queries) | 6 | 656 | 1,072 | 13i |",
            "| 13i: same, with a 1-pair throwaway rerank first (\"wake\") | 6 | 494 | 855 | 13i |",
            "",
            "**What this rules out (as sufficient explanations):** background load (unrelated stack removed, no change); the",
            "real Langfuse tracer (13g's ~2x median did not replicate: 13i with the real tracer is 656 ms); idle gaps of 70 s",
            "(13i plain, 656 ms); the real router LLM (13h); GPU wake-up (a throwaway rerank saves ~127 ms typical across 6/6",
            "pairs but costs 199 ms p50 itself -- net loss). **Localized:** in 13h the extra time is entirely the cross-encoder call",
            "(encode ~2 ms, everything else ~1 ms). **Same-query check:** query `2689` costs 855-1,072 ms in 13i but 2,299-3,326 ms",
            "in the paced/burst batches, so it is the run configuration, not the data. **The one difference not yet tested:** the",
            "batches also run real generation and real citation validation in-process; the fast experiments use fakes for both.",
            "Test prepared (`AEGIS_I_REAL_STACK=1 uv run python experiments/13i_gpu_wake_paired.py`), blocked on Groq quota.",
            "Earlier drafts of this section claimed idle gaps and the tracer each roughly doubled the cost; those claims did not",
            "survive a paired, same-query design and are retracted.",
        ]
    lines.append("")

    OUT_PATH.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
