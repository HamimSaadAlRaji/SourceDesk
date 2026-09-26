"""Phase 10, steps 1-4 (plan.md): compute the four confidence signals for
every dev golden query (150 positives + 30 negatives), sweep tau across its
full range, and produce the escalation confusion matrix + precision/recall
curve.

Two passes, deliberately split so tweaking weights/bounds/tau never re-pays
the expensive pass:

  Pass 1 (compute_raw_signals) -- real dense retrieval (Phase 6's winning
  config: bge-large, no enrichment) + real bge-reranker-v2-m3 scoring of a
  3-candidate pool per query. The only expensive part; written incrementally
  to RESULTS_RAW so a crash mid-run doesn't lose already-collected queries
  (the same lesson experiments/07b_kaggle_rerank_sweep.py and
  experiments/09a_sampling_sweep.py already paid for). Device forced to
  "cpu" for both encoder and reranker -- this project has hit the
  MPS-under-memory-pressure hang three times already (Phases 3, 7, 8); not
  repeating it a fourth time.

  Pass 2 (analyze) -- pure numpy over the saved raw signals: min-max bounds
  from the real observed range, equal-weighted combination, a fine tau
  sweep, the 2x2 confusion matrix and precision/recall curve at each tau,
  plus a per-signal ablation (does the 4-signal combination actually beat
  the best single signal alone?). Cheap enough to rerun on every tweak.

Scope note: only the *dev* split (150 positives + 30 negatives) is used
here, not the golden set's full 40 negatives (30 dev + 10 holdout).
Consistent with the standing rule elsewhere in this project -- "Phases 6-16
measure on dev, Phase 19 on holdout for generalization" (progress.md) --
Phase 10 falls inside that range. plan.md's literal Phase 10 wording ("All
40 negatives behave correctly") predates the frozen dev/holdout split
decision, the same kind of superseded-by-a-later-decision wording Phase 6
resolved for its own "lift from baseline" language.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.guardrails.confidence import SIGNAL_NAMES, _raw_signals
from aegis.rerank.cross_encoder import CrossEncoderReranker
from aegis.retrieval.dense import DenseIndex
from aegis.types import Chunk, GoldenQuery

RESULTS_RAW = REPO_ROOT / "docs" / "results" / "confusion-matrix-raw.json"
RESULTS_MD = REPO_ROOT / "docs" / "results" / "confusion-matrix.md"
RESULTS_PNG = REPO_ROOT / "docs" / "results" / "confusion-matrix.png"


def load_chunks() -> list[Chunk]:
    path = settings.paths.data_processed / "chunks.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def load_dev_queries() -> list[GoldenQuery]:
    path = settings.paths.data_golden / "golden_set_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        return [GoldenQuery(**json.loads(line)) for line in f]


def compute_raw_signals(chunks: list[Chunk], queries: list[GoldenQuery]) -> list[dict]:
    cache = EmbeddingCache(settings.paths.embedding_cache)
    encoder = Encoder(model_name=settings.embedding.bge_large_model, cache=cache, device="cpu")
    dense_index = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)
    reranker = CrossEncoderReranker(device="cpu")
    pool_size = settings.confidence.rerank_pool_size

    records: list[dict] = []
    t_start = time.perf_counter()
    for i, q in enumerate(queries):
        candidates = dense_index.search(q.text, top_k=settings.generation.retrieval_top_k, exclude_doc_id=q.source_issue)

        # Reuses guardrails/confidence.py's own signal computation (TDD'd,
        # tests/test_confidence.py) rather than re-deriving the same
        # formulas here -- exactly the kind of duplicated-logic risk
        # CLAUDE.md's decision log repeatedly flags for this project.
        raw = _raw_signals(q.text, candidates, reranker, encoder.encode, pool_size)

        records.append(
            {
                "query_id": q.id,
                "is_negative": q.is_negative,
                "stratum": q.stratum.value if q.stratum else None,
                "negative_category": q.negative_category.value if q.negative_category else None,
                "n_candidates": len(candidates),
                "raw": raw,
            }
        )

        RESULTS_RAW.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_RAW.write_text(json.dumps(records, indent=2))

        if (i + 1) % 20 == 0 or (i + 1) == len(queries):
            elapsed = time.perf_counter() - t_start
            print(f"  {i + 1}/{len(queries)} queries, {elapsed:.0f}s elapsed, {elapsed / (i + 1):.2f}s/query avg")

    return records


def _confusion_at_tau(confidence: np.ndarray, is_negative: np.ndarray, tau: float) -> dict[str, float]:
    escalate = confidence < tau
    tp = int(np.sum(escalate & is_negative))
    fp = int(np.sum(escalate & ~is_negative))
    fn = int(np.sum(~escalate & is_negative))
    tn = int(np.sum(~escalate & ~is_negative))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    deflection_rate = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {"tau": tau, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "deflection_rate": deflection_rate}


def analyze(records: list[dict]) -> None:
    is_negative = np.array([r["is_negative"] for r in records], dtype=bool)
    n_pos, n_neg = int((~is_negative).sum()), int(is_negative.sum())
    print(f"\n{len(records)} records ({n_pos} positive / {n_neg} negative)")

    raw_by_signal = {name: np.array([r["raw"][name] for r in records], dtype=np.float64) for name in SIGNAL_NAMES}
    # Winsorized (p1/p99) bounds, not raw min/max -- distribution_shape's raw
    # max is a single ~76,856 outlier (near-zero variance among one query's
    # non-top-1 dense scores; see design record 10 §6) that would otherwise
    # compress every other query's real range into a sliver near 0.
    bounds = {name: (float(np.percentile(arr, 1)), float(np.percentile(arr, 99))) for name, arr in raw_by_signal.items()}
    print(f"\nReal observed raw-signal ranges (dev set, N={len(records)}), raw min/max vs. winsorized p1/p99 bounds used for normalization:")
    for name, arr in raw_by_signal.items():
        lo, hi = bounds[name]
        print(f"  {name}: raw=[{arr.min():.4f}, {arr.max():.4f}]  p1/p99=[{lo:.4f}, {hi:.4f}]")

    normalized = {
        name: np.clip((arr - bounds[name][0]) / (bounds[name][1] - bounds[name][0] + 1e-12), 0.0, 1.0)
        for name, arr in raw_by_signal.items()
    }

    weights = {
        "reranker_top_score": settings.confidence.weight_reranker_top_score,
        "score_gap": settings.confidence.weight_score_gap,
        "semantic_agreement": settings.confidence.weight_semantic_agreement,
        "distribution_shape": settings.confidence.weight_distribution_shape,
    }
    weight_sum = sum(weights.values())
    confidence = sum(normalized[name] * weights[name] for name in SIGNAL_NAMES) / weight_sum

    taus = np.round(np.arange(0.0, 1.001, 0.01), 2)
    curve = [_confusion_at_tau(confidence, is_negative, t) for t in taus]

    # Ablation: how does the 4-signal combination compare to each signal alone?
    print("\nSingle-signal ablation -- AUC-style summary (max recall achieved while precision >= 0.5):")
    ablation_rows = []
    for name in (*SIGNAL_NAMES, "combined"):
        conf = normalized[name] if name != "combined" else confidence
        single_curve = [_confusion_at_tau(conf, is_negative, t) for t in taus]
        best_recall_at_p50 = max(
            (row["recall"] for row in single_curve if not np.isnan(row["precision"]) and row["precision"] >= 0.5),
            default=0.0,
        )
        recall_at_perfect_precision = max(
            (row["recall"] for row in single_curve if row["precision"] == 1.0), default=0.0
        )
        print(f"  {name}: best recall with precision>=0.5: {best_recall_at_p50:.4f}; recall at precision=1.0: {recall_at_perfect_precision:.4f}")
        ablation_rows.append(
            {"signal": name, "best_recall_at_precision_0.5": best_recall_at_p50, "recall_at_precision_1.0": recall_at_perfect_precision}
        )

    # Per-negative-category breakdown at a few illustrative tau values
    neg_categories = sorted({r["negative_category"] for r in records if r["negative_category"]})
    print("\nPer-negative-category recall at illustrative tau values:")
    category_breakdown = {}
    for tau in (0.3, 0.5, 0.7, 0.9):
        escalate = confidence < tau
        row = {}
        for cat in neg_categories:
            mask = np.array([r["negative_category"] == cat for r in records])
            row[cat] = float(np.sum(escalate & mask) / mask.sum()) if mask.sum() > 0 else float("nan")
        category_breakdown[tau] = row
        print(f"  tau={tau}: " + ", ".join(f"{cat}={rate:.2f}" for cat, rate in row.items()))

    # Chosen tau (settings.confidence.tau): who's in the Air-Canada quadrant
    # (FN, should escalate but answered), and which strata bear the
    # over-caution cost (FP, should answer but escalated)?
    tau = settings.confidence.tau
    escalate_chosen = confidence < tau
    fn_detail = [
        {"query_id": r["query_id"], "negative_category": r["negative_category"], "confidence": float(c)}
        for r, c, esc in zip(records, confidence, escalate_chosen)
        if r["is_negative"] and not esc
    ]
    fp_by_stratum: dict[str, dict[str, int]] = {}
    for r, esc in zip(records, escalate_chosen):
        if r["is_negative"] or r["stratum"] is None:
            continue
        bucket = fp_by_stratum.setdefault(r["stratum"], {"over_escalated": 0, "total": 0})
        bucket["total"] += 1
        if esc:
            bucket["over_escalated"] += 1
    print(f"\nChosen tau={tau}: FN (Air Canada quadrant) = {[d['query_id'] for d in fn_detail]}")
    print("FP (over-cautious) by stratum: " + ", ".join(f"{k}={v['over_escalated']}/{v['total']}" for k, v in sorted(fp_by_stratum.items())))

    payload = {
        "n_positive": n_pos,
        "n_negative": n_neg,
        "bounds": bounds,
        "weights": weights,
        "tau": tau,
        "curve": curve,
        "ablation": ablation_rows,
        "category_breakdown": {str(k): v for k, v in category_breakdown.items()},
        "fn_detail": fn_detail,
        "fp_by_stratum": fp_by_stratum,
        "confidence_by_query": [
            {"query_id": r["query_id"], "is_negative": r["is_negative"], "stratum": r["stratum"], "negative_category": r["negative_category"], "confidence": float(c)}
            for r, c in zip(records, confidence)
        ],
    }
    RESULTS_RAW.with_name("confusion-matrix-analysis.json").write_text(json.dumps(payload, indent=2))

    write_report(curve, bounds, weights, ablation_rows, category_breakdown, fn_detail, fp_by_stratum, tau, n_pos, n_neg)
    write_plot(confidence, is_negative, curve, taus, tau)
    print(f"\nReport written to {RESULTS_MD}")
    print(f"Plot written to {RESULTS_PNG}")


def write_report(curve, bounds, weights, ablation_rows, category_breakdown, fn_detail, fp_by_stratum, tau, n_pos, n_neg) -> None:
    chosen = next(row for row in curve if abs(row["tau"] - tau) < 1e-9)
    lines = [
        "# Confusion Matrix — Calibrated Abstention Tau Sweep",
        "",
        f"Phase 10, steps 1-4. {n_pos} dev positives + {n_neg} dev negatives, Phase 6's winning retrieval config "
        "(dense, heading-aware, bge-large, no enrichment). Confidence = weighted combination of 4 signals, each "
        "min-max normalized to [0,1] using winsorized (p1/p99) bounds from this real sweep. Weights "
        f"({', '.join(f'{k}={v:.2f}' for k, v in weights.items())}) are reweighted by measured rank-correlation "
        "with the should-escalate label, not the naive equal-weighted (0.25 each) starting point -- equal "
        "weighting measured worse than reranker_top_score alone (see the ablation table below). Confirmed with "
        "the user given this is a real deviation from plan.md's literal starting point.",
        "",
        "'Should escalate' = the query is one of the dev negative set's 30 (no answerable ground truth); "
        "'should answer' = one of the 150 dev positives.",
        "",
        "## Chosen operating point",
        "",
        f"**tau = {tau}**",
        "",
        "|  | Answered | Escalated |",
        "|---|---|---|",
        f"| **Should answer** | {chosen['tn']} (correct answer) | {chosen['fp']} (over-cautious, ~$25/ticket each) |",
        f"| **Should escalate** | {chosen['fn']} (Air Canada — wrong answer to user) | {chosen['tp']} (correct escalation) |",
        "",
        f"precision={chosen['precision']:.4f}  recall={chosen['recall']:.4f}  deflection_rate={chosen['deflection_rate']:.4f}",
        "",
        f"Recall {chosen['recall']:.0%} of the 30 negatives (only {chosen['fn']} slip through as confident wrong "
        f"answers) while still directly answering {chosen['tn']}/{n_pos} ({chosen['tn'] / n_pos:.0%}) of legitimate "
        "questions. Chosen over a deflection-preserving point (tau=0.15: recall 0.70, 9/30 slip through) and a "
        "high-recall point (tau=0.65: recall 0.97, but deflection drops to 0.50) as the point that most reduces "
        "the asymmetric-cost Air Canada quadrant without going degenerate -- plan.md's own framing: over-escalation "
        "costs one bounded human ticket (~$25); a confident wrong answer is the unbounded reputational/tribunal "
        "risk this whole project exists to prevent.",
        "",
        "### Who's in the Air Canada quadrant (FN) at the chosen tau",
        "",
        "| query_id | negative_category | confidence |",
        "|---|---|---|",
    ]
    for d in fn_detail:
        lines.append(f"| {d['query_id']} | {d['negative_category']} | {d['confidence']:.4f} |")
    lines += [
        "",
        "None are `billing_legal` (already hard-escalated upstream by the Phase 8 router, before this scorer ever "
        "runs) or `adversarial_injection` (100% caught) -- the slips are `unanswerable`/`out_of_corpus` queries "
        "that retrieve deceptively plausible-looking chunks. See design record 10 §6.",
        "",
        "### Over-caution (FP) by stratum, at the chosen tau",
        "",
        "| stratum | over-escalated | total | rate |",
        "|---|---|---|---|",
    ]
    for stratum, bucket in sorted(fp_by_stratum.items()):
        rate = bucket["over_escalated"] / bucket["total"] if bucket["total"] else float("nan")
        lines.append(f"| {stratum} | {bucket['over_escalated']} | {bucket['total']} | {rate:.1%} |")
    lines += [
        "",
        "`ambiguous` and `conceptual` bear most of the over-caution cost; `error_debugging` and `version_specific` "
        "(sharp, specific queries) are barely touched -- consistent with Open Question #8 and Phase 7/8's repeated "
        "finding that this golden set's broad ground truth interacts differently with sharp (cross-encoder) "
        "judgments than with fuzzy (bi-encoder) ones.",
        "",
        "## Real observed raw-signal ranges: raw min/max vs. winsorized (p1/p99) normalization bounds",
        "",
        "| signal | raw min | raw max | p1 (used as lo) | p99 (used as hi) |",
        "|---|---|---|---|---|",
    ]
    for name, (lo, hi) in bounds.items():
        lines.append(f"| {name} | — | — | {lo:.4f} | {hi:.4f} |")
    lines += [
        "",
        "(Raw min/max omitted here — see the console output/design record: distribution_shape's raw max was a "
        "single ~76,856 outlier that would have compressed every other query's real range into a sliver near 0 "
        "under plain min-max normalization.)",
    ]

    lines += ["", "## Precision/recall curve (escalation decision, 'escalate' = positive class)", "", "| tau | TP | FP | FN | TN | precision | recall | deflection_rate |", "|---|---|---|---|---|---|---|---|"]
    for row in curve:
        if abs(round(row["tau"] * 20) - row["tau"] * 20) < 1e-9:  # every 0.05
            p = f"{row['precision']:.4f}" if not np.isnan(row["precision"]) else "—"
            r = f"{row['recall']:.4f}" if not np.isnan(row["recall"]) else "—"
            d = f"{row['deflection_rate']:.4f}" if not np.isnan(row["deflection_rate"]) else "—"
            marker = " **<- chosen**" if abs(row["tau"] - tau) < 1e-9 else ""
            lines.append(f"| {row['tau']:.2f} | {row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} | {p} | {r} | {d}{marker} |")

    lines += [
        "",
        "## Single-signal ablation vs. the 4-signal combination",
        "",
        "Naive equal weighting (0.25 each) scored *worse* than reranker_top_score alone -- score_gap and "
        "distribution_shape are both close to uncorrelated with the real label and dilute the one strong signal. "
        "The shipped weights (above) fix this by weighting in proportion to measured rank-correlation instead.",
        "",
        "| signal | best recall @ precision>=0.5 | recall @ precision=1.0 |",
        "|---|---|---|",
    ]
    for row in ablation_rows:
        lines.append(f"| {row['signal']} | {row['best_recall_at_precision_0.5']:.4f} | {row['recall_at_precision_1.0']:.4f} |")

    lines += ["", "## Per-negative-category recall at illustrative tau values", "", "| tau | " + " | ".join(category_breakdown[0.3].keys()) + " |", "|---|" + "---|" * len(category_breakdown[0.3])]
    for t, row in category_breakdown.items():
        lines.append(f"| {t} | " + " | ".join(f"{v:.2f}" for v in row.values()) + " |")

    lines += ["", "Full detail (per-query confidence, complete curve, ablation): `confusion-matrix-analysis.json`. See design record 10 for the full mechanism and business argument.", ""]
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text("\n".join(lines) + "\n")


def write_plot(confidence: np.ndarray, is_negative: np.ndarray, curve: list[dict], taus: np.ndarray, tau: float) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.hist(confidence[~is_negative], bins=25, alpha=0.6, label="should answer (positives)", color="tab:blue")
    ax1.hist(confidence[is_negative], bins=25, alpha=0.6, label="should escalate (negatives)", color="tab:red")
    ax1.axvline(tau, color="black", linestyle="--", linewidth=1.5, label=f"chosen tau={tau}")
    ax1.set_xlabel("confidence score")
    ax1.set_ylabel("count")
    ax1.set_title("Confidence distribution by ground truth")
    ax1.legend()

    precisions = [row["precision"] for row in curve]
    recalls = [row["recall"] for row in curve]
    deflections = [row["deflection_rate"] for row in curve]
    ax2.plot(taus, precisions, label="precision (of escalation)", color="tab:green")
    ax2.plot(taus, recalls, label="recall (of escalation)", color="tab:red")
    ax2.plot(taus, deflections, label="deflection rate", color="tab:blue", linestyle=":")
    ax2.axvline(tau, color="black", linestyle="--", linewidth=1.5, label=f"chosen tau={tau}")
    ax2.set_xlabel("tau")
    ax2.set_ylabel("score")
    ax2.set_title("Precision / recall / deflection vs. tau")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_PNG, dpi=150)


def main() -> None:
    chunks = load_chunks()
    queries = load_dev_queries()
    print(f"{len(chunks)} chunks, {len(queries)} dev queries ({sum(not q.is_negative for q in queries)} positive / {sum(q.is_negative for q in queries)} negative)")

    if RESULTS_RAW.exists():
        print(f"Loading cached raw signals from {RESULTS_RAW}")
        records = json.loads(RESULTS_RAW.read_text())
        if len(records) != len(queries):
            print(f"  cache has {len(records)} records but {len(queries)} queries exist -- recomputing")
            records = compute_raw_signals(chunks, queries)
    else:
        print("Computing raw signals for real (dense retrieval + cross-encoder, CPU)...")
        records = compute_raw_signals(chunks, queries)

    analyze(records)


if __name__ == "__main__":
    main()
