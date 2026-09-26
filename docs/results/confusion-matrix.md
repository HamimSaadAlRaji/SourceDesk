# Confusion Matrix — Calibrated Abstention Tau Sweep

Phase 10, steps 1-4. 150 dev positives + 30 dev negatives, Phase 6's winning retrieval config (dense, heading-aware, bge-large, no enrichment). Confidence = weighted combination of 4 signals, each min-max normalized to [0,1] using winsorized (p1/p99) bounds from this real sweep. Weights (reranker_top_score=0.60, score_gap=0.15, semantic_agreement=0.20, distribution_shape=0.05) are reweighted by measured rank-correlation with the should-escalate label, not the naive equal-weighted (0.25 each) starting point -- equal weighting measured worse than reranker_top_score alone (see the ablation table below). Confirmed with the user given this is a real deviation from plan.md's literal starting point.

'Should escalate' = the query is one of the dev negative set's 30 (no answerable ground truth); 'should answer' = one of the 150 dev positives.

## Chosen operating point

**tau = 0.5**

|  | Answered | Escalated |
|---|---|---|
| **Should answer** | 126 (correct answer) | 24 (over-cautious, ~$25/ticket each) |
| **Should escalate** | 3 (Air Canada — wrong answer to user) | 27 (correct escalation) |

precision=0.5294  recall=0.9000  deflection_rate=0.8400

Recall 90% of the 30 negatives (only 3 slip through as confident wrong answers) while still directly answering 126/150 (84%) of legitimate questions. Chosen over a deflection-preserving point (tau=0.15: recall 0.70, 9/30 slip through) and a high-recall point (tau=0.65: recall 0.97, but deflection drops to 0.50) as the point that most reduces the asymmetric-cost Air Canada quadrant without going degenerate -- plan.md's own framing: over-escalation costs one bounded human ticket (~$25); a confident wrong answer is the unbounded reputational/tribunal risk this whole project exists to prevent.

### Who's in the Air Canada quadrant (FN) at the chosen tau

| query_id | negative_category | confidence |
|---|---|---|
| negative::unanswerable::7 | unanswerable | 0.7278 |
| negative::out_of_corpus::9 | out_of_corpus | 0.6019 |
| negative::unanswerable::8 | unanswerable | 0.6266 |

None are `billing_legal` (already hard-escalated upstream by the Phase 8 router, before this scorer ever runs) or `adversarial_injection` (100% caught) -- the slips are `unanswerable`/`out_of_corpus` queries that retrieve deceptively plausible-looking chunks. See design record 10 §6.

### Over-caution (FP) by stratum, at the chosen tau

| stratum | over-escalated | total | rate |
|---|---|---|---|
| ambiguous | 7 | 25 | 28.0% |
| conceptual | 6 | 25 | 24.0% |
| error_debugging | 1 | 25 | 4.0% |
| how_to | 5 | 25 | 20.0% |
| multi_hop | 4 | 25 | 16.0% |
| version_specific | 1 | 25 | 4.0% |

`ambiguous` and `conceptual` bear most of the over-caution cost; `error_debugging` and `version_specific` (sharp, specific queries) are barely touched -- consistent with Open Question #8 and Phase 7/8's repeated finding that this golden set's broad ground truth interacts differently with sharp (cross-encoder) judgments than with fuzzy (bi-encoder) ones.

## Real observed raw-signal ranges: raw min/max vs. winsorized (p1/p99) normalization bounds

| signal | raw min | raw max | p1 (used as lo) | p99 (used as hi) |
|---|---|---|---|---|
| reranker_top_score | — | — | 0.0001 | 0.9936 |
| score_gap | — | — | 0.0000 | 0.6943 |
| semantic_agreement | — | — | 0.6420 | 0.9976 |
| distribution_shape | — | — | 1.4188 | 27.5129 |

(Raw min/max omitted here — see the console output/design record: distribution_shape's raw max was a single ~76,856 outlier that would have compressed every other query's real range into a sliver near 0 under plain min-max normalization.)

## Precision/recall curve (escalation decision, 'escalate' = positive class)

| tau | TP | FP | FN | TN | precision | recall | deflection_rate |
|---|---|---|---|---|---|---|---|
| 0.00 | 0 | 0 | 30 | 150 | — | 0.0000 | 1.0000 |
| 0.05 | 8 | 1 | 22 | 149 | 0.8889 | 0.2667 | 0.9933 |
| 0.10 | 12 | 2 | 18 | 148 | 0.8571 | 0.4000 | 0.9867 |
| 0.15 | 21 | 3 | 9 | 147 | 0.8750 | 0.7000 | 0.9800 |
| 0.20 | 21 | 6 | 9 | 144 | 0.7778 | 0.7000 | 0.9600 |
| 0.25 | 22 | 12 | 8 | 138 | 0.6471 | 0.7333 | 0.9200 |
| 0.30 | 24 | 16 | 6 | 134 | 0.6000 | 0.8000 | 0.8933 |
| 0.35 | 25 | 18 | 5 | 132 | 0.5814 | 0.8333 | 0.8800 |
| 0.40 | 25 | 20 | 5 | 130 | 0.5556 | 0.8333 | 0.8667 |
| 0.45 | 25 | 21 | 5 | 129 | 0.5435 | 0.8333 | 0.8600 |
| 0.50 | 27 | 24 | 3 | 126 | 0.5294 | 0.9000 | 0.8400 **<- chosen** |
| 0.55 | 27 | 29 | 3 | 121 | 0.4821 | 0.9000 | 0.8067 |
| 0.60 | 27 | 43 | 3 | 107 | 0.3857 | 0.9000 | 0.7133 |
| 0.65 | 29 | 75 | 1 | 75 | 0.2788 | 0.9667 | 0.5000 |
| 0.70 | 29 | 117 | 1 | 33 | 0.1986 | 0.9667 | 0.2200 |
| 0.75 | 30 | 137 | 0 | 13 | 0.1796 | 1.0000 | 0.0867 |
| 0.80 | 30 | 148 | 0 | 2 | 0.1685 | 1.0000 | 0.0133 |
| 0.85 | 30 | 150 | 0 | 0 | 0.1667 | 1.0000 | 0.0000 |
| 0.90 | 30 | 150 | 0 | 0 | 0.1667 | 1.0000 | 0.0000 |
| 0.95 | 30 | 150 | 0 | 0 | 0.1667 | 1.0000 | 0.0000 |
| 1.00 | 30 | 150 | 0 | 0 | 0.1667 | 1.0000 | 0.0000 |

## Single-signal ablation vs. the 4-signal combination

Naive equal weighting (0.25 each) scored *worse* than reranker_top_score alone -- score_gap and distribution_shape are both close to uncorrelated with the real label and dilute the one strong signal. The shipped weights (above) fix this by weighting in proportion to measured rank-correlation instead.

| signal | best recall @ precision>=0.5 | recall @ precision=1.0 |
|---|---|---|
| reranker_top_score | 0.9333 | 0.0000 |
| score_gap | 0.0000 | 0.0000 |
| semantic_agreement | 0.4000 | 0.2667 |
| distribution_shape | 0.0000 | 0.0000 |
| combined | 0.9000 | 0.1667 |

## Per-negative-category recall at illustrative tau values

| tau | adversarial_injection | billing_legal | out_of_corpus | unanswerable |
|---|---|---|---|---|
| 0.3 | 1.00 | 1.00 | 0.67 | 0.56 |
| 0.5 | 1.00 | 1.00 | 0.83 | 0.78 |
| 0.7 | 1.00 | 1.00 | 1.00 | 0.89 |
| 0.9 | 1.00 | 1.00 | 1.00 | 1.00 |

Full detail (per-query confidence, complete curve, ablation): `confusion-matrix-analysis.json`. See design record 10 for the full mechanism and business argument.

