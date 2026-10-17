# 14i — Do follow-ups resolve with the real answer in history?

**Phase:** 14 · **Date:** 2026-10-17 · **Files:** `experiments/14e_multiturn_regime_c.py`, `docs/results/multiturn-regime-c-raw.json`
**Topics from the course:** Query understanding / conversational rewriting (Module 4); Evaluation metrics: NDCG, Recall, MRR (Module 3); Building a golden set (Module 4)

## 1. What I built

The measurement behind the last open Phase 14 exit criterion, "multi-turn follow-ups resolve correctly". All 24 items of Phase 8's multi-turn set run through three history regimes using the real rewriter, retriever, router and confidence scorer:

- **A:** the follow-up alone.
- **B:** plus the user's first turn. This is Phase 8's regime.
- **C:** plus the system's own real answer to that first turn, cut to the 1,000 characters the server passes to the rewriter. This is what the server actually sends.

The pass criterion was written into the script before any number existed.

## 2. Why this approach

**Regime C was the unmeasured one, and it's the one that ships.** Phase 8 measured rewriting with a single short user turn (+52.7% NDCG@5). The server's history includes long assistant answers, which could either help (more context to resolve "it") or hurt (the rewriter latching onto answer details). 14d's n=6 end-to-end check pointed the wrong way (C hit@10 0/6). But it was too small, and it wasn't comparable to Phase 8: the pipeline doesn't exclude the source issue's own chunks, and those are never ground truth, so self-matches could only push real hits out.

**So, Phase 8's exact protocol for the score, plus the served path for behavior.** Each leg's rewritten query is retrieved twice:

- With `exclude_doc_id=source_issue` and ranx NDCG@5 / Recall@10 / MRR. This is Phase 8's protocol, and leg A has to reproduce Phase 8's number. It does, exactly.
- Without exclusion, through the real router and confidence scorer, to answer "would a user have been escalated?"

**No generation for A/B/C.** Generation changes neither what was retrieved nor whether the pipeline escalates, and it's the expensive call. That held the run at 24 real generations, all for turn 1, instead of 96.

**Pre-registered criterion.** MET if (1) the C−A NDCG@5 paired-bootstrap 95% CI lies entirely above 0, and (2) the C−B CI's lower bound is above −0.05: the real answer in history must not measurably hurt. Written into the script docstring before the run, so a convenient criterion couldn't be picked afterwards.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Re-run 14b's full-pipeline script at n=24 | Already written; fully end to end | 4 generations per item (≈96 total), against qwen's 200k/day and 7k/min caps. And no source-issue exclusion, so not comparable to Phase 8. |
| Score generated answers (judge or RAGAS) instead of retrieval | "Resolves correctly" is about the answer, not the chunks | Phase 11 measured the judge at κ=0.21 (N=11) against the 0.6 bar, so it isn't trustworthy as a pass/fail instrument. And the rewriter's job is the query; retrieval scores it directly against human-labeled ground truth. |
| Hand-write fresh multi-turn conversations | Less synthetic than Phase 8's LLM-generated set | One author's conversations, written after seeing the system's behavior. The 24-item set is frozen, anchored to real golden ground truth, and has a Phase 8 baseline to reproduce. |
| Judge by point estimates (C > A, C ≥ B) | Simpler | At n=24, B−A here has a CI including 0 despite a +0.08 mean. A point-estimate rule would pass on noise. |

## 4. How it works

Per item: `pipeline.answer(turn1)` produces the real assistant turn. That's an answer, or the escalation message when the pipeline escalates, exactly what the UI would put in history. Then for each leg:

```python
text = turn2 if history is None else pipeline.rewriter.rewrite(history, turn2)
eval_hits = pipeline.retriever.search(text, top_k=10, exclude_doc_id=item["source_issue"])   # scored
served    = pipeline.retriever.search(text, top_k=10)                                        # behavior
route     = pipeline.router.classify(text)
escalate  = decide_route(route).escalate or decide_escalation(confidence_scorer.score(text, served).confidence, tau).escalate
```

Paired bootstrap over items: 10,000 resamples of the per-item NDCG@5 differences, seed fixed (`golden.random_seed`), 2.5/97.5 percentiles. Pacing: 40 s after each turn-1 generation (≈4.5k input tokens against a 7k/min cap), 2 s between smaller calls. Checkpointed per item.

## 5. What the numbers said

24/24 items complete, 0 errors. Turn 1 escalated on 10/24; the other 14 were answered, all by Groq.

| Leg | NDCG@5 | Recall@10 | MRR | Escalated (served path) | Mean self-matches in served top-10 |
| --- | --- | --- | --- | --- | --- |
| A — follow-up alone | 0.1995 | 0.0897 | 0.2861 | 12/24 | 0.58 |
| B — + user turn | 0.2805 | 0.1103 | 0.4347 | 9/24 | 0.38 |
| C — + user turn + real answer | **0.3116** | **0.1246** | **0.4816** | **8/24** | 0.42 |
| Phase 8, for reference: A / B / oracle | 0.1995 / 0.3045 / 0.7405 | 0.0897 / 0.1203 / 0.4486 | 0.2861 / 0.4573 / 0.8531 | — | — |

**NDCG@5 paired differences** (mean, 95% CI): B−A **+0.0810** [−0.0025, +0.1715]; C−A **+0.1121** [+0.0220, +0.2110]; C−B **+0.0312** [−0.0004, +0.0664].

**Pre-registered criterion: MET.** The C−A lower bound is +0.0220 > 0, and the C−B lower bound is −0.0004 > −0.05.

**Split by whether turn 1 was answered:**

| Turn 1 | n | A | B | C | C−B | C>B / C<B / C=B |
| --- | --- | --- | --- | --- | --- | --- |
| answered | 14 | 0.2079 | 0.3037 | 0.3567 | +0.0530 | 4 / 0 / 10 |
| escalated | 10 | 0.1877 | 0.2479 | 0.2485 | +0.0006 | 2 / 1 / 7 |

The B and C rewrites were textually identical on 9/24 items. Cost: 24 turn-1 generations, 48 rewrites, 72 router calls, plus tier-2 entailment, all on `qwen/qwen3.8-27b`, wall time ≈ 25 min.

## 6. What surprised me / what to watch for

- **Leg A reproduced Phase 8 to four decimals** (0.1995 / 0.0897 / 0.2861). That's the protocol check passing, so the B and C numbers are directly comparable to Phase 8.
- **Regime C is the best regime, not a risk.** Where a real answer exists, it never hurt (0 of 14 items worse) and improved 4. The worry that long answers in history would derail the rewriter didn't materialize at this n. 14d's n=6 "C 0/6" is consistent with noise plus self-match crowding (the served top-10 carries 0.38–0.58 source-issue chunks on average). I did not re-score 14b to confirm.
- **Phase 8's regime B didn't fully reproduce:** 0.2805 here vs 0.3045. Retrieval is deterministic (A matched exactly), so the difference is in the rewriter's outputs. The rewrite model has changed since Phase 8 (`qwen/qwen3.8-27b` now). And B−A's CI includes 0 at n=24, so Phase 8's +52.7% headline isn't re-confirmed at significance; here it's +40.6% on the mean. Phase 8's write-up reported a point estimate without a CI.
- **Margins are thin.** The C−B lower bound sits at −0.0004, 0.05 inside the tolerance. "MET" means "doesn't hurt and beats nothing", not "solved". Absolute quality stays far below the oracle (0.3116 vs 0.7405 NDCG@5), and 8/24 follow-ups still escalate even in regime C. The escalations are the system honestly abstaining, not answering wrong, but a third of follow-ups getting a human handoff is a real usability cost.
- **The set is synthetic.** 24 LLM-generated follow-ups (Phase 8), anchored to real ground truth. Real users' follow-ups are unmeasured; the 14g review queue is where real ones will accumulate.
- **Ground truth is the original issue's full answer set**, often broad, while a follow-up targets part of it. That caps Recall@10 low for every leg alike. Comparisons between legs are fair; the absolute numbers understate quality.

## 7. Where this fits and what's next

Closes Phase 14's last exit criterion, within the limits above. All six build items and all five exit criteria are now met. Next is Phase 15 (security hardening), which inherits a concrete list from this phase: `/trace` and `/chat` are unauthenticated, the rate limiter is keyed on the socket peer (no trusted-proxy handling), `/trace` returns query and chunk text, and feedback candidates carry raw user text.
