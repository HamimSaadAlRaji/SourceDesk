# 11h — RAGAS Faithfulness Root Cause

**Phase:** 11 · **Date:** 2026-09-26 · **Files:** `experiments/11h_faithfulness_root_cause.py`
**Topics from the course:** Evaluating the LLM's performance (RAGAS Faithfulness — how it's computed and
what it misses); handling hallucinations (Phase 9/10's citation validator and confidence scorer, compared
directly against a third, independent groundedness signal)

## 1. What I built

A zero-LLM-cost analysis script correlating Unit 4's real RAGAS `faithfulness` scores against the
independently-computed `CitationValidator.valid_citation_rate` on the same 20 real answers, followed by a
manual, real-context check of the two lowest-scoring cases — one confirmed genuine hallucination, one
confirmed honest and accurate refusal.

## 2. Why this approach

`docs/results/judge-agreement.md` flagged `faithfulness=0.7977` (below the 0.90 target) as real but not
root-caused, with three named hypotheses (generation-grounding problem, RAGAS-scoring-model problem, or
golden-set permissiveness). Rather than guess among them or spend more quota trying to move the number,
CLAUDE.md's own framing applies directly: "a component without a measurement is not finished," and a
number without an explanation is not an analysis. The two real case studies were chosen specifically as the
extremes (lowest score overall, and a specific refusal case) because they're where any real mechanism would
be most visible, and neither needed a new LLM call — this project's own already-generated real answers and
real retrieved context were sufficient evidence.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ----------------- |
| Try to raise the faithfulness number with a prompt change (e.g. discourage refusal-style phrasing) | Would move the metric toward the 0.90 target | Case 2 shows the real mechanism is a metric-level blind spot (NLI can't credit a correct negative claim), not a prompt-quality problem — a prompt change that reduces honest refusals to satisfy this metric would trade away exactly the behavior Phase 10's confidence scorer exists to produce. Optimizing against the project's own North Star to move a number is the failure mode this project is built to avoid, not a legitimate fix. |
| Assume the low mean means the generation pipeline is broadly ungrounded, and start there | Simpler story, matches the "target not met" framing on its own | Case 1 (a real hallucination) and Case 2 (a real, verified-accurate refusal that also scored low) can't both be explained by "the pipeline is ungrounded" — they need two different, real mechanisms, and treating the number as one story would have missed the more consequential finding (that the LLM judge completely missed a real hallucination RAGAS caught). |
| Guess at causes from the score distribution alone, without reading real answers/context | Faster, no manual review needed | The whole point of "what you tried" (plan.md's exit-criterion alternative) is a real investigation, not a restated flag — the actual mechanism (an unsupported citation vs. a structurally-unscoreable negative claim) was only visible by reading the real cited chunk text against the real answer claim, not from the numbers alone. |

## 4. How it works

`_correlation` is a plain Pearson-r computation over the 20 real (faithfulness, valid_citation_rate) pairs
— no library needed for a one-off 20-point correlation, consistent with CLAUDE.md's build-vs-import test
(not worth a dependency for this). `_chunk` extracts one numbered context block from the real
`context_block` string by regex, reused from the same citation-numbering convention `guardrails/citations.py`
already parses elsewhere in the project. The script prints the two real case studies' full answer text and
their cited context side by side, which is what actually surfaced both mechanisms — visible immediately on
reading, not computed.

## 5. What the numbers said

Pearson r = 0.3083 between faithfulness and `valid_citation_rate` across the 20 real paired rows — weak
correlation, meaning the two signals agree only partially. `golden::issue::219` (faithfulness=0.077, the
single lowest score) is a real, confident, non-refusal hallucination: its primary cited chunk (`[2]`) is a
completely unrelated GitHub issue, invisible to the citation validator's tier-1 lexical check
(`valid_citation_rate=0.667`, only partially caught) and invisible to the pre-fix LLM judge entirely (scored
5/5). `golden::issue::15764` (faithfulness=0.250) is a real, honest, verified-accurate refusal — its one
positive claim (`route.tags` is a valid attribute) is directly confirmed by the real cited context chunk
(`[1]`) — yet still scored low, because RAGAS's NLI-based check has no mechanism to credit a correct
negative/absence claim.

## 6. What surprised me / what to watch for

The single worst-scoring real answer wasn't a refusal at all — it was the project's most concerning kind of
failure (a confident, plausible, wrong citation), and it slipped past *both* of the project's other two
groundedness guardrails (the citation validator's tier-1 check, and the LLM judge pre-fix). If `faithfulness`
had been dismissed as "probably just penalizing refusals" without reading the real cases, this specific,
consequential miss would have gone unnoticed. Worth watching for future phases: this is real, if
single-case, evidence that the citation validator's tier-1 lexical check can be fooled by shared vocabulary
between an answer and an unrelated chunk, and that the LLM judge (even post-fix) checks helpfulness, not
grounding, by design (design record 11c §1) — RAGAS's `faithfulness` is, on this evidence, currently the
*only* one of the three real signals that reliably catches this failure mode.

## 7. Where this fits and what's next

Satisfies plan.md's Phase 11 exit-criterion alternative for faithfulness ("a written analysis of why not,
and what you tried") with a real investigation, not a restated flag. Recommend treating `faithfulness=0.7977`
as a disclosed, explained limitation rather than a number to chase further — the mechanism behind roughly
half its shortfall (refusal-answer scoring) is upstream and structural, not a project defect, and "fixing"
it would mean optimizing against the project's own confidence-scorer design. The other half (real
hallucinations RAGAS catches that the other guardrails miss) is itself a positive argument for keeping RAGAS
`faithfulness` in the eval suite, not evidence to discount it.
