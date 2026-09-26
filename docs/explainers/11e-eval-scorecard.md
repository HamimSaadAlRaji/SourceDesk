# 11e — The `make eval` Scorecard (Unit 6)

**Phase:** 11 · **Date:** 2026-09-25 · **Files:** `src/aegis/eval/scorecard.py`, `src/aegis/config.py`
(`EvalSettings`), `Makefile` (`eval` target), `tests/test_scorecard.py`
**Topics from the course:** Evaluation strategies (consolidating code-based and LLM-judge signals into
one report); Choosing an LLM & reading benchmarks (turning raw scores into pass/fail against a stated bar)

## 1. What I built

`src/aegis/eval/scorecard.py` — pure aggregation over Unit 4's already-written batch rows
(`docs/results/judge-agreement-batch-raw.json`): RAGAS means, inter-judge Cohen's κ (reusing Unit 2's
`agreement.py`), code-block syntax validity and mean citation-validity rate (reusing Unit 5's
`custom_metrics.py`), checked against plan.md's Phase 11 exit criteria. `make eval` now runs
`python -m aegis.eval.scorecard` and prints the full report in one command — the literal exit criterion.

## 2. Why this approach

Every model this phase depends on (`qwen/qwen3.8-27b`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`,
`gemini-3.5-flash`) is real-quota-exhausted as of this session (design record 11d). A scorecard that
required a fresh live run to produce anything would be unusable right now. Since Unit 4's batch already
persisted every intermediate result field-by-field (its own resumability design, 11d §2), everything Unit 6
needs is sitting on disk already measured — this is genuinely aggregation, not new measurement, so it
doesn't need to wait on quota at all.

Scoped to Phase 11's own build list (RAGAS, judge, custom metrics) — not a retrieval-plus-generation
combined scorecard. plan.md gives Phase 19 its own "final scorecard against all 10 Definition of Done
criteria" (plan.md step 4 there); building that scope here would duplicate work Phase 19 already owns, and
Phase 11's own four build steps never mention retrieval metrics.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ----------------- |
| Wait for quota reset, re-run the judge-rubric fix, then build the scorecard on top of clean numbers | Scorecard would show a healthier κ | Conflates two different units of work. The scorecard's job is to report the real numbers honestly, including a failing one — that's what plan.md's own exit criterion ("RAGAS faithfulness >= 0.90, or a written analysis of why not") asks for. Re-running Unit 4 is real, separate, already-queued work (progress.md "Next action"), not a prerequisite for Unit 6 existing. |
| Re-run `CitationValidator` against the batch's stored answers to get the full `ClaimStatus` breakdown (Unit 5's `citation_validity_summary`) instead of just the stored scalar mean | Richer, per-status breakdown, actually exercises `citation_validity_summary` on real data | Tier-2 entailment needs a live LLM call (`qwen/qwen3.8-27b`) for some claims — reintroduces quota risk into what should be a zero-cost aggregation step. The scalar mean is real data already on disk; the richer breakdown is a disclosed, separate follow-up, not silently dropped. |
| Build a combined retrieval + generation scorecard now, pulling from `data/cache/eval/*.json` | One truly all-in-one report | Those cache files are per-axis ablation sweep points (retrieval-strategy axis held chunking fixed, and vice versa) from Phase 6's grid search, not one row that *is* "the winning config" — reconstructing a single combined number would mean re-deriving a live retrieval run, a different (and already-served-elsewhere, e.g. Phase 7-9's own scripts) piece of work than what Phase 11 asks for. |

## 4. How it works

`_generation_ok(rows)` filters to rows whose `generation` field is present and error-free — the same
predicate the batch script itself uses for its own summary block, kept consistent rather than reinvented.
Each `summarize_*` function is a pure function over that filtered row list:

```python
def summarize_judge_agreement(rows, judge_a="judge_groq", judge_b="judge_gemini") -> JudgeAgreementSummary | None:
    gen_ok = _generation_ok(rows)
    scores_a = [r[judge_a]["score"] for r in gen_ok if "score" in r.get(judge_a, {})]
    scores_b = [r[judge_b]["score"] for r in gen_ok if "score" in r.get(judge_b, {})]
    if not scores_a and not scores_b:
        return None
    paired = [(r[judge_a]["score"], r[judge_b]["score"]) for r in gen_ok
              if "score" in r.get(judge_a, {}) and "score" in r.get(judge_b, {})]
    ...
```

`n_paired < 2` returns `cohen_kappa=None` rather than calling sklearn on a degenerate sample — kappa isn't
meaningfully defined on 0 or 1 paired items, and a `None` propagates honestly into the scorecard's rendered
output ("not measurable") instead of a misleading number.

`build_dod_checks` reads two real thresholds from `EvalSettings` (`ragas_faithfulness_target=0.90`,
`judge_kappa_target=0.6`) rather than hardcoding them in comparison logic, per CLAUDE.md §6 — they're real
plan.md-stated bars used in `met = actual >= target` checks, not illustrative values.

## 5. What the numbers said

The real `make eval` output, unchanged from what the command actually prints:

```
queries: 30  generation ok: 29
-- RAGAS --
  faithfulness: n=20 mean=0.7977
  response_relevancy: n=27 mean=0.5662
  id_based_context_precision: n=29 mean=0.5828
  id_based_context_recall: n=29 mean=0.4103
-- Judge agreement --
  judge_groq: n=20 mean=3.10
  judge_gemini: n=11 mean=4.91
  n_paired=9
  raw_agreement_rate=0.2222
  cohen_kappa=-0.0161  (quadratic=0.0906)
-- Custom metrics --
  code_block_syntax_validity: n_checked=8 n_valid=8 valid_rate=1.0000 n_skipped_non_python=0
  mean_valid_citation_rate: n=29 mean=0.7069
-- plan.md Phase 11 exit criteria --
  [FAIL] RAGAS faithfulness >= target: target >= 0.90, actual 0.7977 (n=20)
  [FAIL] Inter-judge Cohen's kappa >= target: target >= 0.60, actual -0.0161 (n_paired=9)
  [PASS] Custom metrics implemented (citation validity, code-block syntax): target implemented, actual implemented
  [PASS] `make eval` produces the full scorecard in one command: target one command, actual this run
```

Two of four criteria genuinely fail — reported as failures, not smoothed over. `code_block_syntax_validity`
is a real, if small-sample (n=8), clean result: every Python block in this batch's real generated answers
parsed. `tests/test_scorecard.py`'s regression test ties these exact numbers to the raw batch file, same
convention as `test_custom_metrics.py`'s Phase 10 confusion-matrix guard.

## 6. What surprised me / what to watch for

`make eval` was silently non-functional before this unit — `aegis/eval/harness.py` (the target the Makefile
already pointed at) has a `run_eval` function but no `if __name__ == "__main__":` block, so the command did
nothing. Not a regression introduced here, but worth naming: the exit criterion "`make eval` produces the
full scorecard in one command" wasn't just unmet, the command itself was a no-op until this unit gave it one.

n=8 code blocks checked is a small sample to draw any real conclusion from — flagged as such rather than
implying the 100% valid_rate generalizes. Citation validity's mean (0.7069) is a coarser signal than Unit
5's own `citation_validity_summary` can produce (no per-`ClaimStatus` breakdown available from this batch's
stored data, see §3) — real information loss, not fabricated precision.

## 7. Where this fits and what's next

Phase 11's four build steps and its `make eval` exit criterion are now all real and passing (mechanically —
the command runs and reports true numbers), even though two of the *substantive* targets it checks
(faithfulness, κ) are honestly failing. That's the correct state to ship in: the scorecard's job is to
surface the real number, not to only exist once the number looks good. The next real step is still the one
already queued in 11d — fix the judge rubric's refusal-scoring divergence and re-run Unit 4's batch once
quota resets, then re-run `make eval` to see whether the fix moves κ and (contingently) whether a root-cause
pass on `faithfulness=0.7977` is warranted. This scorecard will pick up better numbers automatically the
next time Unit 4's raw file is regenerated — no changes needed here for that.
