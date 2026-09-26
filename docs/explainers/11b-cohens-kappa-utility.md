# 11b — Cohen's Kappa Utility

**Phase:** 11 · **Date:** 2026-09-25 · **Files:** `src/aegis/eval/agreement.py`, `tests/test_agreement.py`
**Topics from the course:** Evaluation strategies (system vs. component, code-based vs. LLM-judge vs.
human feedback); Evaluating the LLM's performance (RAGAS: Response Relevancy, Faithfulness)

## 1. What I built

`cohen_kappa(labels_a, labels_b, weights=None)` and `raw_agreement_rate(labels_a, labels_b)` —
`eval/agreement.py`, a thin wrapper over `sklearn.metrics.cohen_kappa_score`. Takes two equal-length
label sequences from two raters over the same items, returns the chance-corrected agreement score
(`cohen_kappa`) and the plain fraction where both raters gave the identical label
(`raw_agreement_rate`). No project-specific logic yet — this unit is the reusable statistics primitive;
Unit 3 (LLM-as-judge, built twice) and Unit 4 (the N=30 batch run) are what actually produce the two
label sequences to feed it.

## 2. Why this approach

Plan.md's Phase 11 step 3 treats κ as a hard gate: "If κ < 0.6 the judge is unreliable and every metric
built on it is decoration." That's a strong claim to rest on a hand-rolled formula nobody but this
project has ever run. A silently wrong kappa implementation is worse than no kappa at all — it would
produce a confident, wrong "the judge is reliable" (or "unreliable") verdict that every later unit takes
on faith, which is exactly the class of bug CLAUDE.md's testing section calls out (analyzer, chunkers,
RRF, confidence scorer, citation validator) as needing TDD rather than trust. Kappa belongs in that list
for the same reason, even though it isn't named there explicitly.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ----------------- |
| Hand-roll the formula (`(po - pe) / (1 - pe)` directly) | No new dependency; the formula is genuinely simple | Doesn't prove anything a reader would credit — Cohen's kappa is a named, standard statistic with one correct definition, not a project-specific technique like RRF fusion or the confidence scorer's weighted-signal combination. `sklearn`'s version is the one a reviewer can independently verify against, and it already correctly handles weighted (ordinal) variants this project may need in Unit 3, which a from-scratch version would have to reimplement separately and re-verify |
| Leave `scikit-learn` as a transitive-only dependency (it was already installed via the `sentence-transformers`/`ragas` stack) instead of adding it directly | Zero-line change; it already works | Importing from a package that isn't declared as a direct dependency is fragile — if an upstream package (e.g. `sentence-transformers`) ever drops its own `scikit-learn` requirement, this module breaks with no signal in `pyproject.toml` explaining why. `uv add scikit-learn` costs one line and makes the real dependency explicit |

## 4. How it works

```python
def cohen_kappa(labels_a: list, labels_b: list, weights: str | None = None) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError(...)
    return float(cohen_kappa_score(labels_a, labels_b, weights=weights))
```

`weights=None` is unweighted (nominal) kappa — every disagreement counts equally, correct for
categorical rubrics (e.g. pass/fail/escalate). `weights="quadratic"` discounts disagreements by how far
apart the two labels are — correct if Unit 3's helpfulness rubric ends up an ordinal scale (e.g. 1-5),
where a 4-vs-5 disagreement is a much smaller problem than a 1-vs-5 one and unweighted kappa would
overstate how badly the two judges actually diverge. Not decided yet which the real rubric needs —
exposed as a parameter so Unit 3 picks it once the rubric shape exists, rather than guessing now.

`cast to float` matters the same way it did for `eval/metrics.py`'s `ranx` wrapper: `cohen_kappa_score`
returns a numpy scalar, and every caller in this project expects plain Python floats (JSON-serializable
for the eventual `docs/results/judge-agreement.md` write-up and `make eval` scorecard).

## 5. What the numbers said

Not yet measured — Phase 11 Unit 4. This unit's own tests use a hand-computed textbook example
(Wikipedia's Cohen's kappa worked example: 50 items, 25/25 rater-A marginal split, `po=0.70`,
`pe=0.50`, `kappa=0.40`) to verify the wrapper is correct, not real project data. The real number —
inter-judge κ between Gemini `gemini-3.5-flash` and `openai/gpt-oss-120b` scoring the same N=30
generated answers — comes from Unit 4.

## 6. What surprised me / what to watch for

Nothing surprising in the utility itself — the interesting part of this decision was upstream of the
code (the 2026-09-25 scope change: judge-vs-judge instead of judge-vs-hand-label, see progress.md's
Decision Log). Worth flagging forward: `raw_agreement_rate` and `cohen_kappa` will very likely diverge
more than usual once real data exists, *because* both judges are LLMs rather than one LLM and one human
— two LLM judges scoring the same rubric plausibly share more systematic bias with each other (verbosity
bias, a shared tendency toward the same "safe middle" score) than an LLM and a human would, which could
inflate raw agreement without kappa (which corrects for exactly this kind of shared-category-skew
agreement) moving as much. That's a real risk to watch for in Unit 4, not a proven finding — flagging it
now since it's a consequence of the 2026-09-25 scope change, not something this file's own numbers show.

## 7. Where this fits and what's next

Unit 3 (LLM-as-judge, built twice — Gemini and `openai/gpt-oss-120b`) is next. It produces the two label
sequences this utility consumes. Unit 4 (N=30 batch run) is where `cohen_kappa` actually gets called
against real project data and DoD tracker item #8 (κ ≥ 0.6) gets a real answer.
