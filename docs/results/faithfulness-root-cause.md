# RAGAS Faithfulness Root Cause (Phase 11)

Why is real RAGAS `faithfulness` (mean 0.7977, n=20/29, `docs/results/judge-agreement.md`) below plan.md's
0.90 target? Zero new LLM calls — pure analysis of Unit 4's real batch data plus manual inspection of the
lowest-scoring real answers against their real retrieved context. Reproducible:
`experiments/11h_faithfulness_root_cause.py`.

## Step 1: is faithfulness tracking a real, independent groundedness signal?

Correlated the 20 real faithfulness scores against the same rows' `CitationValidator.valid_citation_rate`
— a cheaper, independently-computed groundedness proxy from the same generated answers.

**Pearson r = 0.3083.** Weak-to-moderate positive correlation, not zero and not strong — faithfulness is
measuring something related to citation validity, but not the same thing, and not perfectly. That gap is
exactly where the two real mechanisms below live.

## Two distinct, real mechanisms behind the low mean — not one story

### Mechanism 1: a genuine hallucination RAGAS caught and the other two guardrails missed

**`golden::issue::219`** — the single lowest real score (faithfulness=0.077) — is **not** an honest
refusal. It's a confident, detailed, plausible-sounding answer about fixing a real `mypy` type error with
FastAPI file uploads, recommending `Annotated[UploadFile, File()]`. The answer cites chunk `[2]` as its
primary source four times. **Chunk `[2]` is a completely unrelated GitHub issue** about receiving files in
dynamically-named form fields — it never mentions `mypy`, `Annotated`, or type-checking at all.

This is a real, confident, wrong-citation case — exactly the failure mode this entire project exists to
catch (the Air Canada North Star). And it was caught cleanly by only one of this project's three
groundedness signals:

| signal | result on this answer |
|---|---|
| RAGAS `faithfulness` | **0.077 — correctly flagged as almost entirely unsupported** |
| `CitationValidator.valid_citation_rate` | 0.667 — partially caught (tier-1 lexical overlap between "UploadFile"/"File" tokens in the answer and unrelated chunks likely inflated this) |
| LLM judge (`gpt-oss-120b`, pre-fix) | **5/5 — completely missed it**, scored the answer as ideal helpfulness with no grounding check |

The technical content is real, correct FastAPI/mypy advice in general — it just isn't supported by *this
context*, which is exactly the distinction faithfulness exists to enforce. This case alone argues the
0.7977 mean isn't just metric noise — real, low scores are catching real, consequential problems the
project's other two guardrails did not.

### Mechanism 2: RAGAS's NLI-style check structurally can't validate honest refusals

**`golden::issue::15764`** (faithfulness=0.250) is a well-behaved, honest, *verified-accurate* answer: it
says the context lacks specific information about a FastAPI 0.137.0 type-checker regression, then adds one
careful positive claim — that `route.tags` is a real, valid attribute used in the documented
`generate_unique_id_function` pattern. Checked directly against the real cited context: **chunk `[1]`
literally shows `route.tags` being used** (`"Tag" in route.tags`), and chunk `[3]` documents
`generate_unique_id_function` using tags — the positive claim is genuinely, verifiably supported.

Yet this answer scores only 0.25. The mechanism: RAGAS's `Faithfulness` extracts claims from the answer and
checks each one for NLI-style entailment against the retrieved context. A claim like *"the context does not
contain information about X"* has no way to be entailed by a context passage — the passage simply doesn't
discuss X, and passages don't explicitly assert their own topical absence. **The metric has no mechanism to
credit a correct negative/absence claim**, so every honest refusal is structurally penalized regardless of
how accurate it is. This is the same *category* of problem Unit 4 already diagnosed in the LLM judge
(penalizing honest refusals) — but a different, metric-level mechanism, not a prompt-wording problem, and
not something a rubric fix can touch.

## What this means for the 0.7977 mean and the 0.90 target

The real mean understates true faithfulness specifically on refusal answers (a metric limitation, not a
grounding failure), while correctly, sharply flagging at least one real, serious hallucination the project's
other two guardrails missed (a real, working signal, not noise). Both are true at once — the low mean is
not evidence the generation pipeline is broadly ungrounded, and it is not evidence the metric is useless
either.

**Not a project bug to fix.** RAGAS's `Faithfulness` implementation is upstream, imported code (`ragas`
Build-vs-Import ledger, Phase 11) — this is a documented, real limitation of NLI-based faithfulness metrics
applied to a corpus and prompt design (Phase 9's "say so if you don't know") that deliberately produces
negative-claim answers RAGAS wasn't built to score well. Two honest options for the future, neither pursued
this session: (a) exclude or separately report refusal-answer faithfulness scores, since they're measuring
the metric's blind spot rather than answer quality; (b) accept the real, lower measured mean as a disclosed,
explained limitation rather than a target to chase, since chasing it upward with prompt changes could
mean *removing* honest refusals to raise the number — optimizing against this project's own North Star.

## What's next

This satisfies plan.md's Phase 11 exit-criterion alternative ("a written analysis of why not, and what you
tried") for the faithfulness target — real investigation, not just a flagged number. Recommend closing this
out as a disclosed, explained limitation rather than pursuing a fix, given the mechanism (an upstream
metric's structural blind spot) isn't something this project's own prompt or model choices can correct
without trading away the exact behavior (honest refusal) the confidence scorer (Phase 10) exists to produce.
