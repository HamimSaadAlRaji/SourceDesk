# 16e — The LLM-Backed Gate: Router, Model Probe, and a Model That Disappeared

**Phase:** 16 · **Date:** 2026-10-24 · **Files:** `src/aegis/eval/router_gate.py`, `src/aegis/eval/model_probe.py`, `src/aegis/eval/gate.py` (`compare_files`, `write_baseline`), `tests/test_router_gate.py`, `tests/test_model_probe.py`, `ci/baselines/router.json`, `llm-gate` job in `.github/workflows/eval.yml`, `OpenAISettings` role models in `src/aegis/config.py`
**Topics from the course:** Evaluation strategies (code-based vs. LLM-judge; system vs. component); Handling hallucinations (the billing/legal route is the hard-escalation guard).

## 1. What I built

A second job in `eval.yml` that makes real OpenAI calls: a model-availability probe (one tiny call per configured model) and the router eval (58 labeled dev queries through the served `RouterClassifier`), compared to `ci/baselines/router.json`. A ledger cap of **$0.15 per run** is set by environment variable. The job runs on same-repo PRs and manual dispatch only; never on push-to-main (merges must not spend) and never on fork PRs (secrets are withheld there).

Building it surfaced a production outage: **`gpt-4.1-nano` has been removed from the OpenAI API.** The router, query-rewrite and citation-entailment roles all used it, so on 2026-10-24 every query would have escalated as `provider_unavailable`. The served models for those three roles are now `gpt-4.1-mini`.

## 2. Why this approach

- **Gate the safety metric with zero tolerance.** `billing_recall` (8 billing/legal queries that must reach a human) has a band of 0.0. Anything else is a quality metric with a measured noise band.
- **Score billing precision too.** `RouterClassifier` maps unparseable model output to `billing_account_legal` as a fail-safe, which *raises* billing recall. Precision exposes that inflation (4 unparseable queries on a mixed set drop precision to 0.25 while recall reads 1.0; pinned in a test).
- **An outage must not be scored as a prediction.** A provider error raises and fails the job instead of being folded into a route. Proven by the budget test in §5.
- **A real call is the only availability check.** `/v1/models` still listed `gpt-4.1-nano` while calls to it returned 404. The probe costs ~$0.00001 per distinct model.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Check `/v1/models` for availability | Free | Listed the dead model as available. |
| Keep the router on `gpt-5-nano` (cheapest) | $0.0027 per 58 queries, accuracy 0.7759 | 13.6 s worst-case latency in 12 calls (reasoning model); that tail lands on a p95 budget that is already unmet. |
| `gpt-4o-mini` | Cheapest stable option, $0.0071 per 58, p50 719 ms | One query less accurate (0.7586), older model line likelier to be retired next. Reasonable; not chosen. |
| **`gpt-4.1-mini` (chosen, user decision)** | Accuracy 0.7759, billing 8/8 at precision 1.0, p50 724 ms, max 1.8 s | $0.0189 per 58 (about $0.0003 per routed query), the dearest of the three; still small beside generation. |
| Run LLM checks on every push to a branch | Faster feedback | Spend scales with pushes. PRs and manual dispatch bound it. |
| Percent tolerance on accuracy | Scales | 58 queries means one flip is 0.0172; the band is stated in queries. |

## 4. How it works

`score_routes` returns accuracy, per-route recall, billing recall and billing precision; an undefined metric (no support) is `NaN`, which `evaluate` treats as a failure. `query_set_fingerprint` hashes the question set and labels, not the router under test, so a prompt or model change moves the metrics, not the fingerprint. The CI cap is `OPENAI_BUDGET_USD=0.15` with a per-run ledger at `results/ci_spend.sqlite` (uploaded as an artifact, and its summary is appended to the job summary); the ledger checks each call's worst case before the provider is contacted.

## 5. What the numbers said

Router eval, 58 labeled dev queries (25 docs_howto, 25 known_issue, 8 billing), prompt unchanged, temperature 0:

| Model | Accuracy | Billing recall / precision | Cost per 58 | p50 / max latency (12 calls) |
| --- | --- | --- | --- | --- |
| gpt-4.1-nano | 0.6379 (15g4) | 8/8 | — | **removed from the API** |
| gpt-4.1-mini | 0.7759 | 1.0 / 1.0 | $0.01885 | 724 / 1,767 ms |
| gpt-4o-mini | 0.7586 | 1.0 / 1.0 | $0.00707 | 719 / 783 ms |
| gpt-5-nano | 0.7759 | 1.0 / 1.0 | $0.00266 | 1,165 / 13,642 ms |

**Repeatability of the chosen model:** three runs gave accuracy **0.7759, 0.7586, 0.7759** and known_issue recall 0.64, 0.60, 0.64: one query flipped between runs at temperature 0. Billing recall and precision were 1.0 in all of them (and in 15g4's run). Bands, by the same rule as retrieval (2× the largest benign delta, rounded up to 0.005): **accuracy 0.035** (two flipped queries pass, three fail), **billing precision 0.05** (one extra false billing escalation, a 0.111 drop, fails), **billing recall 0.0**. Baseline: the modal run (0.7759; billing 1.0/1.0), `ci/baselines/router.json`.

Verified locally: the flipped run (0.7586) **passes** (Δ −0.0172); a synthetic missed billing question (recall 0.875) **fails**, exit 1. The model probe returned ok for both distinct served models (`gpt-5-mini`, `gpt-4.1-mini`). **The budget cap is real:** run against a deliberate $0.002 cap, the gate made 7 calls, spent $0.001534, then the ledger raised `BudgetExceededError`, the gate failed, and no results file was written.

Spend this unit: **$0.0801** on the project ledger (three candidate-model runs $0.0286, served-model repeat runs including one that failed on a connection error, probes) plus **$0.0015** on the throwaway ledger of the cap test. Project total ≈ **$0.432 of $10**.

**First CI run of the LLM tier** (run 40068902808, `workflow_dispatch` on main, 2026-10-24): both jobs green. `llm-gate` took 1 m 16 s; probe ok for `gpt-5-mini` and `gpt-4.1-mini`; router accuracy 0.7759, billing recall 1.0000, billing precision 1.0000: identical to the baseline. **Real cost $0.01536** (60 calls, 45,221 input / 501 output tokens; router $0.015314, probes $0.000047), about 20% under the local estimate of $0.019, presumably prompt caching on the long fixed router prompt (not verified). CI spend lives in each run's `results/ci_spend.sqlite` artifact and is **not** in the project's local ledger; project total ≈ $0.431 local + $0.015 CI = **$0.446 of $10**. The same run's `retrieval-gate` reproduced its runner-taken baseline exactly (Δ +0.0000 on recall@10, MRR, NDCG@10), which looked like it closed 16d's open question. **Corrected 2026-10-24 (16h):** three consecutive runs agreed; a later run did not (it matched the M1 instead), so this is "repeatable within a hardware class", not "repeatable".

## 6. What surprised me / what to watch for

- **A model can vanish between phases, silently, while still being listed.** No test or earlier gate could see it; an LLM-calling system needs a canary on every model it names. The probe is that canary.
- **`temperature=0` is not determinism** (one query flipped across identical runs). Gate bands for LLM metrics must be measured, not assumed zero.
- **A transient `APIConnectionError` hit one run** (server disconnected after the SDK's retries). The gate fails rather than scoring a partial run, which is correct, but it means a CI run can go red for an infrastructure reason; if that recurs the fix is a workflow-level retry, not a wider band.
- **Rewrite and entailment are unmeasured on `gpt-4.1-mini`.** Only the router role has evidence. They moved to the larger sibling of the model they were built on, which should not hurt, but the claim is "unmeasured", not "equivalent". Phase 11/19 should re-measure entailment (it feeds the citation validity numbers).
- **Historical results still say `gpt-4.1-nano`** (15g, 15g1-15g4 records, `v2c.py` prompt note). They describe what was measured then; the config comment records the swap.
- **Not gated yet:** generation quality (citation validity, code-block syntax) and the escalation confusion matrix. The latter needs the cross-encoder on CPU: 4.09 s per query on the M1 (736 s for 180 queries), and the runner encodes ~10× slower than the M1, so a full pass is on the order of two hours there. It cannot run on every PR; see §7.

- **A manual dispatch cancelled the push-to-main run** because both shared the concurrency group `eval-<ref>`. The group now includes the event name.

## 7. Where this fits and what's next

The LLM tier now guards the router, the Air Canada route in particular, and every configured model's availability. Remaining Phase 16 work, in order: (1) generation gate on a fixed-context subset (citation validity, code syntax; gpt-5-mini + entailment, estimated $0.05–0.10 per run, to be measured first); (2) the escalation check as a path-filtered/manual job, since its cost is CPU time, not dollars; (3) the p95 latency gate; (4) results as a PR comment; (5) the deliberate regression, screenshotted. Needs from the user: the `OPENAI_API_KEY` repository secret.
