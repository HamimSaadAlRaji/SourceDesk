# 16g — Proving the Gate Fails: a Deliberate Regression

**Phase:** 16 · **Date:** 2026-10-24 · **Files:** `src/aegis/embedding/encoder.py` (the regression, on branch `ci-smoke-pr`, never merged), `docs/results/ci-gate-regression-{checks,comment,run}.png`
**Topics from the course:** Evaluation strategies (system vs. component); Retrieval evaluation metrics (Recall, MRR, NDCG). This is the unit that shows *why* RAG regressions are silent: nothing here raises, nothing here fails a unit test.

## 1. What I built

No new mechanism: this unit uses the gate to catch a realistic bug. A 3-line change to the encoder caps query length at 384 tokens ("speed up encoding"). Pushed to a PR branch, it leaves unit tests, ruff and mypy green and turns `retrieval-gate` red, with the metric diff posted on the PR.

## 2. Why this regression

The plan's example is "drop reranking or break the analyzer". Neither is in this system's served path (reranking is off since Phase 7; the dense path doesn't use the analyzer), so breaking them would prove nothing. I wanted a change someone would plausibly make, that no existing test could see, and that lands near the plan's "quietly drops recall ~8%" rather than a catastrophe. The golden queries are whole GitHub issues (median 128 words, p75 292, p90 427, max 1,453), so *any* sequence cap bites; the question was which cap is subtle. I swept it locally against the restored snapshot first:

| `max_seq_length` | recall@10 | MRR | NDCG@10 |
| --- | --- | --- | --- |
| 512 (default) | 0.4372 | 0.8640 | 0.7130 |
| **384 (used)** | **0.3976** | **0.8307** | **0.6654** |
| 320 | 0.3572 | 0.8143 | 0.6183 |
| 256 | 0.3346 | 0.7674 | 0.5789 |
| 192 | 0.3275 | 0.7865 | 0.5719 |
| 128 | 0.2830 | 0.6927 | 0.4916 |
| 64 | 0.2094 | 0.5944 | 0.3783 |
| 48 | 0.1754 | 0.5053 | 0.3136 |
| 32 | 0.1336 | 0.4392 | 0.2373 |

384 is the mildest cap tried that clearly leaves every band: −9.1% relative recall locally. (Not swept: 448, 480. I did not look for the largest cap the gate would still miss.)

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Drop reranking | The plan's example | Reranking is already off in the served config (Phase 7: it made retrieval worse). Removing it changes nothing. |
| Break the analyzer | The plan's example | Not in the dense retrieval path; only BM25 uses it. |
| Swap the encoder model | One-line change | Dimension mismatch (384 vs 1,024) crashes loudly. That demonstrates error handling, not silent regression. |
| Cap at 128 or below | Unmissable | A 35%+ drop is caught by any threshold; it shows the gate works on disasters, which proves little. |
| Weaken the router prompt | Exercises the LLM tier | Costs spend and needs its own sweep; the router gate's failure path is already proven with a synthetic missed billing question (16e). |

## 4. How it works

```python
model = SentenceTransformer(model_name, device=device)
model.max_seq_length = 384  # cap sequence length to speed up encoding
_MODEL_CACHE[key] = model
```

Queries longer than 384 tokens are truncated before embedding. Nothing raises; the vectors are valid, just describing less of the query. No unit test encodes a long real query, so none notices.

## 5. What the numbers said

**Prediction, written before the push** (from the local sweep): the runner should show recall@10 ≈ 0.404, MRR ≈ 0.83, NDCG@10 ≈ 0.67, failing all three bands, with `test.yml` green.

**Result** (run 40076908719, PR #1, commit `b886a9d`, runner baseline in the first column):

| Metric | Baseline | Current | Δ | Band | Status |
| --- | --- | --- | --- | --- | --- |
| recall@10 | 0.4435 | 0.3902 | −0.0534 (−12.0% relative) | ±0.0250 | ❌ FAIL |
| MRR | 0.8640 | 0.8307 | −0.0333 | ±0.0100 | ❌ FAIL |
| NDCG@10 | 0.7188 | 0.6590 | −0.0599 | ±0.0200 | ❌ FAIL |

The router gate stayed green (accuracy 0.7759, billing 1.0 / 1.0; it has nothing to do with the encoder), `report` succeeded in 13 s despite the failure, and both `test` checks (pull_request and push) passed in 2 m. The failing step exited with code 1. The earlier sticky comment on the PR was **edited in place** ("commented 1 hour ago · edited"), not duplicated, and its footer shows commit `b886a9d`.

**Prediction vs. actual:** the prediction was right about the outcome and off on two numbers. MRR matched exactly (0.8307). Recall was 0.014 lower than predicted (0.3902 vs ≈0.404) and NDCG about 0.01 lower (0.6590 vs ≈0.67). The local-to-runner offset is of the size seen in 16d (about 0.012) and here it pushed the same way on both metrics; I did not isolate it further. Every verdict was unaffected: each metric missed its band by at least 2× (recall by 2.1×, MRR by 3.3×, NDCG by 3.0×).

**Screenshots:** `docs/results/ci-gate-regression-checks.png` (one failing check, four successful), `…-comment.png` (the red table), `…-run.png` (the step log ending in exit code 1).

**Recovery** (run 40084720757, PR head `d5a1c84`, the revert): all three jobs green; the same single comment was edited back to "✅ All evaluation gates passed" (still one comment on the PR). Its numbers were recall@10 0.4317 (Δ −0.0118), MRR 0.8674 (Δ +0.0033), NDCG@10 0.7100 (Δ −0.0089): not the +0.0000 of earlier green runs, because that run's hardware returned the M1's numbers rather than the runner baseline's. All within band; see 16h. Screenshot: `ci-gate-recovered-comment.png`.

## 6. What surprised me / what to watch for

- **The regression is invisible to everything except this gate**: 797 unit tests, ruff and mypy pass with it applied. This is the argument for the phase in one commit.
- **Truncation hurts a lot more than I expected.** A cap at 384 tokens, only slightly below the default 512, loses 9–12% of recall; 128 loses 35%. The reason is the golden set: the queries are issue bodies, so the discriminating text is spread through them. A production traffic mix of short questions would be far less sensitive; this set measures sensitivity to long inputs.
- **The recall band (±0.025) caught a −0.053 drop with room to spare, but the gate's own sensitivity floor is the band**: a regression that costs 0.02 recall would pass. That is the price of a band set from noise (16c), and a weaker regression such as a 448-token cap is exactly what could slip through. Not tested.
- **Only the retrieval tier was exercised**; the router gate's failure path is proven only with a synthetic input (16e).

## 7. Where this fits and what's next

Exit criterion 2 ("deliberate regression demonstrably fails the gate, screenshotted") is met by the red run. Recovery (revert → comment returns to green) was verified in run 40084720757. Phase 16's other build items still open: the generation-subset gate, the escalation check (CPU-bound), and the p95 latency gate.
