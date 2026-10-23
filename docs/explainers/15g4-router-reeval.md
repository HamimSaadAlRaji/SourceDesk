# 15g4 — Router Re-eval: nano Stays, Because Only the Billing Decision Is Acted On

**Phase:** 15 (provider migration, step 2) · **Date:** 2026-10-19 · **Files:** `experiments/15g4_router_reeval.py`, `docs/results/router-reeval-openai{.md,-raw.json}`
**Topics from the course:** Agentic routing / query classification (Module 3); evaluating a classifier with a confusion matrix (Module 4)

## 1. What I built

A re-evaluation of the router on Phase 8's 58 labeled dev queries (25 `docs_howto`, 25 `known_issue`, 8 `billing_account_legal`), using the served prompt and settings. Arms: **gpt-4.1-nano** (served since 15g), **gpt-4.1-mini**, and **qwen/qwen3.8-27b on Groq**. The question: does the served router still escalate every billing query, and is nano good enough?

## 2. Why this approach

- **A like-for-like baseline was missing.** Phase 8's 0.6897 was measured on `llama-3.1-8b-instant`, since retired. The Groq model that replaced it (qwen) was never scored on this set, so it got a free arm. Comparing nano with a retired model would have told us nothing about the migration.
- **Unparseable output counted separately.** Served, it falls back to `billing_account_legal` (fail safe), which would quietly raise billing recall. All three arms produced 0 unparseable outputs, so billing recall here is real classification.
- **Read the matrix against what the route actually does.** `router/policy.py` acts only on `billing_account_legal` (hard escalation). `docs_howto` vs `known_issue` changes nothing downstream today. The decision weighs the confusion cells that change behavior.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
|---|---|---|
| Switch the router to gpt-4.1-mini | +13.8 pts accuracy (0.7759 vs 0.6379), known_issue recall 0.64 vs 0.20 | That gain is entirely in the docs_howto/known_issue split, which nothing acts on. On the acted-on billing decision both are 8/8 with 0 false escalations. mini costs 4× per call ($0.000325 vs $0.0000814). Revisit if a later phase uses the route for retrieval. |
| Router back on qwen (Groq) | Best accuracy 0.7931, free | It ties mini, and on billing it ties nano. It spends the Groq daily quota that the generator fallback and second judge depend on. |
| Compare only against Phase 8's 0.6897 | Already on record | Different, retired model. Not a migration comparison. |

## 4. How it works

The labels come from `08a_router_eval.load_labeled_queries` (imported, unchanged). Each call goes through `chat_completion` with the served `SYSTEM_PROMPT`, temperature 0, JSON mode and 50-token cap, with purpose `router_eval` in the ledger. A 429 is retried. Any other API error aborts the arm rather than scoring a partial one. Significance uses an exact McNemar test on per-query correct/incorrect pairs.

## 5. What the numbers said

| Arm | Accuracy | docs_howto R / P | known_issue R / P | **billing R / P** | unparseable | cost/call |
|---|---|---|---|---|---|---|
| Phase 8 llama-3.1-8b (retired) | 0.6897 (40/58) | 0.80 / 0.65 | 0.48 / 0.75 | 1.00 / 0.73 | — | — |
| **gpt-4.1-nano (served)** | 0.6379 (37/58) | 0.96 / 0.55 | **0.20** / 0.83 | **1.00 / 1.00** | 0 | $0.0000814 |
| gpt-4.1-mini | 0.7759 (45/58) | 0.84 / 0.70 | 0.64 / 0.80 | 1.00 / 1.00 | 0 | $0.000325 |
| qwen/qwen3.8-27b | 0.7931 (46/58) | 0.84 / 0.72 | 0.68 / 0.81 | 1.00 / 1.00 | 0 | free (TPD) |

- **Billing, the acted-on decision: 8/8 recall and 0 false escalations out of 50 technical queries in every arm.** With only 8 positives, the 95% Wilson lower bound on recall is 0.68. That's a small-sample pass, not proof.
- **nano's errors are almost all known_issue → docs_howto** (it predicts docs_howto for 20 of 25 known_issue queries). Exact McNemar: mini vs nano 11 vs 3 discordant, p = 0.057; qwen vs nano 12 vs 3, p = 0.035; qwen vs mini p = 1.0.
- p50 latency: nano 715 ms, mini 784 ms. qwen's 4,569 ms includes my 2.5 s rate-limit pacing, so it isn't comparable.
- **Spend $0.023569, against my estimate of "≤ $0.02".** The estimate came from the single 420-token router call. These queries average 779 input tokens (max 2,399), because many are full GitHub-issue texts. I estimated from an unrepresentative call when I should have counted tokens on the real set, which takes seconds and costs nothing. I did exactly that for step 1 and skipped it here. Total $0.159055 of $10.

## 6. What surprised me / what to watch for

- **A worse accuracy number, and the right decision is to keep the model anyway.** nano is the least accurate arm, but on the only route the system acts on, it's perfect. The accuracy-only reading would pay 4× per query for a distinction nothing uses.
- **That holds only while docs_howto/known_issue stays decorative.** If a later phase uses the route to weight issue chunks, nano's 0.20 known_issue recall becomes a real retrieval regression, and mini (or qwen) becomes the choice. Recorded here so the switch is a one-line config change with its evidence already in place.
- **8 billing positives is a thin test of the most important route.** A larger billing/legal probe set (paraphrases, technical-sounding billing questions) would make this claim much stronger. That's future work, not built here.

## 7. Where this fits and what's next

The router stays on gpt-4.1-nano (no config change). Provider migration steps 1–2 are complete. Next is **15h, the real-stack security batch**, which runs on gpt-5-mini as generator and needs a cost estimate and the user's OK first. That estimate will be counted from the real attack payloads, not extrapolated.
