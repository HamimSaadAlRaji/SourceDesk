# 14d — Conversation Memory (history-aware rewriting in `/chat`)

**Phase:** 14 · **Date:** 2026-10-01 · **Files:** `src/aegis/pipeline.py`, `src/aegis/api/app.py`, `src/aegis/query/rewrite.py`, `src/aegis/api/store.py`, `src/aegis/observability/audit.py`, `src/aegis/config.py`, `experiments/14b_multiturn_real.py`, `tests/test_pipeline_history.py`
**Topics from the course:** Query parsing / rewriting (Module 3); Latency vs. response quality (Module 5, #35)

## 1. What I built

`POST /chat` accepts an optional `history` (`[{role, text}, ...]`). When present, Phase 8's `QueryRewriter` turns the follow-up into a standalone question as the **first** pipeline stage; the cache, router, retriever and confidence scorer all see the standalone text. The `done` event carries `rewritten_query` so a UI can show what was actually searched. The rewrite is a `rewrite` span in the trace, and the audit trail reports it.

## 2. Why this approach

- **Rewrite before the cache.** A cache key of "what about the free tier?" would serve one conversation's answer to another. Keyed on the standalone question, it is a normal key. The router also needs the resolved text: a pronoun-only follow-up is unroutable. Tests spy on the cache, router and retriever and pin that they all receive the standalone text (`test_cache_router_and_retriever_all_see_the_standalone_query`).
- **No history → no rewriter call, no `rewrite` stage.** First turns pay nothing. An unchanged rewrite is not reported as one.
- **The client holds the history; the server stays stateless.** The history only feeds the rewriter for that caller's own question, so a forged history can at worst mis-rewrite their own query.
- **A failed or blank rewrite falls back to the raw query, and now logs it** (`rewrite.api_error`, `rewrite.blank_output`). Phase 8's rewriter swallowed Groq errors silently, the same pattern that hid the router outage in Phase 13.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Server-side sessions keyed by `conversation_id` | Server controls the history; can't be forged | Needs TTL/eviction/storage for a single-process demo, and the forgery risk is confined to the caller's own query. Revisit if history ever feeds anything shared (e.g. the cache write is keyed on the *rewritten* text, which is the one place a forged history could matter — see §6). |
| Rewrite after cache lookup, only on a miss | Saves an LLM call on hits | Cross-conversation cache contamination; routing on raw pronouns. |
| Always run the rewriter, even with no history | Simpler code | An LLM call on every first turn for zero benefit (856 ms measured on a real rewrite). |
| Send the full history to the rewriter | No truncation judgment | Unbounded prompt cost on a path that already has an LLM call. Window = last 6 messages, each cut to 1,000 chars — guard rails, **not tuned**. |

## 4. How it works

`_run` runs `_rewrite` (timed as stage `rewrite`, own span with input `{query, history}` and output `{rewritten_query, changed}`) before `_answer`. The effective `Query` carries the standalone text; `PipelineResult.rewritten_query` is set only if it differs from the user's text. The root span's input keeps the user's original, so the audit trail shows both. `InteractionStore` records both (`query_text`, `rewritten_query`), so the feedback → golden-candidate loop can use the standalone form.

## 5. What the numbers said

`uv run pytest -q`: **539 passed** (+8 pipeline-history, +5 API, +1 store, +1 rewriter log, +1 audit; RED first).

**Real HTTP, one conversation:** with history `[user: "How do I declare a query parameter in FastAPI?", assistant: ...]`, the follow-up "what about making it required?" became "How do I make a query parameter required in FastAPI?", the system answered, and the `rewrite` stage took **856 ms**. The same follow-up with no history **escalated** (nothing to resolve "it" against).

**End-to-end retrieval check, n=6** (`experiments/14b_multiturn_real.py`, real router/retriever/cross-encoder/generation; items sampled from the 24-item synthetic set; raw `docs/results/multiturn-real-raw.json`). A = follow-up alone, B = history of one short user turn (Phase 8's regime), C = history including the system's real answer (the server's regime):

| | any ground-truth chunk in top-10 | mean recall@10 | escalated |
|---|---|---|---|
| A: no history | 2/6 | 0.044 | 4/6 |
| B: user turn | 1/6 | 0.035 | 1/6 |
| C: user turn + real answer | 0/6 | 0.000 | 1/6 |

## 6. What surprised me / what to watch for

- **This does not demonstrate that follow-ups "resolve correctly" by retrieval.** Hit rates are *not* better with history (0–1/6 vs 2/6) and n=6 cannot separate them from noise. This is not a contradiction of Phase 8's +52.7% NDCG@5 (24 items, offline, regime B only); it is a smaller, different, noisier measurement. What it does not do is confirm the claim for regime C, and I have not run C at scale.
- **The rewrites read as faithful** (e.g. "What is the exact CDN URL causing the Swagger UI bundle loading issue in the FastAPI documentation?"), but I only eyeballed six, and they consistently add "FastAPI". That may raise retrieval confidence on its own.
- **The behaviour change that matters for the North Star: rewriting turns abstentions into answers.** Context-free follow-ups escalated 4/6; with history only 1/6 did. That is the intended benefit, but this run cannot say whether those new answers were *right*: the ground-truth chunks were labelled for the original issue, and the retrieval metric shows no gain. Until regime C is measured on all 24 items, treat answered-after-rewrite as unverified.
- **Cache write uses the rewritten text.** A forged history could steer the rewrite and so which key an answer is cached under, for any later user who asks the same standalone question. Low risk (answers are still grounded and cited), but it is a Phase 15 item.
- **Rewrite costs a serial LLM call (856 ms measured once)** on every follow-up, before the already-slow retrieval + confidence path. Not yet counted against the p95 budget.
- **`max_history_messages` / window / per-message cut are unmeasured guard rails.**

## 7. Where this fits and what's next

Open item: run regime C on all 24 items (about 24 real generations, quota-limited) to get a real number for "follow-ups resolve". Then the static UI, which will send `history` and display `rewritten_query`; the review queue/export; rate limiting.
