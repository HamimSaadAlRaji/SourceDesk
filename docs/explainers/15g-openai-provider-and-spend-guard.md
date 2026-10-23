# 15g — OpenAI as the Primary Provider, Behind a Hard $10 Spend Guard

**Phase:** 15 (step 0 of the provider migration) · **Date:** 2026-10-18 · **Files:** `src/aegis/cost/{ledger,__main__}.py`, `src/aegis/generation/llm/{openai_provider,chat,__init__}.py`, `src/aegis/router/classifier.py`, `src/aegis/query/rewrite.py`, `src/aegis/guardrails/citations.py`, `src/aegis/factory.py`, `src/aegis/config.py` (`OpenAISettings`, `*.provider`, `rewrite_max_tokens`), `pyproject.toml` (`openai>=2.54,<3`), `tests/{conftest,test_spend_ledger,test_openai_provider,test_structured_logging}.py`
**Topics from the course:** What makes production challenging — cost (Module 5, #29); multi-provider fallback (Phase 9)

## 1. What I built

OpenAI is now first in the generation fallback chain (**openai → groq → gemini**) and the default provider for the router, rewriter and citation-entailment calls. Each role's model is in `OpenAISettings`, and Groq is one setting away for any role.

Every OpenAI call goes through a sqlite **spend ledger** that enforces the user's **hard $10 cap for the whole project**. Before the call, it reserves a worst-case cost and refuses if that doesn't fit. After the call, it records the billed usage. `uv run python -m aegis.cost` prints the running total.

Nothing has been measured on the new models yet. Steps 1–2 (generation bake-off, router re-eval) do that before any number is trusted.

## 2. Why this approach

**Why move to a paid provider.** The free tiers set the pace of the whole project. Groq's 200k tokens/day and 7k tokens/minute, and Gemini's 20 requests/day, have deferred measurements by days at a time (Phase 9's sweep, Phase 11's κ, 15g). The user chose OpenAI for most calls, and kept Groq/Gemini as fallbacks and as independent second raters (a judge from a different model family from the generator; plan in `progress.md`, 2026-10-18).

**Why a guard in code, not just the billing dashboard.** A dashboard limit acts after spend has happened, and only at account level. The user's constraint is per project and absolute: "maximum 10 usd … no more than that". So the guard runs *before* each request:

```
worst_case = cost(model, input_tokens = counted prompt tokens, output_tokens = max_tokens)
refuse if   spent + live_reservations + worst_case > budget      (one IMMEDIATE sqlite transaction)
```

`max_tokens` (`max_completion_tokens` for reasoning models, which caps hidden reasoning plus the answer) makes the worst case a true upper bound. Because reservations live in sqlite, not memory, the server and an experiment running at the same time can't jointly overshoot (tested: 20 concurrent threads, exactly 5 admitted for 5 slots).

**Hitting the cap degrades; it doesn't cause an outage.** A refusal raises `BudgetExceededError` before the first token. `FallbackLLMClient` already treats anything before the first token as a provider failure, so generation falls through to free Groq/Gemini (tested). Short calls surface it as `LLMCallError`, so the router escalates with `provider_unavailable`, the rewriter uses the raw query, and entailment fails closed. These are the same paths 14f built for outages.

**One dispatcher for short calls.** The router, rewriter and entailment each had their own hard-coded Groq SDK block. `chat_completion(provider, model, …)` replaces all three and maps every provider failure to one exception type, so each caller keeps exactly one fail-safe branch.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
|---|---|---|
| Rely on the OpenAI dashboard spend limit only | No code | It acts after the fact, per account, and can't tell this project from other use of the key. It's kept as an outer layer (recommended to the user), not the guard. |
| Count spend after each call (no reservation) | Simpler | Concurrent calls can each pass the check and jointly overshoot. The reservation makes "never above $10" hold by construction, not by luck. |
| LiteLLM / a provider-abstraction library | One API for every vendor, built-in cost tracking | One more dependency to wrap three providers this project already wraps in ~40 lines each. Its cost tracking is post-hoc and can't refuse a call beforehand. CLAUDE.md §2 would need it to buy something; it doesn't. |
| Replace Groq/Gemini entirely | Simpler config | It loses the free fallback when the paid budget runs out, and the cross-family second judge Phase 11 needs. |
| Infer the provider only from config | Explicit | Experiments that pin `qwen/qwen3.8-27b` would be sent to OpenAI under a Groq model name. `resolve_role` infers Groq for namespaced model names. Every Groq model here has a `/` and no OpenAI model does. |

## 4. How it works

- **`SpendLedger`** has two tables: `spend` (settled calls: model, purpose, input/cached/output tokens, cost) and `reservations` (in flight). `cost()` raises `KeyError` for an unpriced model, so an unknown model is never treated as free. Prices are in `OpenAISettings.prices`, taken from the official pricing page (fetched 2026-10-18), with cached input discounted. Reservations older than 10 minutes are ignored, so a crashed process can't lock the budget.
- **`OpenAIProvider.stream`:** reserve, then stream with `stream_options={"include_usage": True}` (usage arrives in the final chunk), then settle. If the call fails before any output, the reservation is released (nothing billed). If a stream is cut after some output, it settles a conservative estimate rather than nothing. Reasoning models (`gpt-5*`, `o*`) get `max_completion_tokens` + `reasoning_effort` and no temperature/top_p, which they reject.
- **Token counting for the reservation** uses tiktoken's `o200k_base` (the gpt-4.1/4o/5 tokenizer), plus OpenAI's documented per-message overhead.
- **`rewrite_max_tokens = 200`** is new. The Groq rewriter sent no output cap at all, and the ledger needs one to reserve a worst case. It's a guard rail: a standalone question is one sentence.

## 5. What the numbers said

`uv run pytest -q`: **717 passed, 1 xfailed** (was 699 + 1). That's +9 ledger, +8 provider/dispatch, +1 OpenAI-path logging test.

**Real spend so far: $0.000091, 2 calls, both accidental, both recorded by the ledger** (§6). Each was 420 input + 9 output tokens on `gpt-4.1-nano` for the router. As a side effect, that's the first real end-to-end proof: OpenAI answered, `usage` was read, and the cost settled correctly ($0.0000456/call = 420 × $0.10/1M + 9 × $0.40/1M).

**Not measured yet: any quality, latency or cost number for the new models.** Step 1 (generation bake-off, ~$0.10) and step 2 (router re-eval on the 58-query routing set, ~$0.01) come next. The provisional role assignments in `OpenAISettings` are the user-approved plan, not results.

## 6. What surprised me / what to watch for

- **The first test run after the switch made a real paid call.** Two logging tests mocked only the Groq SDK. With OpenAI now the default router provider, `RouterClassifier()` built a real OpenAI client and called it. It happened once in the full run and once more when I re-ran the failing test alone to debug, before I'd understood why. Cost: $0.000091. The fix is suite-wide, in `tests/conftest.py`: every test gets an OpenAI client factory that raises and a throwaway ledger. A test that forgets to inject a fake now fails loudly instead of spending. A full suite run after the guard left the real ledger at exactly 2 calls (checked by timestamp: both before the guard file existed). Lesson: changing a *default provider* is a change to every test that relied on the old default without saying so.
- **The ledger only sees calls made through this code.** RAGAS (via LangChain) and any Batch API job create their own clients, so they must be wired through the ledger before they run on OpenAI (step 4). Otherwise their spend is invisible to the guard. That has to be done before the first RAGAS-on-OpenAI run, and it's flagged here so it can't be forgotten.
- **A crash between "provider billed" and "settle" loses one record.** It's bounded by that one call's reservation. The recommended dashboard limit is the backstop for that gap.
- **The served defaults changed before measurement.** That's deliberate, because nothing is served or measured until steps 1–2 run, but the qwen-era numbers (citation rate 0.7296, faithfulness 0.7977, p95 5.6 s) no longer describe the default configuration. They stay in the record as the "qwen era" column, not overwritten.

## 7. Where this fits and what's next

1. **Step 1:** generation bake-off on the 20 Phase 9 queries (gpt-4.1-mini, gpt-4o-mini, gpt-4.1-nano), with prompt v2.
2. **Step 2:** router re-eval, nano vs mini.
3. **Step 3 (record 15h):** the real-stack security batch.
4. **Step 4:** re-measure the published numbers. This needs the RAGAS-through-the-ledger wiring first.

Each step reports its estimated and actual spend against the $10 cap.
