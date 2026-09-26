# Groq Model Replacement — `openai/gpt-oss-120b` vs. `qwen/qwen3.8-27b`

Open Question #15 (progress.md). Groq retired `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`
outright between 2026-08-17 and 2026-08-18, breaking every default in `config.py`. This session
(2026-09-25) measured the two live candidates at real answer length — not the trivial `max_tokens=10`
probe used to first flag the problem — before deciding a replacement.

Both runs: same 6 real dev-positive golden queries (`random_seed=20260802`), same real dense retrieval
(Phase 6 winning config), same real v2 prompt, same real `GenerationSettings.context_budget_tokens=4096`
input budget. Non-streaming, so `usage.completion_tokens_details` is readable.
Scripts: `experiments/11a_gpt_oss_reasoning_tax.py`, `experiments/11b_qwen_reasoning_check.py`.

## Results

| model | reasoning_effort | max_tokens | N succeeded | mean reasoning_tokens | mean completion_tokens | mean latency (ms) | latency range (ms) |
|---|---|---|---|---|---|---|---|
| `openai/gpt-oss-120b` | `"low"` | 1024 | 6/6 | 67.7 | 282.5 | 16,697 | 1,186 – 29,044 |
| `qwen/qwen3.8-27b` | (none — not reasoning-first) | 1024 | 4/6 (2 real `RateLimitError`) | 0 | 180.5 (of the 4 that succeeded) | 2,564 (of the 4) | 653 – 7,857 |
| `qwen/qwen3.8-27b` | (none) | **1000** | 6/6 (2 retried) | 0 | 254.7 | 8,692 | 653 – 32,309 |

## Finding 1: gpt-oss-120b's reasoning tax is real but not fatal — latency is the actual problem

Every one of the 6 real calls finished (`finish_reason="stop"`) with visible text — unlike the earlier
trivial `max_tokens=10` probe, which came back completely empty. At real answer length, 25–45% of
`completion_tokens` per query went to hidden `reasoning_tokens` (range: 18/52=34.6% to 129/579=22.3%),
consistently leaving room for a real, cited answer.

The disqualifying number is latency: **mean 16.7s, up to 29.0s**. That's close to Gemini's
already-measured, already-rejected-as-too-slow 21.4s mean (`docs/results/llm-bakeoff.md`) — the entire
premise of "Groq primary, Gemini secondary" is Groq's speed advantage, and gpt-oss-120b doesn't have one.

## Finding 2: `groq/compound-mini` (last session's verified small-task pick) died too

A live `client.models.list()` re-check this session found `groq/compound-mini` — verified clean for the
four small-task config roles just five weeks earlier (2026-08-18) — now 404s. Groq's free-tier catalog
churns faster than one session's verification survives; see design record 11 §6.

## Finding 3: `qwen/qwen3.8-27b` (renamed from the rejected `qwen3.6-27b`) is clean, with one real constraint

`qwen3.6-27b` was rejected last session for being reasoning-first (a trivial call burned its whole
budget on hidden tokens). Groq's live catalog now lists `qwen/qwen3.8-27b` instead — a renamed
successor, not verified as behaving the same way. It doesn't: **0 reasoning tokens across all 6 real
generations** at real answer length.

2 of the first 6 calls hit a real `RateLimitError`: a **1000 output-tokens/minute** free-tier cap, one
token under this project's `max_tokens=1024` default. Confirmed the fix directly — retried both failed
queries at `max_tokens=1000`, both succeeded (`completion_tokens` 426 and 380). Full 6-query set at 1000:
0 reasoning tokens throughout, mean latency 8.7s (real variance: 653ms–32.3s, not a tight band).

## Decision

Presented to the user as a 3-way tradeoff (`AskUserQuestion`): `qwen/qwen3.8-27b` for all five
Groq-backed config roles, `openai/gpt-oss-120b` for primary only (still needing a separate small-task
model given `compound-mini`'s death), or reconsidering Gemini as primary. **User chose
`qwen/qwen3.8-27b` for everything** — zero reasoning tax beats gpt-oss-120b's real-but-survivable tax,
8.7s mean latency is ~2.5x faster than Gemini's 21.4s (preserving the "Groq primary because it's fast"
rationale, just at a slower absolute speed than the dead 70B model's 1.5s), and one verified model
covering all five roles is simpler than pairing two independently-verified models — especially with one
of the two already dead again this session.

Applied to `config.py`: `GroqSettings.model`, `RouterSettings.model`,
`QuerySettings.rewrite_model`/`expansion_model`/`hyde_model`, `CitationSettings.entailment_model` →
`qwen/qwen3.8-27b`; `GroqSettings.max_tokens` 1024 → 1000. Verified through the real
`GroqProvider`+`GenerationPipeline` path (not just this raw-SDK measurement): 1,125ms latency,
`valid_citation_rate=1.0` on a real golden query. `uv run pytest -q`: 373/373 passed.

## What wasn't chased further

- Latency variance within qwen3.8-27b itself (653ms to 32.3s across 6 calls) isn't explained — could be
  per-query answer length, could be Groq's shared-tenancy queueing on the free tier. Worth watching in
  Unit 4's N=30 batch; not re-derived here since the go/no-go decision didn't hinge on it.
- No further flagship-class Groq models were tried beyond the two already investigated
  (`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`) — the live catalog (`client.models.list()`, checked this
  session) has no other general-purpose chat model besides `allam-2-7b` (Arabic-focused, 4K context) and
  the `openai/gpt-oss-*` family.
