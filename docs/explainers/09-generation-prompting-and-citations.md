# 09 — Grounded Generation & Citations

**Phase:** 9 · **Date:** 2026-08-16
**Files:** `src/aegis/generation/budgeter.py`, `src/aegis/generation/prompts/`, `src/aegis/generation/llm/`, `src/aegis/generation/pipeline.py`, `src/aegis/guardrails/citations.py`, `experiments/09a_sampling_sweep.py`, `experiments/09b_llm_bakeoff.py`, `experiments/09c_prompt_v2_check.py`
**Topics from the course:** How LLMs work (tokens, autoregression) — topic 3; Sampling strategies — topic 21; Choosing an LLM & benchmarks — topic 22; Prompt engineering — topic 23; Advanced prompting & reasoning (context pruning) — topic 24; Handling hallucinations — topic 25

## 1. What I built

The first phase that calls an LLM to *answer* rather than classify/rewrite/expand. Five pieces, composed into one `GenerationPipeline.answer(query, retrieved)` call: a **context budgeter** that fits ranked retrieval results into a fixed token budget and places them to fight the "lost in the middle" effect; **versioned prompt templates** (`v1`, `v2`) that demand numbered inline citations, permission to say "I don't know," and a hard no-policy-commitments rule; a **multi-provider streaming LLM client** (Groq primary, Gemini secondary, narrow fallback boundary) — `google-genai` added as a new dependency this phase; and a **two-tier citation validator** (cheap lexical overlap, then LLM entailment only on claims tier 1 can't confidently pass) that turns every generated answer into a per-claim support verdict.

All six of plan.md's build items now have real code and real measurement behind them, including one closed-loop iteration: prompt v1's real measured citation-coverage gap led to a v2 fix, which was itself measured (after two quota-blocked attempts) and found to win on 6/6 real matched comparisons — now the shipped default (§5). One item is still genuinely incomplete: the full sampling-sweep temperature x top_p grid (one config measured in full, the rest blocked by Groq's daily quota, not yet revisited). The provider bake-off completed with a caveat on sample size (§5).

## 2. Why this approach

Every design choice here traces back to the North Star: prevent a confident, uncited, unescalated wrong answer. Citations aren't a nice-to-have UI feature — they're the mechanism that makes an answer checkable at all, which is the one thing the Phase 0 baseline (and Air Canada's real system) didn't have. The two-tier validator exists because checking citations with an LLM call on every claim would be slow and expensive at the volumes this system is meant to handle; cheap lexical overlap resolves the easy majority for free (measured: 73.3% of non-missing claims in the one real sweep run, see §5), and the LLM only gets called on the genuinely ambiguous cases.

The budgeter's bookend ordering is a direct response to a real, well-documented transformer behavior (attention degrades in the middle of long contexts) rather than a defensive guess — plan.md names it explicitly ("relevance degrades in the middle of long contexts"), and Phase 7's reranking work already established that this corpus's chunks vary enormously in size (up to 6,325 tokens on oversized atomic code blocks), so a real budgeter has to handle truncation, not just selection.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ----------------- |
| Skip-ahead ("bin-packing") context selection — let a smaller, lower-ranked chunk fill space a bigger higher-ranked one couldn't | Uses the token budget more fully; no wasted headroom | Would put a rank-7 chunk in the prompt while dropping rank-2 just because rank-2 was bigger — relevance order should govern *selection*, not just placement. Strict-prefix packing (stop at the first chunk that doesn't fit) is simpler to reason about and never produces that inversion. |
| One-tier citation validation (LLM entailment only) | Simpler code, one code path | plan.md is explicit: "cost control matters." Every claim needing an LLM call at generation-serving volume is a real latency/cost tax the lexical tier avoids for the 73% of claims it can resolve cheaply. |
| `eval/auto_label.py`'s substring-containment overlap check (`token in text`), reused as-is | Already exists, already used for a conceptually similar purpose (Phase 4 ground-truth labeling) | TDD caught it producing false-positive matches (a token like `"on"` matching inside an unrelated word like `"injection"`) before it shipped — see §6. Reimplemented as exact word-boundary token-set intersection instead. |
| Citation numbers track display (bookended) position | Simpler — number = position, no indirection | Would make "[3]" mean different things depending on how many chunks were retrieved that call, since the bookend algorithm's third-position slot holds different ranks depending on list length. Numbers tracking rank instead means "[1]" always means "the single most relevant retrieved chunk," which is what a citation consumer (a person, or the validator) actually wants to trust. |
| A shared mutable `self.last_provider_used` field on `FallbackLLMClient` for observability | Simplest way to expose which provider answered | Phase 6 already found and fixed a real thread-safety bug from exactly this pattern (`EmbeddingCache`'s single shared `sqlite3.Connection`). `generate_with_provider()` returns `(text, provider_name)` from a stateless internal generator instead — no shared state to race on if this client is ever called concurrently (Phase 14). |
| Broad mid-stream fallback (retry with the next provider on *any* failure, even after partial output) | Higher availability — one provider dropping mid-response doesn't fail the whole request | Splicing a second provider's fresh generation onto a first provider's partial output would produce an answer no single model actually wrote — a citation could end up describing text a *different* model generated. The fallback boundary is narrow on purpose: only pre-first-token failures trigger fallback; a mid-stream failure propagates. |

## 4. How it works

### Context budgeting: strict-prefix selection, then bookend placement

```python
def pack_context(retrieved: list[Retrieved], budget_tokens: int) -> PackedContext:
    ranked = sorted(retrieved, key=lambda r: r.rank)
    selected: list[PackedChunk] = []
    used_tokens = 0

    for item in ranked:
        remaining = budget_tokens - used_tokens
        cost = count_tokens(item.chunk.text)

        if cost <= remaining:
            selected.append(PackedChunk(citation_id=len(selected) + 1, chunk=item.chunk, score=item.score, token_count=cost))
            used_tokens += cost
            continue

        if not selected and remaining > 0:
            # The #1 chunk alone exceeds the whole budget -- truncate, don't drop.
            truncated_text = _truncate_to_budget(item.chunk.text, remaining)
            ...

        break  # strict prefix: a lower-ranked chunk never displaces a higher-ranked one
```

Selection is a single pass, in rank order, with one deliberate exception (truncate an oversized #1 result rather than return an empty context). Citation numbers are assigned during *this* pass (`citation_id=len(selected) + 1`), so they always track rank. Placement is a separate step:

```python
def _bookend_order(items: list[PackedChunk]) -> list[PackedChunk]:
    n = len(items)
    ordered = [None] * n
    left, right = 0, n - 1
    for i, item in enumerate(items):
        if i % 2 == 0:
            ordered[left] = item; left += 1
        else:
            ordered[right] = item; right -= 1
    return ordered
```

For 5 survivors ranked `[1,2,3,4,5]`, this produces display order `[1,3,5,4,2]` — rank 1 opens the context, rank 2 closes it, and rank 5 (the weakest survivor) lands dead center, the position long-context attention serves worst.

### Two-tier citation validation

```python
overlap = _lexical_overlap(sentence, cited_text)   # exact word-set intersection, see §6
if overlap >= self._threshold:                     # 0.4, see §5
    status = SUPPORTED_LEXICAL                      # free
else:
    tier2_calls += 1
    supported = self._entailment_fn(sentence, cited_text)   # llama-3.1-8b-instant, JSON mode
    status = SUPPORTED_ENTAILMENT if supported else UNSUPPORTED
```

A sentence is classified `MISSING_CITATION` (no `[n]` marker) or `INVALID_CITATION` (marker points at a citation number not present in the context actually shown to the model) before either tier runs — both are free, structural checks. `EXEMPT` sentences (short connective lead-ins, e.g. "Here's how to do that:") skip validation entirely and don't count in `valid_citation_rate`'s denominator.

### Fallback with a narrow failure boundary

```python
def _stream_with_provider(self, messages, temperature, top_p, max_tokens):
    for provider in self.providers:
        gen = provider.stream(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        try:
            first = next(gen)
        except Exception as exc:
            attempts.append((provider.name, exc)); continue   # fall back
        yield provider.name, first
        for token in gen:
            yield provider.name, token   # any exception here propagates -- no more fallback
        return
    raise AllProvidersFailedError(attempts)
```

Only a failure *before* the first token triggers fallback. A failure after (a genuine mid-stream drop) propagates to the caller instead of silently switching providers.

## 5. What the numbers said

**Real end-to-end smoke test** (real Qdrant retrieval, real Groq generation, real citation validation, one FastAPI question about custom exception handlers): 6 claims produced, 0 `MISSING_CITATION`, 0 `INVALID_CITATION` — every sentence that needed a citation had one, syntactically. 4 of 6 needed tier 2 (entailment); one was correctly flagged `UNSUPPORTED` — the model had invented example code not literally present in the retrieved context, and the validator caught it (see §6 for *why* it caught it).

**Sampling sweep** (plan.md step 5), real Groq `llama-3.3-70b-versatile`, N=20 dev-positive queries, temperature=0.0, top_p=0.5:

| metric | value |
|---|---|
| mean `valid_citation_rate` (faithfulness proxy) | 0.6634 |
| mean latency | 15,522 ms |
| claim breakdown (171 non-exempt claims) | supported_lexical=99 (57.9%), supported_entailment=23 (13.5%), unsupported=13 (7.6%), missing_citation=36 (21.1%) |

Two things this number says directly: (1) the two-tier design works as intended — 99/135 (73.3%) of claims that passed validation resolved at tier 1, for free; (2) prompt v1 does **not** meet the exit criterion "100% of factual claims carry a parseable citation" — 21.1% of real claims had no citation marker at all. This is a real, measured gap, not assumed met.

The rest of the planned 6-config x 20-query grid is **not measured**. Groq's free-tier daily token quota for `llama-3.3-70b-versatile` (100,000 tokens/day) was fully consumed by this one config plus earlier smoke-testing in the same session — a real ~4,800 tokens/query cost at `max_tokens=1024`. A scoped-down retry (4 configs x 6 queries, `max_tokens=400`) found the quota already at `100,000/100,000` and completed only 2 of 24 planned calls. Full detail, including the exact error messages and the two attempts' numbers, in `docs/results/sampling-sweep.md`.

**Provider bake-off** (plan.md step 6): run for real once `GEMINI_API_KEY` landed mid-session. Groq's daily quota was still mostly exhausted from the sampling sweep, so the two providers weren't evenly sampled: Groq succeeded on 2/15 queries (13 real `RateLimitError`s — an availability failure, not a quality one), Gemini succeeded on 12/15 (3 transient `503 UNAVAILABLE` "high demand" errors). Full table and the two same-query head-to-head comparisons in `docs/results/llm-bakeoff.md`. The load-bearing finding wasn't citation quality — it was latency: Gemini's mean (21,437ms) was ~14x Groq's (1,549ms), which is the real, measured justification for "Groq primary, Gemini secondary" rather than the reverse; every single Gemini call in this run exceeded the project's 3-second latency budget on its own.

**Two real Gemini integration bugs found and fixed before the bake-off could run at all:**
1. `gemini-2.5-flash` (the originally configured model, confirmed present in `client.models.list()`) returned a real 404 on an actual call: `"This model models/gemini-2.5-flash is no longer available to new users."` The model-list endpoint doesn't reflect real per-account availability. Fixed by testing real candidate models directly against a real call (not just checking the list) and pinning to `gemini-3.5-flash`, the first one that both appeared live and actually answered.
2. `gemini-3.5-flash` is an extended-thinking model by default: a real call at `max_tokens=20` came back with `finish_reason=MAX_TOKENS` and empty visible text — the entire token budget was spent on hidden `thought` tokens before any answer text. Fixed with `thinking_config=types.ThinkingConfig(thinking_budget=0)`.

**Prompt v2** was written to target the diagnosed opening-sentence citation gap (explicit first-sentence rule + a worked example, `generation/prompts/v2.py`). Validating it took three attempts across two providers, each hitting a real, different quota wall, before landing real data:

1. Against Gemini (Groq had no headroom at the time): 2/10 v1, 0/10 v2, stopped by Gemini's free-tier limit of **20 requests/day/model** (not token-based, unlike Groq) — already nearly spent by this session's earlier Gemini calls.
2. Against Groq, a session later: 1/8 v1, 0/8 v2 — Groq's daily quota was still exhausted hours after the first wall, indicating a genuine once-a-day reset rather than fast trickle-refill.
3. Against Groq again, once its quota had actually reset: **complete** — v1 finished all 8 sampled queries, v2 completed 6 before a later quota wall. The 6 queries both versions answered give a clean matched-pair comparison.

| | v1 | v2 |
|---|---|---|
| mean `valid_citation_rate` (N=6, matched) | 0.4199 | 0.5641 |
| wins / ties / losses (v2 vs. v1) | — | 5 / 1 / 0 |

+34.3% relative improvement, v2 never worse than v1 on any of the 6 real queries. The one tie is a query where the retrieved context genuinely doesn't answer the question (`issue::11104`, python-multipart 0.0.7) — both versions correctly scored 0.0 rather than fabricating. Confirmed the mechanism on real text, not just the aggregate: for `issue::2689`, v1's answer opens "The issue you're describing is related to redundant validation when using `response_model` in FastAPI." (no citation); v2's answer to the identical question opens "The issue you're encountering is related to redundant validation in FastAPI when using `response_model` and a Pydantic model with validators **[4]**." (cited) — exactly the pattern v2 was built to fix. Full detail: `docs/results/prompt-v1-vs-v2.md`.

**`GenerationSettings.prompt_version` promoted from `"v1"` to `"v2"`.** N=6 is smaller than ideal, but the decision clears CLAUDE.md's real bar (measured, not invented) rather than the informal N≥15-20 target stated earlier in this same investigation — real data, a mechanistic explanation checked against actual generated text, and zero losses across every real comparison available.

**Real fallback demonstration** (not a fake-provider unit test — an actual two-call sequence against the live Groq API): a `GroqProvider` pointed at a nonexistent model name raised a real `NotFoundError` (404, `model_not_found`) before yielding any token; `FallbackLLMClient` caught it and transparently completed the stream from a second `GroqProvider` on a different, unaffected model (`llama-3.1-8b-instant` — confirmed to have a separate quota bucket from the exhausted 70B model). 5 real chunks, joined answer: "Automatic API documentation generation." This is real infrastructure, not `tests/test_fallback_llm.py`'s fakes — the fake-provider tests prove the *logic*; this proves the logic holds against a real API's real error shape.

Full test suite: 356/356 passed (`uv run pytest -q`), up from 318 at the end of Phase 8 — 38 new tests across the budgeter, prompts, fallback client, and citation validator.

## 6. What surprised me / what to watch for

**TDD caught a real correctness bug in the citation validator's own test suite, not by inspection.** The first `_lexical_overlap` implementation reused Phase 4's `auto_label.py` technique almost verbatim: `token in candidate_text` (substring containment). A test constructing a claim about "the sky is blue" against a chunk about "dependency injection" expected `UNSUPPORTED`, but got `SUPPORTED_LEXICAL` — because the token `"on"` is a substring of `"injection"`, and `"the"`/`"is"`/`"in"` all had similar spurious hits. Fixed to exact word-boundary token-set intersection. Worth flagging because the *original* technique is still live in Phase 4's golden-set labeling — this doesn't retroactively invalidate that work (a coarser overlap notion was a defensible, documented choice there, for a different comparison at a different scale), but it's a reminder that a technique proven adequate for one purpose isn't automatically correct for a stricter one.

**The sentence splitter (borrowed from `chunking/semantic.py`, built for parsing existing prose) didn't understand LLM-generated Markdown — fixed same-day.** In the full-pipeline smoke test, one answer included a fenced code block right after a sentence ending in `:` with no following period before the block — the regex splitter glued the lead-in sentence, the entire code block, and the next real sentence into one "sentence." The claim ended up validated `UNSUPPORTED` regardless (the code was genuinely invented, not grounded), so the *outcome* was arguably still correct that one time — but for an accidental reason, not because the splitter worked as designed. Reproduced as a failing test first (`test_fenced_code_block_does_not_get_glued_to_surrounding_prose`), then fixed by reusing `chunking/segments.py`'s `split_code_segments` (Phase 2 — already hardened against CRLF and trailing-HTML fence edge cases) to strip fenced code out *before* sentence splitting, rather than inventing new fence-detection logic. Worth flagging as a small methodology note: the fix reuses existing, already-measured project code instead of writing a second fence parser, the same call Phase 1 made for canned-reply detection and Phase 2 made for unclosed-fence handling.

**Groq's free-tier daily token quota is a real, sharp production constraint, and it bit this exact phase's own experiments.** 100,000 tokens/day on the 70B model sounds generous until real ~4,000-5,000-token-per-call context-plus-completion costs are accounted for — a single 20-query experiment consumed 96% of a full day's budget. This isn't a hypothetical "what if Groq rate-limits us" risk from plan.md's own risk table anymore; it's a measured number from this session. It's also a genuinely good argument *for* the multi-provider fallback client this same phase built — not just against outages, but against a same-organization daily quota wall, which is arguably the *more* likely real-world trigger for a production system on free/cheap tiers.

**Gemini's free tier hit differently, and finding that out cost real quota.** Groq's limit is token-shaped (scales with how much you ask for); Gemini's is request-shaped (a flat 20 calls/day per model, regardless of size) — two genuinely different constraints that a system built to survive only one of them wouldn't survive the other. Discovering this required spending several of those 20 requests on smoke tests and a bake-off before the wall showed up, which is exactly why the *first* v2 prompt comparison attempt ran out of budget before producing a trustworthy result. In hindsight, checking documented free-tier limits before spending real requests on exploration would have preserved more budget for the comparison that mattered most (v1 vs. v2) — a real instance of the same "benchmark before committing" lesson Phase 7 learned about cross-encoder throughput, in a new shape (API quotas instead of compute time).

**Both quotas turned out to be genuine once-a-day resets, not fast-refilling buckets — confirmed by waiting, not assumed.** A small probe call succeeding shortly after a large-call failure (both providers, at different points) initially looked like a fast rolling-window refill; a follow-up real experiment attempt hours later hit the identical wall (`Used: 98750/100000` on Groq, still blocked on Gemini), disproving that read. Only a genuine wait — most of a day — actually restored enough headroom to complete the v1-vs-v2 comparison for real. The lesson: a single successful small probe after a failure is weak evidence about quota shape; only a second real attempt at realistic scale, later, tells you what actually happened.

**A "no longer available to new users" model can still appear in `models.list()`.** `gemini-2.5-flash` was listed as a real, selectable model, and only failed on an actual generation call. This means model-availability checks for a new integration need a real test call, not just a successful list/lookup — the same "verify against a real call, not just that the artifact exists" discipline this project already applies to its own retrieval indexes (Open Question #7: a stale cache can serve a result for a config whose underlying index no longer matches).

**The missing-citation gap (21.1%) has at least two distinct causes, not one.** Spot-checking real answer text: some missing citations are genuine prompt-following failures (an opening topic sentence stating a real claim with zero citation, when the model clearly had a supporting chunk). Others are the validator's own heuristic being too blunt — a sentence that merely restates the user's own stated problem back to them (not a new claim from context) still trips `_requires_citation`'s ">4 words, doesn't end in ':'" rule and gets flagged `MISSING_CITATION` even though it arguably shouldn't need one. Both are real; conflating them would produce a prompt-only fix (v2) that doesn't actually close the measured gap.

## 7. Where this fits and what's next

This phase's `GenerationPipeline` is the object Phase 10's abstention gate will sit in front of: the confidence scorer doesn't need to be invented from scratch — `ValidationResult.valid_citation_rate` and the per-claim status breakdown this phase already produces are exactly the kind of signal Phase 10's plan.md step 1 asks for ("inter-chunk semantic agreement," "score-distribution shape," and now, additionally, "how well-grounded was the answer that was actually produced").

Two items from this section's earlier draft are now closed: the sentence-splitter code-block blind spot (fixed same-day via TDD, §6) and the v1-vs-v2 validation (§5 — v2 won 5/6 real matched comparisons with 0 losses and is now the shipped default). One remains concretely open for a follow-up session: complete the remaining sampling-sweep temperature x top_p configs once Groq's quota allows another multi-query run — the script needs no further changes, just quota. A larger confirmatory run for v2 (N≥15-20, matching the original v1 baseline's scale) would still strengthen the evidence further, but isn't required to justify the default already shipped.

**Addendum, 2026-09-26.** `llama-3.3-70b-versatile` (the model the sampling sweep above ran against) is now
retired entirely (Open Question #15, Phase 11 Unit 1) — re-scoped `experiments/09a_sampling_sweep.py` to
`qwen/qwen3.8-27b` and re-ran it for real, restoring `N_QUERIES` to 20 (its per-minute quota shape affords
this, unlike the old model's once-a-day wall) and adding fixed inter-call pacing against the real ITPM cap
found in Phase 11 Unit 4. Real result: the two `temperature=0.0` configs (`top_p=0.5` and `top_p=1.0`)
completed in full and produced **identical** output — same mean `valid_citation_rate` (0.7296), same exact
claim-status counts, across all 20 real queries. This is the correct, mechanistically expected result, not a
bug: at `temperature=0.0` decoding is effectively greedy, so `top_p` has nothing left to affect — a real,
disclosed confirmation of exactly what plan.md's step 5 asked to "measure... anyway" rather than assume. The
remaining two configs (temperature 0.5 and 1.0) did not complete — `qwen/qwen3.8-27b` hit a real 200,000
tokens/day cap partway into the third config, the same real cap magnitude Unit 4 found on both `gpt-oss`
models, now confirmed on this model too. Full detail, including the two now-historical (and not directly
comparable, different-model) original attempts: `docs/results/sampling-sweep.md`.
