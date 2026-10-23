# 15k — Claims Split on Paragraphs and List Items (Open Question #18)

**Phase:** 15 · **Date:** 2026-10-22 · **Files:** `src/aegis/guardrails/citations.py` (`_prose_blocks`, `_iter_claim_sentences`), `tests/test_citations.py` (+4), `experiments/15k_rescore_claim_split.py`, `docs/results/claim-split-rescore-raw.json`
**Topics from the course:** Faithfulness and citation grounding (Module 4); evaluating the evaluator

## 1. What I built

The citation validator now ends a claim at a **paragraph break** and at the start of a **list item** (`-`, `*`, `+`, `•`, `1.`, `1)`), as well as at sentence ends. Before, a bulleted answer became a few very long "claims" (up to 1,718 characters in 15h) that passed or failed as one unit on word overlap. The decision-relevant stored answers were re-scored under the fixed validator.

## 2. Why this approach

This was found in 15h: v3's apparent citation gain (+0.113) was about 40% claim granularity, because v3 answers carry twice as many bullet lines. The flaw is also a guardrail weakness, not just a measurement one: one supported bullet's overlap could carry an unsupported bullet in the same "claim" to `SUPPORTED_LEXICAL`. The new test pins it: "FastAPI automatically encrypts every outgoing database connection [1]" is `UNSUPPORTED` on its own line, where before it could hide behind its neighbour.

**A plain newline is deliberately *not* a boundary.** A hard-wrapped sentence would split, and its first half would become an uncited fragment scored `missing_citation`. List markers and blank lines are always structural. The 15h sensitivity check split on *every* line; this is the stricter-to-justify, safer version.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
|---|---|---|
| Split on every newline (15h's sensitivity patch) | Simplest | It breaks hard-wrapped sentences into uncited fragments (pinned by `test_hard_wrapped_sentence_stays_one_claim`). |
| Statement-level decomposition by an LLM (RAGAS-style) | Finest granularity | It's an LLM call per answer inside the served validator, adding latency and cost, and RAGAS already does this in eval (Phase 11). The validator must stay deterministic and cheap. |
| Leave it, report the caveat | No change to stored numbers | The flaw lets an unsupported bullet pass, so it's a correctness bug in a guardrail, not only a metric caveat. |

## 4. How it works

```python
for paragraph in re.split(r"\n\s*\n", text):          # blank line = boundary
    for line in paragraph.splitlines():
        if starts_list_item(line) and current_item:   # "- ", "* ", "1. ", "1) " = boundary
            yield " ".join(current_item); current_item = []
        current_item.append(line_without_marker)      # plain newline: same claim
    yield " ".join(current_item)
# each block -> split_sentences(block), as before; fenced code still skipped
```

## 5. What the numbers said

Re-score of stored answers, with no generation and the same contexts and judges. **Spend $0.006259** (nano-judge entailment). The old numbers stay in their original raw files.

| Answer set (prompt) | qwen judge old → new | nano judge old → new | claims old → new |
|---|---|---|---|
| qwen (v2) | 0.7296 → **0.7539** | 0.6329 → 0.6471 | 111 → 124 |
| gpt-4.1-mini (v2) | 0.4486 → 0.4441 | 0.4361 → 0.4372 | 138 → 138 |
| gpt-4o-mini (v2) | 0.3458 → 0.3490 | 0.3264 → 0.3303 | 126 → 121 |
| gpt-4.1-nano (v2) | 0.6665 → 0.6665 | 0.6288 → 0.6288 | 129 → 132 |
| **gpt-5-mini (v2)** | 0.7158 → **0.7217** | 0.6242 → 0.6383 | 59 → 76 |
| gpt-5-nano (v2) | 0.4217 → 0.4514 | 0.3717 → 0.4047 | 79 → 124 |
| **gpt-5-mini (v3, served)** | 0.8292 → **0.7842** | 0.6875 → 0.7009 | 34 → 79 |

**Both decisions that rested on the old splitter hold:**
- gpt-5-mini vs qwen (generator, 15g3): −0.032, CI [−0.160, +0.084] (qwen judge) and −0.009 [−0.163, +0.141] (nano judge), still a tie.
- v3 vs v2 (served prompt, 15h rule (b)): +0.063, CI [−0.048, +0.172] and +0.063 [−0.052, +0.167]. The upper bound is ≥ 0, so the rule still passes.

`uv run pytest -q`: 726 passed after this unit (+4 tests, three red first; the hard-wrap test passed before and after, as a guard against over-splitting).

## 6. What surprised me / what to watch for

- **Models that don't use lists barely move** (gpt-4.1-mini, 4o-mini, 4.1-nano within ±0.005). The bug was specific to list-heavy answers, and gpt-5-mini under v3 is the most list-heavy configuration measured, which is why it surfaced only there.
- **gpt-5-mini still has fewer claims than qwen** (76 vs 124), so its long-sentence style remains a coarser measurement. Statement-level faithfulness (RAGAS, step 4) is still the real check.
- **Phase 9–11 numbers computed before this fix aren't re-scored.** They're historical, from an older splitter, and say so. The qwen 0.7296 baseline is now 0.7539 under the current validator.

## 7. Where this fits and what's next

This closes Open Question #18. Every citation rate from here on uses the fixed splitter. Next is #20, the validator certifying poisoned advice (design record 15l).
