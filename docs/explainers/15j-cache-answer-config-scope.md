# 15j — Cache Scope Carries the Answer Configuration (Open Question #19)

**Phase:** 15 · **Date:** 2026-10-22 · **Files:** `src/aegis/generation/pipeline.py` (`GenerationPipeline.cache_fingerprint`), `src/aegis/pipeline.py` (`AnswerPipeline._cache_version`, 4 cache call sites), `tests/test_pipeline.py` (+3)
**Topics from the course:** Caching (Module 5, #33); what makes production challenging: configuration drift

## 1. What I built

Both answer caches are now scoped to `corpus_version :: prompt=<version> :: llm=<provider:model,...>`, not just the corpus version. Changing the prompt or the generator chain makes every cached answer unreachable, so the next request is generated and validated under the new configuration. The served scope is `prompt=v3::llm=openai:gpt-5-mini,groq:qwen/qwen3.8-27b,gemini:gemini-3.5-flash`.

## 2. Why this approach

A cache hit skips routing, the confidence gate, generation and citation validation (Phase 12/13 design). Before this fix, the scope was the corpus version plus the partition, so the two configuration changes made in 15g3 (qwen → gpt-5-mini) and 15h (prompt v2 → v3, a **security** fix) were invisible to the cache. Any question answered before the switch would keep getting the old answer, including one shaped by a v2-forgeable context block. This was found while designing 15h, which needed one cache directory per prompt version to avoid exactly this.

The fingerprint is read **per request** (a property, not a constructor-time string), because the prompt can be swapped on a live pipeline. 15h does that, and so does the new test.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
|---|---|---|
| A new required `answer_config=` keyword on both caches (like 15b's `partition=`) | Explicit at the cache API | It changes ~25 call sites in tests and experiments for no extra safety. The pipeline is the only production caller, and it already passes a version string, so folding the fingerprint into it changes 4 lines. |
| Bump `corpus_version` by hand on every prompt/model change | No code | It relies on someone remembering. 15g3 and 15h both changed configuration and neither bumped it, which is the bug. |
| Prompt version only | Smallest fix for the issue as filed | The generator switch has the same staleness. Answers from a different model were never validated under the current configuration. |
| Include sampling params / max_tokens | More complete | They're unchanged across every served configuration so far. The fingerprint covers what has actually changed; extending it is one line if those start moving. |

## 4. How it works

```python
# GenerationPipeline
@property
def cache_fingerprint(self) -> str:
    chain = ",".join(f"{p.name}:{p.model}" for p in self.llm.providers)   # FallbackLLMClient order
    return f"prompt={self.prompt.version}::llm={chain}"

# AnswerPipeline: all four exact/semantic get/set calls use
self._cache_version == f"{self.corpus_version}::{self.generation.cache_fingerprint}"
```

The cache classes are unchanged, and the scope string is hashed into the key as before (exact tier) or stored as the scope column (semantic tier).

## 5. What the numbers said

No measurement needed; it's a correctness fix, verified by tests:
- an answer cached under v2 is a **miss** under v3, and the new answer is generated;
- an answer cached under a qwen chain is a **miss** under a gpt-5-mini chain;
- the same configuration still **hits** (no regression of the cache itself).

The tests were written first and failed for the right reason: the second pipeline was served the first one's answer. `uv run pytest -q`: **722 passed**. Spend: $0.

## 6. What surprised me / what to watch for

- **Every existing cache entry is now unreachable** (the key changed), the same one-time effect as 15b. Correct here: none of them were produced by the served configuration.
- **A fallback answer is cached under the full chain, not the provider that answered.** If OpenAI fails and qwen answers, that answer is cached under the gpt-5-mini-first fingerprint and replayed later even after OpenAI recovers. That's acceptable (it was validated when generated) but worth knowing; `CachedAnswer.provider` still records who wrote it.
- The fingerprint uses `name`/`model` attributes. A provider without them shows its class name or `?`, which still differs between configurations but less readably.

## 7. Where this fits and what's next

This closes Open Question #19. Next is #18 (claim splitting on bullets and paragraphs), which changes how every citation rate is computed, then #20 (the validator certifying poisoned advice).
