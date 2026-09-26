# 11 — Groq Model Replacement & RAGAS Protocol Wiring

**Phase:** 11 · **Date:** 2026-09-25 · **Files:** `src/aegis/config.py`, `src/aegis/eval/ragas_adapters.py`,
`experiments/11a_gpt_oss_reasoning_tax.py`, `experiments/11b_qwen_reasoning_check.py`,
`experiments/11c_ragas_smoke_test.py`
**Topics from the course:** Choosing an LLM & reading benchmarks; Advanced prompting & reasoning
(reasoning models); Evaluating the LLM's performance (RAGAS: Response Relevancy, Faithfulness)

## 1. What I built

Two things, both prerequisites for Phase 11 Unit 1 rather than Unit 1 itself: (1) a real, measured
replacement for every dead Groq model default in `config.py` (`llama-3.3-70b-versatile`/
`llama-3.1-8b-instant`, retired by Groq between sessions), and (2) a working `BaseRagasLLM`/
`BaseRagasEmbeddings` bridge between this project's Groq/Gemini clients and RAGAS's
`Faithfulness`/`ResponseRelevancy` metrics, replacing an adapter that looked complete last session but
had never actually been exercised end-to-end. Output: `qwen/qwen3.8-27b` in every Groq-backed config
role, and a real RAGAS smoke test producing real scores (`docs/results/ragas-smoke-test.md`).

## 2. Why this approach

**Model replacement.** The constraint wasn't "find *a* replacement" — last session already had one
candidate (`openai/gpt-oss-120b`) and a trivial-call red flag (empty output, reasoning tokens ate the
whole `max_tokens=10` budget). The constraint was CLAUDE.md's own rule: don't trust a number you didn't
measure at the shape that matters. A trivial `max_tokens=10` call says nothing about whether a
reasoning-first model is usable for ~200-600 token real answers — reasoning tokens could plausibly be a
fixed small overhead that vanishes proportionally at real length, or a per-token-of-output tax that scales
with it. Only measuring at real length resolves which. It turned out to be closer to a fixed cost
(25-45% of completion_tokens, roughly stable across very different completion lengths from 52 to 625
tokens) — survivable — but that same real-length test surfaced the actual disqualifying number
(16.7s mean latency) that the trivial call couldn't have shown at all, since `max_tokens=10` finishes in
under a second regardless of provider.

**RAGAS wiring.** `ragas_adapters.py` was written last session against `ragas`'s *newest* interface
(`llm_factory`/`InstructorLLM`, `BaseRagasEmbedding`) because that's what a fresh read of the installed
source recommended as "the" adapter surface. What wasn't visible without actually running a metric is
that `Faithfulness`/`ResponseRelevancy` — the two metrics this project actually needs first — are built
on an older, still-live `PydanticPrompt` protocol that never migrated to the new LLM interface. Both
protocols are real, both are documented, and nothing in either class's docstring says "the other one
won't work with these metrics." Only a real call surfaces that.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ----------------- |
| Trust the trivial-call reasoning-tax result and pick Gemini as primary instead of measuring gpt-oss-120b further | Faster to decide; Gemini's numbers were already fully measured (Phase 9 bake-off) | The trivial call couldn't distinguish "always empty" from "empty only below some token floor" — real-length measurement was cheap (6 real calls, ~2 min) and changed the actual reason for rejection from "broken" to "slow," a materially different, more defensible claim for the design record |
| Add `langchain-groq` + `langchain-google-genai`, use `ragas.llms.LangchainLLMWrapper` (the officially-documented non-OpenAI path) | Zero hand-written adapter code; `LangchainLLMWrapper`'s `generate_text`/`agenerate_text`/`is_finished` are already correct and tested by ragas itself | Two more real dependencies for what's fundamentally a ~15-line shim once the real contract (`PromptValue.to_string()` in, `LLMResult` out) is known. `langchain-core` (the types actually needed: `LLMResult`, `ChatGeneration`, `AIMessage`) was already a transitive cost of the `ragas<0.4` pin — the chat-model packages weren't |
| Patch `is_langchain_llm`'s `hasattr` check (e.g. monkey-patch `InstructorLLM` to expose a dummy `run_config`) so it stops misrouting to the LangChain branch | Smallest possible diff; keeps last session's `InstructorLLM`-based adapter unchanged | Wouldn't actually work — even if the routing bug were patched, `ragas_llm.generate(prompt_value, n=n, ...)` would then call `InstructorLLM.generate(prompt: str, response_model: Type)` positionally, which has a completely different signature (no `n`/`temperature`/`stop`, and a required `response_model` with nothing to pass). The routing bug was masking a deeper, real contract mismatch, not causing one on its own |
| Delete the `InstructorLLM`-based builders entirely as dead code | Nothing in Unit 1 calls them; keeping unused code contradicts "no half-finished implementations" | They're not speculative — plan.md's Phase 11 step 2 explicitly wants "rubric-based, structured output" for the Unit 3 LLM-judge, which is exactly `InstructorLLM.generate(prompt, response_model=Rubric)`'s actual contract. Renamed to `build_groq_judge_llm`/`build_gemini_judge_llm` with a docstring correction instead of deleting scoped-but-not-yet-started work |

## 4. How it works

**The two-protocol split, concretely.** `ragas==0.3.9` carries two independent LLM abstractions:

```python
# "Modern" — what llm_factory returns. Structured output only.
class InstructorBaseRagasLLM(ABC):
    def generate(self, prompt: str, response_model: Type[T]) -> T: ...
    async def agenerate(self, prompt: str, response_model: Type[T]) -> T: ...

# "Legacy" but still what PydanticPrompt-based metrics require.
class BaseRagasLLM(ABC):
    def generate_text(self, prompt: PromptValue, n=1, temperature=0.01, stop=None, callbacks=None) -> LLMResult: ...
    async def agenerate_text(self, ...) -> LLMResult: ...
    def is_finished(self, response: LLMResult) -> bool: ...
```

`Faithfulness`/`ResponseRelevancy` call `PydanticPrompt.generate_multiple`, which picks a branch via:

```python
def is_langchain_llm(llm) -> bool:
    return hasattr(llm, "agenerate") and not hasattr(llm, "run_config")
```

`InstructorLLM` has `.agenerate` (the structured-output one) but is not a `BaseRagasLLM`, so it has no
`run_config` — the check misfires, ragas treats it as a LangChain chat model, and calls
`llm.agenerate_prompt(...)`, a method that exists on LangChain models and nowhere on `InstructorLLM`.
Real symptom: `AttributeError: 'InstructorLLM' object has no attribute 'agenerate_prompt'`.

**The fix** (`GroqRagasLLM`/`GeminiRagasLLM` in `ragas_adapters.py`) implements the *legacy* contract
directly against the raw `groq`/`google-genai` SDKs already used by `generation/llm/groq_provider.py` /
`gemini_provider.py`:

```python
def generate_text(self, prompt: PromptValue, n=1, temperature=0.01, stop=None, callbacks=None) -> LLMResult:
    client = Groq(api_key=self._api_key)
    out = []
    for _ in range(n):
        completion = client.chat.completions.create(
            model=self.model, temperature=temperature, stop=stop,
            max_tokens=settings.groq.max_tokens,
            messages=[{"role": "user", "content": prompt.to_string()}],  # ragas already rendered the full prompt text
        )
        choice = completion.choices[0]
        out.append((choice.message.content or "", choice.finish_reason))
    return self._to_result(out)  # wraps as LLMResult(generations=[[ChatGeneration(...), ...]])
```

`prompt.to_string()` matters: `PromptValue` here is `langchain_core.prompt_values.StringPromptValue` —
ragas has already fully rendered the NLI-statement/entailment/question-generation instructions plus
few-shot examples plus the actual claim/context into one string before this method ever sees it. The
adapter's only job is "send this string to the chat API, hand back the text and finish reason."

The embeddings side has the identical split: `ResponseRelevancy` calls `.embed_query()`
(`ragas.embeddings.base.BaseRagasEmbeddings`, plural — extends `langchain_core.embeddings.Embeddings`),
not `.embed_text()` (`BaseRagasEmbedding`, singular, this file's original base class). Fixed the same way:
implement the plural interface's 4 abstract methods (`embed_query`/`embed_documents`/`aembed_query`/
`aembed_documents`), still backed by Phase 3's `Encoder` at zero API cost.

## 5. What the numbers said

Full detail in `docs/results/groq-model-replacement.md` and `docs/results/ragas-smoke-test.md`. Headlines:

- `openai/gpt-oss-120b`, 6 real dev-positive queries, real answer length: 6/6 finished with visible text
  (unlike the trivial-call probe), 25-45% of `completion_tokens` spent on hidden reasoning, mean latency
  **16.7s** (1.2-29.0s).
- `groq/compound-mini` — verified clean for the small-task roles 2026-08-18 — 404s as of this session. A
  five-week-old verification didn't survive.
- `qwen/qwen3.8-27b` (renamed from the rejected `qwen3.6-27b`): **0 reasoning tokens** across the same 6
  real generations. 2/6 hit a real 429 at `max_tokens=1024` (1000 output-tokens/minute cap); fixed and
  confirmed at `max_tokens=1000`. Mean latency **8.7s** (653ms-32.3s) — real variance, not a tight band.
- RAGAS smoke test, one real query: `faithfulness=0.8750` (16.2s, 2 LLM calls), `response_relevancy=0.7820`
  (55.5s, 3 LLM calls + one-time local embedding-model load), `id_based_context_precision`/`recall` both
  `0.8000` (<1ms, no LLM).

## 6. What surprised me / what to watch for

**Groq's free-tier catalog is unstable on two independent axes, not one.** Open Question #15 originally
framed this as "Groq retired two models" — an availability problem. This session found `groq/compound-mini`
(availability: died) and `qwen3.6-27b`→`qwen3.8-27b` (behavior: same-ish name, opposite reasoning-token
behavior) in the same five-week gap. A config default verified once should not be trusted as still-true in
a later session without re-checking `client.models.list()` and re-running the real-length probe — this
now matters for Phase 16's CI gate too, not just interactive sessions.

**A library documented as "the modern interface" isn't automatically wired to every metric that predates
it.** `llm_factory`'s docstring is unambiguous about being for "structured output generation using
Instructor" — the mistake last session wasn't misreading that docstring, it was assuming a newer, more
prominently-documented interface must be *the* interface, without checking whether the specific metrics
needed (`Faithfulness`, `ResponseRelevancy`) actually consume it. They don't; they predate it and were
never migrated. Same category of lesson as the ragas 0.4.x packaging bug from last session (GitHub
#2741/#2745/#2753) — a real, load-bearing gap between a library's newest documented surface and what
its own older, still-default code paths actually call.

**The `hasattr`-based `is_langchain_llm` check turns a clean incompatibility into a confusing one.**
If ragas had type-checked `llm` against `BaseRagasLLM` directly instead of duck-typing "has `.agenerate`,
lacks `run_config`," the original mistake would have failed with an immediately diagnosable
`TypeError`/`isinstance` message instead of an `AttributeError` two method calls deep that could easily
be misread as *this project's* bug rather than a protocol mismatch.

## 7. Where this fits and what's next

Unblocks Phase 11 Units 2-6 (Cohen's κ hand-labeling, LLM-as-judge, the N=30 batch run, custom metrics,
`make eval`) and any Phase 8-10 Groq-dependent path that needed re-running after the model retirement
(none re-run yet this session — router/query-understanding/citation-entailment all now point at
`qwen/qwen3.8-27b` but haven't been re-measured against it; their Phase 8-10 numbers still reflect the
dead models). `build_groq_judge_llm`/`build_gemini_judge_llm` (the `InstructorLLM`-based builders, now
correctly scoped) are what Unit 3's rubric-based judge should build on. `GeminiRagasLLM` is written but
not yet smoke-tested with a real call — worth doing before Unit 4 needs a fallback if Groq quota runs out
mid-batch.
