"""Wires RAGAS (plan.md Phase 11, step 1) to this project's own LLM
providers and encoder instead of RAGAS's defaults (OpenAI for both).

RAGAS's own `llm_factory` only dispatches four hardcoded provider strings
(openai, anthropic, google, litellm) to `instructor`'s adapters -- verified
by reading the installed `ragas==0.3.9` source directly, not assumed from
docs written for a different version. Neither Groq nor the new
`google-genai` SDK this project already uses (`generation/llm/
groq_provider.py`, `gemini_provider.py`) is in that list.

**Two different LLM protocols exist inside ragas 0.3.9, and they are not
interchangeable** -- found the hard way, running the real Unit 1 smoke test
(`experiments/11c_ragas_smoke_test.py`), not by reading changelogs:

- `Faithfulness`/`ResponseRelevancy` (this file's actual Unit-1 need) are
  built on `PydanticPrompt`, which requires a `BaseRagasLLM`:
  `generate_text`/`agenerate_text` taking a `PromptValue` and returning a
  langchain-core `LLMResult`, plus `is_finished`.
- `llm_factory` (and `InstructorLLM`, what this file originally built)
  returns an `InstructorBaseRagasLLM`: `generate(prompt: str,
  response_model: Type[T]) -> T`, for *structured-output* calls -- a
  different contract, not a legacy-vs-modern variant of the same one. This
  is true of `llm_factory`'s own openai/anthropic path too, not just this
  project's Groq/Gemini bypass -- confirmed by reading `llm_factory`'s
  docstring directly ("Create an LLM instance for structured output
  generation using Instructor").
- Mixing them doesn't raise a clean type error: `PydanticPrompt.
  generate_multiple`'s `is_langchain_llm()` check is `hasattr(llm,
  "agenerate") and not hasattr(llm, "run_config")`. `InstructorLLM` has
  `.agenerate` but isn't a `BaseRagasLLM` (no `run_config` field), so it
  gets misdetected as a LangChain chat model and ragas calls
  `llm.agenerate_prompt(...)` -- a method `InstructorLLM` doesn't have at
  all, `AttributeError`, not `TypeError`.

`GroqRagasLLM`/`GeminiRagasLLM` below are hand-written `BaseRagasLLM`
subclasses for `Faithfulness`/`ResponseRelevancy` specifically, wrapping the
same raw `groq`/`google-genai` SDKs already used elsewhere in this project
rather than adding `langchain-groq`/`langchain-google-genai` (two more real
dependencies) just to reuse `ragas.llms.LangchainLLMWrapper` -- the
`langchain-core` types (`LLMResult`/`ChatGeneration`/`AIMessage`) this file
needs are already a paid transitive cost of the `ragas<0.4` pin (Session Log
2026-08-18), so this is genuinely a thin shim, not new dependency surface.

`build_groq_ragas_llm`/`build_gemini_ragas_llm` (this file's Unit-1 API)
build these. The original `InstructorLLM`-based builders are kept as
`build_groq_judge_llm`/`build_gemini_judge_llm` -- wrong protocol for
Faithfulness/ResponseRelevancy, but the right one for Unit 3's rubric-based
LLM-judge (plan.md: "structured output"), which genuinely wants
`response_model`-typed calls.

The embeddings side reuses `embedding.encoder.Encoder` (Phase 3) locally --
RAGAS's `answer_relevancy`/`ResponseRelevancy` metric needs embeddings, but
the corpus's own encoder already exists, is already MPS-accelerated, and
costs no Groq/Gemini quota at all. Same two-protocol split as the LLM side
bit here too, found the same way (a real `AttributeError` from the real
smoke test, not read in advance): `ragas.embeddings` has both a "modern"
`BaseRagasEmbedding` (singular -- `embed_text`/`embed_texts`) and a legacy
`BaseRagasEmbeddings` (plural -- extends langchain-core's `Embeddings` ABC:
`embed_query`/`embed_documents`). `ResponseRelevancy` calls `.embed_query`,
i.e. it wants the legacy plural interface, not the modern one this file
originally built against. `RagasLocalEmbeddings` below implements the
plural interface; `Encoder.encode` already batches
(`EmbeddingSettings.batch_size`), so `embed_documents` is a single real
batched call, not one call per text.
"""

from __future__ import annotations

import typing as t

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import InstructorLLM
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig

from aegis.config import settings
from aegis.embedding.encoder import Encoder

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from langchain_core.prompt_values import StringPromptValue as PromptValue


class GroqRagasLLM(BaseRagasLLM):
    """`BaseRagasLLM` for `Faithfulness`/`ResponseRelevancy`. Non-streaming
    (these metrics need the full text + finish_reason at once, not tokens).

    `max_tokens` is its own field, not `settings.groq.max_tokens` -- that
    setting is tuned for `GroqSettings.model`'s (`qwen/qwen3.8-27b`) real,
    measured 1000 output-tokens/minute ceiling (Open Question #15), which
    RAGAS's own statement-generation + NLI-entailment calls can exceed on
    longer answers (real `LLMDidNotFinishException`, Unit 4 dry run) --
    see `JudgeSettings.ragas_llm_max_tokens`'s comment for the full story."""

    def __init__(self, model: str, api_key: str, max_tokens: int = settings.judge.ragas_llm_max_tokens) -> None:
        self.run_config = RunConfig()
        self.multiple_completion_supported = False
        self.cache = None
        self.model = model
        self._api_key = api_key
        self.max_tokens = max_tokens

    def is_finished(self, response: LLMResult) -> bool:
        return all(
            (g.generation_info or {}).get("finish_reason") in (None, "stop")
            for gens in response.generations
            for g in gens
        )

    def _to_result(self, texts_and_finish: list[tuple[str, str | None]]) -> LLMResult:
        generations = [
            ChatGeneration(message=AIMessage(content=text), generation_info={"finish_reason": finish})
            for text, finish in texts_and_finish
        ]
        return LLMResult(generations=[generations])

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        from groq import Groq

        client = Groq(api_key=self._api_key)
        temp = temperature if temperature is not None else self.get_temperature(n)
        out = []
        for _ in range(n):
            completion = client.chat.completions.create(
                model=self.model,
                temperature=temp,
                stop=stop,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt.to_string()}],
            )
            choice = completion.choices[0]
            out.append((choice.message.content or "", choice.finish_reason))
        return self._to_result(out)

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=self._api_key)
        temp = temperature if temperature is not None else self.get_temperature(n)
        out = []
        for _ in range(n):
            completion = await client.chat.completions.create(
                model=self.model,
                temperature=temp,
                stop=stop,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt.to_string()}],
            )
            choice = completion.choices[0]
            out.append((choice.message.content or "", choice.finish_reason))
        return self._to_result(out)


class GeminiRagasLLM(BaseRagasLLM):
    """`BaseRagasLLM` for `Faithfulness`/`ResponseRelevancy`, Gemini side.
    `thinking_budget=0` for the same real reason as `generation/llm/
    gemini_provider.py`: the default model spends its whole budget on hidden
    `thought` tokens otherwise (docs/explainers/09-generation-prompting-and-
    citations.md §6)."""

    def __init__(self, model: str, api_key: str) -> None:
        self.run_config = RunConfig()
        self.multiple_completion_supported = False
        self.cache = None
        self.model = model
        self._api_key = api_key

    def is_finished(self, response: LLMResult) -> bool:
        return all(
            (g.generation_info or {}).get("finish_reason") in (None, "STOP")
            for gens in response.generations
            for g in gens
        )

    def _config(self, temperature: float):
        from google.genai import types

        return types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=settings.gemini.max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        from google import genai

        client = genai.Client(api_key=self._api_key)
        temp = temperature if temperature is not None else self.get_temperature(n)
        out = []
        for _ in range(n):
            response = client.models.generate_content(model=self.model, contents=prompt.to_string(), config=self._config(temp))
            out.append((response.text or "", response.candidates[0].finish_reason.name if response.candidates else None))
        return LLMResult(
            generations=[
                [ChatGeneration(message=AIMessage(content=text), generation_info={"finish_reason": finish}) for text, finish in out]
            ]
        )

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        from google import genai

        client = genai.Client(api_key=self._api_key)
        temp = temperature if temperature is not None else self.get_temperature(n)
        out = []
        for _ in range(n):
            response = await client.aio.models.generate_content(model=self.model, contents=prompt.to_string(), config=self._config(temp))
            out.append((response.text or "", response.candidates[0].finish_reason.name if response.candidates else None))
        return LLMResult(
            generations=[
                [ChatGeneration(message=AIMessage(content=text), generation_info={"finish_reason": finish}) for text, finish in out]
            ]
        )


def build_groq_ragas_llm(model: str = settings.judge.ragas_llm_model, api_key: str = settings.groq.api_key) -> GroqRagasLLM:
    return GroqRagasLLM(model=model, api_key=api_key)


def build_gemini_ragas_llm(model: str = settings.gemini.model, api_key: str = settings.gemini.api_key) -> GeminiRagasLLM:
    return GeminiRagasLLM(model=model, api_key=api_key)


def build_groq_judge_llm(model: str = settings.groq.model, api_key: str = settings.groq.api_key) -> InstructorLLM:
    """Structured-output (`response_model`) protocol -- for Unit 3's
    rubric-based LLM-judge, not for Faithfulness/ResponseRelevancy (see
    module docstring)."""
    import instructor
    from groq import AsyncGroq

    client = instructor.from_groq(AsyncGroq(api_key=api_key))
    return InstructorLLM(client=client, model=model, provider="groq")


def build_gemini_judge_llm(model: str = settings.gemini.model, api_key: str = settings.gemini.api_key) -> InstructorLLM:
    """Structured-output (`response_model`) protocol -- for Unit 3's
    rubric-based LLM-judge, not for Faithfulness/ResponseRelevancy (see
    module docstring)."""
    import instructor
    from google import genai

    client = instructor.from_genai(genai.Client(api_key=api_key), use_async=True, model=model)
    return InstructorLLM(client=client, model=model, provider="genai")


class RagasLocalEmbeddings(BaseRagasEmbeddings):
    """Adapts `Encoder` (sync, batched, numpy) to RAGAS's legacy
    `Embeddings`-derived interface (`embed_query`/`embed_documents`, plus
    async variants) -- the one `ResponseRelevancy` actually calls, not the
    modern `BaseRagasEmbedding` singular class this originally subclassed
    (see module docstring). No real async work happens (the encoder is
    CPU/MPS-bound, not I/O-bound) -- the async methods just call the sync
    path directly, same as RAGAS's own non-OpenAI embedding wrappers do
    when their underlying client has no native async form."""

    def __init__(self, encoder: Encoder) -> None:
        self.run_config = RunConfig()
        self.cache = None
        self.encoder = encoder

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.encoder.encode(texts).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)
