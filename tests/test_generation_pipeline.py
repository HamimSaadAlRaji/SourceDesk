"""TDD for the generation pipeline (plan.md Phase 9) -- the composition of
budgeter -> prompt -> LLM -> citation validator. The LLM is faked (a spy),
same treatment `test_rerank_pipeline.py` gives `RerankedRetriever`; the
citation validator is real (with a fake `entailment_fn`), so this exercises
the actual budgeting/prompt/validation wiring end to end without ever
touching the network.
"""

from aegis.chunking.tokens import count_tokens
from aegis.config import settings
from aegis.generation.pipeline import GenerationPipeline
from aegis.generation.prompts import get_prompt
from aegis.generation.prompts.v1 import TEMPLATE as V1
from aegis.guardrails.citations import CitationValidator
from aegis.types import Chunk, DocType, Query, Retrieved


def _chunk(id: str, text: str) -> Chunk:
    return Chunk(
        id=id,
        doc_id=id,
        text=text,
        doc_type=DocType.DOCUMENTATION,
        content_hash="hash",
        token_count=count_tokens(text),
        embedding_text=text,
    )


def _retrieved(id: str, text: str, rank: int) -> Retrieved:
    return Retrieved(chunk=_chunk(id, text), score=1.0, rank=rank)


class _FakeLLM:
    def __init__(self, text: str, provider: str = "fake-provider"):
        self.text = text
        self.provider = provider
        self.calls: list[dict] = []

    def generate_with_provider(self, messages, temperature, top_p, max_tokens):
        self.calls.append(
            {"messages": messages, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
        )
        return self.text, self.provider


def _validator(entailment_result: bool = True):
    return CitationValidator(entailment_fn=lambda claim, chunk_text: entailment_result, lexical_overlap_threshold=0.4)


def test_answer_builds_messages_from_v1_prompt_with_budgeted_context():
    retrieved = [_retrieved("doc::1", "FastAPI validates request bodies with Pydantic models.", 1)]
    llm = _FakeLLM("FastAPI validates request bodies with Pydantic models. [1]")
    pipeline = GenerationPipeline(llm=llm, validator=_validator(), prompt=V1, budget_tokens=1000)

    pipeline.answer(Query(id="q1", text="how does FastAPI validate requests?"), retrieved)

    messages = llm.calls[0]["messages"]
    assert messages[0]["content"] == V1.system_prompt
    assert "how does FastAPI validate requests?" in messages[1]["content"]
    assert "[1]" in messages[1]["content"]
    assert "doc::1" in messages[1]["content"]


def test_answer_forwards_sampling_params_to_the_llm():
    retrieved = [_retrieved("doc::1", "some context text here", 1)]
    llm = _FakeLLM("some answer.")
    pipeline = GenerationPipeline(
        llm=llm, validator=_validator(), prompt=V1, budget_tokens=1000, temperature=0.3, top_p=0.9, max_tokens=256
    )

    pipeline.answer(Query(id="q1", text="a question"), retrieved)

    call = llm.calls[0]
    assert call["temperature"] == 0.3
    assert call["top_p"] == 0.9
    assert call["max_tokens"] == 256


def test_answer_reports_the_winning_provider_and_supported_citations():
    text = "FastAPI validates request bodies with Pydantic models."
    retrieved = [_retrieved("doc::1", text, 1)]
    llm = _FakeLLM(f"{text} [1]", provider="groq")
    pipeline = GenerationPipeline(llm=llm, validator=_validator(entailment_result=True), prompt=V1, budget_tokens=1000)

    result = pipeline.answer(Query(id="q1", text="how does FastAPI validate requests?"), retrieved)

    assert result.provider == "groq"
    assert result.validation.valid_citation_rate == 1.0
    assert result.citations == ["doc::1"]


def test_unsupported_citation_is_excluded_from_the_citations_list():
    retrieved = [_retrieved("doc::1", "Dependency injection uses the Depends function.", 1)]
    llm = _FakeLLM("The sky is a distinctive shade of orange on Wednesdays. [1]")
    pipeline = GenerationPipeline(llm=llm, validator=_validator(entailment_result=False), prompt=V1, budget_tokens=1000)

    result = pipeline.answer(Query(id="q1", text="unrelated question"), retrieved)

    assert result.citations == []
    assert result.validation.valid_citation_rate == 0.0


def test_answer_uses_the_configured_default_prompt_version_when_none_is_passed():
    # Checks against whatever `settings.generation.prompt_version` actually
    # is, not a hardcoded version -- that default has already changed once
    # (v1 -> v2, on real measured evidence, design record 09 §5) and
    # shouldn't require touching this test again if it changes again.
    retrieved = [_retrieved("doc::1", "some context", 1)]
    llm = _FakeLLM("an answer.")
    pipeline = GenerationPipeline(llm=llm, validator=_validator(), budget_tokens=1000)

    pipeline.answer(Query(id="q1", text="a question"), retrieved)

    expected = get_prompt(settings.generation.prompt_version)
    # v3+ fills a per-request key into the system prompt (design record 15e), so
    # compare the fixed text before the first placeholder.
    assert llm.calls[0]["messages"][0]["content"].startswith(expected.system_prompt.split("{key}")[0])
