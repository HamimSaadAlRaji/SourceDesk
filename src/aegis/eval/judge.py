"""LLM-as-judge for generation helpfulness (plan.md Phase 11, step 2) --
rubric-based, structured output via `instructor`.

Two independent judges, not judge-vs-hand-label: the user declined to
hand-label any answers (2026-09-25, progress.md Decision Log), so this
project measures **inter-judge agreement** (Cohen's kappa, `eval/
agreement.py`) instead of judge-vs-human agreement -- a real, disclosed
limitation, not an equivalent substitute (design record 11b §6). Both
judges score the same (query, answer, context) triple independently.
Neither generated the answer being judged for the common path: Gemini
(`GeminiSettings.model`, the project's fallback *generation* provider, but
not the primary path for most of the N=30 batch) and `openai/gpt-oss-120b`
(`JudgeSettings.groq_model`, architecturally distinct from the primary
generator `qwen/qwen3.8-27b`) -- see `config.py`'s `JudgeSettings` comment
for why gpt-oss-120b's reasoning-token tax, disqualifying for *generation*
latency, is fine for an off-critical-path judge call.

Rubric kept deliberately narrow: one holistic 1-5 helpfulness score, not a
multi-dimensional breakdown. RAGAS's Faithfulness/ResponseRelevancy
(Unit 1) already cover grounding/relevance; the custom citation-validity
metric (Unit 5) covers citation mechanics. What's left for a judge to add
is the thing none of those measure directly -- does this answer actually
resolve the user's real problem, clearly and at the right scope. `reasoning`
is required alongside the score specifically so Unit 4's disagreement cases
are reviewable, not just a bare number difference.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, Field

from aegis.config import settings

JUDGE_PROMPT_TEMPLATE = """You are evaluating how helpful an AI support assistant's answer is to a real user question about FastAPI.

CRITICAL RULE -- apply this before anything else: if the context does not actually contain the information needed to answer the question, and the assistant honestly says so instead of guessing, that is a CORRECT, HELPFUL answer -- an honest refusal is more helpful than a confident guess. Score it 4 or 5. Do NOT score it low with reasoning like "does not provide concrete guidance", "leaves the problem unresolved", or "offers no solution" -- refusing to guess when the context doesn't support an answer is not a failure to help, it IS the correct, helpful behavior. Only score an honest refusal below 4 if the context actually did contain a usable answer that the assistant missed.

Score the answer's HELPFULNESS on a scale of 1 to 5. Most real answers are NOT a 5 -- reserve 5 for an answer you cannot think of any concrete way to improve (including a correct, honest refusal with no missed answer in the context). Before giving a 5, identify what a 4 version of this same answer would be missing; if you can't name anything, it's a 5.
1 = does not address the user's actual question, or is misleading
2 = partially addresses it but leaves the core problem unresolved
3 = addresses the question but is unclear, incomplete, or harder to use than it should be
4 = clearly and correctly addresses the question (including a correct, honest refusal when the context lacks the answer), minor room for improvement
5 = directly, clearly, and completely resolves the user's actual problem, or is the best possible honest response given the context -- nothing you'd change

Judge helpfulness only -- not citation formatting, not whether every sentence has a citation marker (a separate mechanism already checks that).

Context the assistant had access to:
{context_block}

User's question:
{query}

Assistant's answer:
{answer}

First, in your reasoning, state explicitly whether this is a case where the context lacks the answer and the assistant correctly said so. Then score this answer's helpfulness."""


class HelpfulnessJudgment(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str


class _InstructorLLM(Protocol):
    def generate(self, prompt: str, response_model: type[HelpfulnessJudgment]) -> HelpfulnessJudgment: ...


JudgeFn = Callable[[str, str, str], HelpfulnessJudgment]


def _default_judge_fn(llm: _InstructorLLM) -> JudgeFn:
    def judge(query: str, answer: str, context_block: str) -> HelpfulnessJudgment:
        prompt = JUDGE_PROMPT_TEMPLATE.format(context_block=context_block, query=query, answer=answer)
        return llm.generate(prompt, response_model=HelpfulnessJudgment)

    return judge


class Judge:
    """Wraps one judge model. `name` identifies which judge produced a
    given score -- Unit 4 runs two of these (`"groq"`/`"gemini"`) over the
    same N=30 answers and feeds both label sequences to
    `eval/agreement.cohen_kappa`."""

    def __init__(self, name: str, judge_fn: JudgeFn | None = None, llm: _InstructorLLM | None = None) -> None:
        if judge_fn is None and llm is None:
            raise ValueError("Judge needs either judge_fn or llm")
        self.name = name
        self._judge_fn = judge_fn or _default_judge_fn(llm)

    def judge(self, query: str, answer: str, context_block: str) -> HelpfulnessJudgment:
        return self._judge_fn(query, answer, context_block)


def build_groq_judge(model: str = settings.judge.groq_model) -> Judge:
    from aegis.eval.ragas_adapters import build_groq_judge_llm

    return Judge(name="groq", llm=build_groq_judge_llm(model=model))


def build_gemini_judge(model: str = settings.gemini.model) -> Judge:
    from aegis.eval.ragas_adapters import build_gemini_judge_llm

    return Judge(name="gemini", llm=build_gemini_judge_llm(model=model))
