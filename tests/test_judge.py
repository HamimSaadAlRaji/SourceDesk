"""TDD for the LLM-as-judge wiring (plan.md Phase 11, step 2). The LLM is
faked (a spy), same treatment `test_generation_pipeline.py` gives
`GenerationPipeline`'s LLM -- this exercises the real prompt-formatting and
score-validation logic without ever touching the network. The two real
judge models (Gemini, `openai/gpt-oss-120b`) are exercised by a real
integration script instead (Unit 4's batch run), same "I/O-bound components
get integration tests" split CLAUDE.md's testing section calls for.
"""

import pytest
from pydantic import ValidationError

from aegis.eval.judge import HelpfulnessJudgment, Judge, JUDGE_PROMPT_TEMPLATE


class _FakeInstructorLLM:
    def __init__(self, judgment: HelpfulnessJudgment):
        self.judgment = judgment
        self.calls: list[dict] = []

    def generate(self, prompt: str, response_model):
        self.calls.append({"prompt": prompt, "response_model": response_model})
        return self.judgment


def test_judge_returns_the_llms_judgment():
    fake = _FakeInstructorLLM(HelpfulnessJudgment(score=4, reasoning="clear and correct"))
    judge = Judge(name="fake", llm=fake)

    result = judge.judge(query="how do I validate a body?", answer="use a Pydantic model [1]", context_block="[1] ...")

    assert result.score == 4
    assert result.reasoning == "clear and correct"


def test_judge_passes_the_response_model_to_the_llm():
    fake = _FakeInstructorLLM(HelpfulnessJudgment(score=3, reasoning="ok"))
    judge = Judge(name="fake", llm=fake)

    judge.judge(query="q", answer="a", context_block="c")

    assert fake.calls[0]["response_model"] is HelpfulnessJudgment


def test_judge_prompt_includes_the_real_query_answer_and_context():
    fake = _FakeInstructorLLM(HelpfulnessJudgment(score=5, reasoning="great"))
    judge = Judge(name="fake", llm=fake)

    judge.judge(
        query="how do I use Depends?",
        answer="Use the Depends function in your path operation [1].",
        context_block="[1] Depends injects a dependency.",
    )

    prompt = fake.calls[0]["prompt"]
    assert "how do I use Depends?" in prompt
    assert "Use the Depends function in your path operation [1]." in prompt
    assert "[1] Depends injects a dependency." in prompt


def test_judge_name_identifies_which_judge_produced_a_score():
    fake = _FakeInstructorLLM(HelpfulnessJudgment(score=4, reasoning="ok"))

    assert Judge(name="groq", llm=fake).name == "groq"
    assert Judge(name="gemini", llm=fake).name == "gemini"


def test_judge_requires_either_judge_fn_or_llm():
    with pytest.raises(ValueError, match="judge_fn or llm"):
        Judge(name="fake")


def test_custom_judge_fn_bypasses_the_default_instructor_wiring():
    calls = []

    def fake_fn(query, answer, context_block):
        calls.append((query, answer, context_block))
        return HelpfulnessJudgment(score=2, reasoning="incomplete")

    judge = Judge(name="fake", judge_fn=fake_fn)

    result = judge.judge(query="q", answer="a", context_block="c")

    assert result.score == 2
    assert calls == [("q", "a", "c")]


def test_helpfulness_judgment_rejects_score_below_one():
    with pytest.raises(ValidationError):
        HelpfulnessJudgment(score=0, reasoning="bad")


def test_helpfulness_judgment_rejects_score_above_five():
    with pytest.raises(ValidationError):
        HelpfulnessJudgment(score=6, reasoning="bad")


def test_judge_prompt_template_does_not_penalize_honest_refusal():
    # Regression guard for a real design intent: an honest "I don't know"
    # when the context lacks the answer should score *high*, not low. If
    # this instruction is ever accidentally deleted from the template, a
    # judge could start penalizing exactly the behavior Phase 10's
    # abstention gate is built to produce -- the North Star failure mode
    # this project exists to prevent, scored backwards.
    assert "honest refusal" in JUDGE_PROMPT_TEMPLATE.lower() or "don't have enough information" in JUDGE_PROMPT_TEMPLATE.lower()


def test_judge_prompt_template_states_the_refusal_rule_before_the_numeric_scale():
    # Real diagnosed failure mode (docs/results/judge-agreement.md,
    # `openai/gpt-oss-120b`): the honest-refusal instruction existed in the
    # old template but sat *after* the 1-5 scale definition, as a side
    # note -- and the judge's own real reasoning text overrode it on pure
    # task-completion grounds ("does not help the user solve their
    # problem"). Moving the rule ahead of the scale is the actual fix being
    # tested here, not just the rule's presence (already guarded above).
    lower = JUDGE_PROMPT_TEMPLATE.lower()
    refusal_pos = lower.find("honest refusal") if "honest refusal" in lower else lower.find("don't have enough information")
    scale_pos = lower.find("1 = does not address")
    assert refusal_pos != -1 and scale_pos != -1
    assert refusal_pos < scale_pos


def test_judge_prompt_template_does_not_penalize_refusal_for_lacking_guidance():
    # Mirrors the real observed override language from gpt-oss-120b's own
    # reasoning text ("does not provide any concrete guidance", "offers no
    # solution, explanation, or guidance") -- the fix explicitly names and
    # preempts exactly this framing, not just the general refusal case.
    lower = JUDGE_PROMPT_TEMPLATE.lower()
    assert "guidance" in lower or "solve" in lower


def test_judge_prompt_template_anchors_against_the_ceiling_effect():
    # Real diagnosed failure mode: Gemini scored 10/11 real answers a
    # perfect 5/5 (docs/results/judge-agreement.md), carrying almost no
    # discriminative information. The fix adds an explicit anchoring
    # instruction reserving 5 for answers with no plausible improvement,
    # without going multi-dimensional (design record 11c already rejected
    # that -- this keeps one holistic score).
    lower = JUDGE_PROMPT_TEMPLATE.lower()
    assert "reserve 5" in lower or "reserve a 5" in lower


def test_judge_prompt_template_asks_for_an_explicit_refusal_classification_first():
    # The reasoning field is required specifically so disagreement cases
    # are reviewable (module docstring) -- forcing the judge to name
    # up-front whether this *is* a refusal case makes that review (and any
    # future re-diagnosis) cheap: grep the reasoning text instead of
    # re-reading every judgment by hand.
    assert "first" in JUDGE_PROMPT_TEMPLATE.lower()
