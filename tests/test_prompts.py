"""TDD for the prompt template registry/message-building (plan.md Phase 9,
step 3). Wording itself isn't unit-tested (no other prompt in this project
is -- see `router/classifier.py`'s `SYSTEM_PROMPT`); what's tested is the
mechanical contract every version must honor: two messages, system first,
query and context both actually present in the user turn.
"""

import pytest

from aegis.generation.prompts import get_prompt
from aegis.generation.prompts.v1 import TEMPLATE as V1


def test_get_prompt_returns_the_requested_version():
    assert get_prompt("v1") is V1


def test_get_prompt_raises_on_unknown_version():
    with pytest.raises(ValueError, match="Unknown prompt version"):
        get_prompt("v99")


def test_build_messages_returns_system_then_user():
    messages = V1.build_messages("how do I use dependency injection?", "[1] some context")

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == V1.system_prompt


def test_build_messages_embeds_query_and_context_in_the_user_turn():
    messages = V1.build_messages("what is Depends?", "[1] Depends is a function.")

    user_content = messages[1]["content"]
    assert "what is Depends?" in user_content
    assert "[1] Depends is a function." in user_content
