"""Is every configured OpenAI model still servable? (design record 16e)

OpenAI removed gpt-4.1-nano between Phase 15 and 2026-10-24: calls returned 404
"deprecated" while /v1/models still listed it, so the listing cannot be trusted
and only a real call proves a model works. One tiny call per distinct role model
costs about $0.00001 each, so this runs on every gate run. It turns a silent
production outage (router, rewrite and entailment all failing) into a red build.

    python -m aegis.eval.model_probe
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from aegis.config import settings

_PROMPT = [{"role": "user", "content": 'Reply with JSON only: {"ok": true}'}]
# Large enough for a reasoning model's hidden tokens (gpt-5* returned empty output at 60).
_MAX_TOKENS = 200


def configured_models() -> list[str]:
    o = settings.openai
    roles = [o.generation_model, o.router_model, o.rewrite_model, o.entailment_model, o.judge_model, o.ragas_model]
    return list(dict.fromkeys(roles))


def probe_all(models: list[str], probe: Callable[[str], None]) -> dict[str, str]:
    """Model -> error text for every model whose probe raised. Empty = all healthy."""
    failures: dict[str, str] = {}
    for model in models:
        try:
            probe(model)
        except Exception as exc:  # noqa: BLE001 -- any failure means "not servable"
            cause = exc.__cause__ or exc
            failures[model] = f"{type(cause).__name__}: {str(cause)[:200]}"
    return failures


def _real_probe(model: str) -> None:
    from aegis.generation.llm.chat import chat_completion

    chat_completion(
        "openai", model, _PROMPT, temperature=0.0, max_tokens=_MAX_TOKENS, purpose="availability_probe", json_mode=True
    )


def main() -> int:
    models = configured_models()
    failures = probe_all(models, _real_probe)
    for model in models:
        print(f"{model:16s} {'FAIL  ' + failures[model] if model in failures else 'ok'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
