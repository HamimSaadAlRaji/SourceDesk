from __future__ import annotations

from aegis.generation.prompts.base import PromptTemplate
from aegis.generation.prompts.v1 import TEMPLATE as V1
from aegis.generation.prompts.v2 import TEMPLATE as V2

_VERSIONS: dict[str, PromptTemplate] = {V1.version: V1, V2.version: V2}


def get_prompt(version: str) -> PromptTemplate:
    try:
        return _VERSIONS[version]
    except KeyError:
        raise ValueError(f"Unknown prompt version: {version!r}. Known: {sorted(_VERSIONS)}") from None


__all__ = ["PromptTemplate", "get_prompt"]
