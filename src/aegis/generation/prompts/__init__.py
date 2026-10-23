from __future__ import annotations

from aegis.generation.prompts.base import PromptTemplate
from aegis.generation.prompts.v1 import TEMPLATE as V1
from aegis.generation.prompts.v2 import TEMPLATE as V2
from aegis.generation.prompts.v2c import TEMPLATE as V2C
from aegis.generation.prompts.v3 import TEMPLATE as V3

_VERSIONS: dict[str, PromptTemplate] = {V1.version: V1, V2.version: V2, V2C.version: V2C, V3.version: V3}


def get_prompt(version: str) -> PromptTemplate:
    try:
        return _VERSIONS[version]
    except KeyError:
        raise ValueError(f"Unknown prompt version: {version!r}. Known: {sorted(_VERSIONS)}") from None


__all__ = ["PromptTemplate", "get_prompt"]
