"""The interface every provider client implements (plan.md Phase 9, step 1:
"one streaming interface"). Deliberately minimal -- `stream` is the only
required method, so `.generate()` (non-streaming) can be built once, on top
of it, in `fallback.py`, rather than duplicated per provider.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol


class ProviderClient(Protocol):
    name: str

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Iterator[str]: ...
