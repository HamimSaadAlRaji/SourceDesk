"""`uv run python -m aegis.cost` -- the paid-API spend so far against the
project's hard budget (design record 15g)."""

import json

from aegis.generation.llm.openai_provider import default_ledger

print(json.dumps(default_ledger().summary(), indent=2))
