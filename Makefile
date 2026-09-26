.PHONY: ingest index eval serve test

ingest:
	uv run python -m aegis.ingest.run

index:
	uv run python -m aegis.embedding.run

eval:
	uv run python -m aegis.eval.scorecard

serve:
	uv run uvicorn aegis.api.app:app --reload

test:
	uv run pytest -q
