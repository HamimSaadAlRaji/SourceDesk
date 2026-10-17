"""ASGI entry point: `uv run uvicorn aegis.api.main:app`. Builds the real
pipeline at startup (lifespan), so importing this module is cheap."""

from aegis.api.app import create_app
from aegis.observability.logging import configure_logging

configure_logging()
app = create_app()
