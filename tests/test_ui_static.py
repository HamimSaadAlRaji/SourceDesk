"""The UI is static files served by the API. The SSE parser is real JS logic
(frames split across network reads), so it is exercised under node when
available; the rest of the UI is verified in a real browser (design record 14e)."""

import json
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from aegis.api.app import create_app
from test_api import _FakePipeline


def test_index_and_assets_are_served():
    with TestClient(create_app(_FakePipeline())) as c:
        html = c.get("/")
        assert html.status_code == 200 and "text/html" in html.headers["content-type"]
        for path in ("/static/app.js", "/static/sse.js", "/static/style.css"):
            assert c.get(path).status_code == 200, path


def test_ui_never_injects_model_text_as_html():
    from aegis.api.app import FsPath

    js = (FsPath(__file__).parent.parent / "src/aegis/api/static/app.js").read_text()
    assert "innerHTML" not in js and "insertAdjacentHTML" not in js


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_sse_parser_handles_frames_split_across_chunks():
    script = """
    const { createSSEParser } = require('./src/aegis/api/static/sse.js');
    const p = createSSEParser();
    const out = [];
    const wire = 'event: token\\ndata: {"text":"a\\\\nb"}\\n\\nevent: done\\ndata: {"x":1}\\n\\n';
    for (const piece of [wire.slice(0, 9), wire.slice(9, 40), wire.slice(40)]) out.push(...p.feed(piece));
    console.log(JSON.stringify(out));
    """
    from aegis.api.app import FsPath

    root = FsPath(__file__).parent.parent
    r = subprocess.run(["node", "-e", script], cwd=root, capture_output=True, text=True, check=True)
    assert json.loads(r.stdout) == [
        {"type": "token", "data": {"text": "a\nb"}},
        {"type": "done", "data": {"x": 1}},
    ]


def test_ui_explains_every_escalation_cause():
    # The cause set is closed (aegis.events.EscalationCause); a new cause
    # without UI text would fall through to a generic sentence.
    from typing import get_args

    from aegis.events import EscalationCause

    from pathlib import Path

    js = (Path(__file__).parent.parent / "src/aegis/api/static/app.js").read_text()
    for cause in get_args(EscalationCause):
        assert f"{cause}:" in js, cause
