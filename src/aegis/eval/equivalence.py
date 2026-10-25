"""Comparing two orchestrations on real LLM output (Phase 17, design record 17d).

The question is whether the graph chooses the same *path* as the direct pipeline. A paid,
temperature-0 LLM is not bit-reproducible (Open Question #4), so running both paths live
would put model noise into a comparison that is meant to isolate orchestration. Instead the
first path runs live and records every LLM call on a `Tape`; the second replays it. A
replayed run can only differ where the graph *asked a different question* (a replay miss,
reported loudly) or *used the answer differently*, which is the thing being measured.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aegis.pipeline import PipelineResult

# Per-run by construction: a fresh trace id, wall-clock timings.
_PER_RUN = {"trace_id", "stage_latencies_ms"}


class ReplayMissError(RuntimeError):
    """The replaying path made an LLM call the recording path never made."""


class Tape:
    def __init__(self, entries: dict[str, Any] | None = None, replay_only: bool = False) -> None:
        self.entries: dict[str, Any] = entries if entries is not None else {}
        self.replay_only = replay_only
        self._used: set[str] = set()
        self._lock = threading.Lock()

    @staticmethod
    def key(kind: str, args: Any) -> str:
        return hashlib.sha256(json.dumps([kind, args], sort_keys=True, default=str).encode()).hexdigest()

    def call(self, kind: str, args: Any, fn: Callable[[], Any]) -> Any:
        """`kind` namespaces the key so the same text sent to two different stages can never share an entry."""
        key = self.key(kind, args)
        with self._lock:
            if key in self.entries:
                self._used.add(key)
                return self.entries[key]
        if self.replay_only:
            raise ReplayMissError(f"{kind}: no recorded call for {json.dumps(args, default=str)[:200]}")
        value = fn()  # outside the lock: a slow paid call must not serialize the other threads
        with self._lock:
            self.entries[key] = value
            self._used.add(key)
        return value

    @property
    def unused(self) -> int:
        """Recorded calls the replaying path never made: the mirror image of a replay miss."""
        return len(set(self.entries) - self._used)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.entries))

    @classmethod
    def load(cls, path: Path, replay_only: bool = False) -> Tape:
        return cls(json.loads(path.read_text()), replay_only)


def taped_fn(tape: Tape, kind: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a stage's LLM function (router classify, citation entailment) so calls go through the tape."""
    return lambda *args: tape.call(kind, list(args), lambda: fn(*args))


_V3_KEY = re.compile(r"key: ([0-9a-f]{16}) ===")


def _without_boundary_key(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Prompt v3 draws a fresh random boundary key per request (Phase 15's injection defence), so
    two requests with identical content never match byte for byte. The key carries no information
    about the request, so it is replaced by a fixed token before keying. Only the key found in the
    `key: ... ===` markers is replaced; every other character still has to match."""
    found = {m.group(1) for msg in messages for m in _V3_KEY.finditer(msg["content"])}
    out = []
    for msg in messages:
        content = msg["content"]
        for key in found:
            content = content.replace(key, "<KEY>")
        out.append({**msg, "content": content})
    return out


class TapedLLM:
    """The generator's LLM client, taped. Keyed on the full request, so a changed prompt or
    sampling parameter is a different call, not a stale hit."""

    def __init__(self, inner: Any, tape: Tape) -> None:
        self.inner, self.tape = inner, tape

    def generate_with_provider(
        self, messages: list[dict[str, str]], temperature: float, top_p: float, max_tokens: int
    ) -> tuple[str, str]:
        args = [_without_boundary_key(messages), temperature, top_p, max_tokens]
        text, provider = self.tape.call(
            "generation", args, lambda: self.inner.generate_with_provider(messages, temperature, top_p, max_tokens)
        )
        return text, provider


def compare_results(a: PipelineResult, b: PipelineResult) -> list[str]:
    """Names of the fields on which two results disagree; empty means identical. `stages` is the
    set of stages that ran, compared separately because their durations are per-run."""
    da, db = a.model_dump(exclude=_PER_RUN), b.model_dump(exclude=_PER_RUN)
    diffs = [name for name in da if da[name] != db[name]]
    if set(a.stage_latencies_ms) != set(b.stage_latencies_ms):
        diffs.append("stages")
    return diffs
