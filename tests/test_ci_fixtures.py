"""CI fixture integrity (design record 16c). The release asset is the gate's whole
view of the corpus; if it is swapped, truncated or half-downloaded, the gate must
refuse to run rather than report metrics for something else. The fingerprint that
stamps every gate report is derived from the verified file hashes, so these tests
pin that a changed byte changes the fingerprint and a missing/corrupt file raises."""

from __future__ import annotations

import gzip
import json

import pytest

from aegis.eval.ci_fixtures import FixtureError, build_manifest, verify_fixtures, wait_until_green


def _make(tmp_path, snapshot=b"snapshot-bytes", chunks=b'{"id": "a"}\n'):
    (tmp_path / "collection.snapshot.gz").write_bytes(gzip.compress(snapshot))
    (tmp_path / "chunks.jsonl.gz").write_bytes(gzip.compress(chunks))
    manifest = build_manifest(tmp_path, collection="c", n_chunks=1, dim=4, qdrant_image="qdrant/qdrant:v1.18.3")
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    return manifest


def test_verify_returns_the_fingerprint_recorded_in_the_manifest(tmp_path):
    manifest = _make(tmp_path)
    assert verify_fixtures(tmp_path).fingerprint == manifest["fingerprint"]


def test_a_changed_snapshot_is_refused(tmp_path):
    _make(tmp_path)
    (tmp_path / "collection.snapshot.gz").write_bytes(gzip.compress(b"a different snapshot"))
    with pytest.raises(FixtureError, match="collection.snapshot.gz"):
        verify_fixtures(tmp_path)


def test_a_changed_chunk_file_is_refused(tmp_path):
    _make(tmp_path)
    (tmp_path / "chunks.jsonl.gz").write_bytes(gzip.compress(b'{"id": "b"}\n'))
    with pytest.raises(FixtureError, match="chunks.jsonl.gz"):
        verify_fixtures(tmp_path)


def test_a_missing_file_is_refused(tmp_path):
    _make(tmp_path)
    (tmp_path / "chunks.jsonl.gz").unlink()
    with pytest.raises(FixtureError, match="missing"):
        verify_fixtures(tmp_path)


def test_a_missing_manifest_is_refused(tmp_path):
    _make(tmp_path)
    (tmp_path / "MANIFEST.json").unlink()
    with pytest.raises(FixtureError, match="MANIFEST"):
        verify_fixtures(tmp_path)


def test_the_fingerprint_depends_on_both_files(tmp_path):
    for name in "abc":
        (tmp_path / name).mkdir()
    a = _make(tmp_path / "a")["fingerprint"]
    b = _make(tmp_path / "b", snapshot=b"other")["fingerprint"]
    c = _make(tmp_path / "c", chunks=b'{"id": "z"}\n')["fingerprint"]
    assert len({a, b, c}) == 3


# --- waiting for the restored collection (first hosted-runner run, 2026-10-24) ---
# Qdrant reported status=yellow (optimizer still running) right after the snapshot
# upload on a 2-core runner, where it had been green instantly on an M1. Querying
# during optimization searches a different segment layout than the baseline did, so
# the restore must wait for green -- not accept yellow, and not hang forever.

def _infos(*statuses, points=10):
    it = iter(statuses)
    return lambda: {"status": next(it), "points_count": points}


def test_wait_returns_once_the_collection_turns_green():
    sleeps: list[float] = []
    info = wait_until_green(_infos("yellow", "yellow", "green"), expected_points=10, timeout_s=60, sleep=sleeps.append)
    assert info["status"] == "green"
    assert len(sleeps) == 2


def test_wait_does_not_accept_yellow_when_the_clock_runs_out():
    clock = iter(range(0, 1000, 10))
    with pytest.raises(FixtureError, match="yellow"):
        wait_until_green(lambda: {"status": "yellow", "points_count": 10}, expected_points=10, timeout_s=30,
                         sleep=lambda s: None, monotonic=lambda: next(clock))


def test_wait_refuses_a_green_collection_with_the_wrong_point_count():
    with pytest.raises(FixtureError, match="points"):
        wait_until_green(_infos("green", points=9), expected_points=10, timeout_s=60, sleep=lambda s: None)
