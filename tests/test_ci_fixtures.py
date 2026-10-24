"""CI fixture integrity (design record 16c). The release asset is the gate's whole
view of the corpus; if it is swapped, truncated or half-downloaded, the gate must
refuse to run rather than report metrics for something else. The fingerprint that
stamps every gate report is derived from the verified file hashes, so these tests
pin that a changed byte changes the fingerprint and a missing/corrupt file raises."""

from __future__ import annotations

import gzip
import json

import pytest

from aegis.eval.ci_fixtures import FixtureError, build_manifest, verify_fixtures, wait_until_restored


def _make(tmp_path, snapshot=b"snapshot-bytes", chunks=b'{"id": "a"}\n'):
    (tmp_path / "collection.snapshot.gz").write_bytes(gzip.compress(snapshot))
    (tmp_path / "chunks.jsonl.gz").write_bytes(gzip.compress(chunks))
    manifest = build_manifest(tmp_path, collection="c", n_chunks=1, dim=4, qdrant_image="qdrant/qdrant:v1.18.3", segments_count=4)
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


# --- readiness of the restored collection (first hosted-runner run, 2026-10-24) ---
# On a 2-core runner Qdrant merged the snapshot's segments after the restore (status
# yellow for ~20 s, segments 4 -> 5 -> 2) and rebuilt HNSW: same snapshot, same
# queries, recall@10 0.2687 instead of 0.4317. The gate is only meaningful on the
# snapshot's own graph, so the restore requires the optimizer to be disabled
# (max_optimization_threads == 0) and the segment layout to equal the exported one.
# "yellow" is then the normal, permanent state (pending merges that can never run),
# so readiness cannot be "green".


def _info(status="yellow", points=10, indexed=10, segments=4, optimizer="ok", threads=0):
    return {
        "status": status, "points_count": points, "indexed_vectors_count": indexed,
        "segments_count": segments, "optimizer_status": optimizer,
        "config": {"optimizer_config": {"max_optimization_threads": threads}},
    }


def _wait(infos, **kw):
    it = iter(infos)
    return wait_until_restored(
        lambda: next(it), expected_points=10, expected_segments=4, timeout_s=kw.pop("timeout_s", 60),
        sleep=lambda s: None, stable_polls=kw.pop("stable_polls", 3), **kw,
    )


def test_ready_when_layout_matches_and_holds_steady_even_while_yellow():
    assert _wait([_info(), _info(), _info()])["segments_count"] == 4


def test_a_still_filling_collection_is_waited_out():
    info = _wait([_info(indexed=3), _info(indexed=7), _info(), _info(), _info()])
    assert info["indexed_vectors_count"] == 10


def test_refuses_immediately_when_the_optimizer_is_enabled():
    # Without this check the first poll can see the original segments just before a merge starts.
    for threads in (None, 2):
        with pytest.raises(FixtureError, match="optimizer"):
            _wait([_info(threads=threads)])


def test_a_changed_segment_layout_never_counts_as_ready():
    clock = iter(range(0, 1000, 10))
    with pytest.raises(FixtureError, match="segments"):
        _wait([_info(segments=2)] * 50, timeout_s=30, monotonic=lambda: next(clock))


def test_a_wrong_point_count_never_counts_as_ready():
    clock = iter(range(0, 1000, 10))
    with pytest.raises(FixtureError, match="points"):
        _wait([_info(points=9)] * 50, timeout_s=30, monotonic=lambda: next(clock))


def test_a_red_collection_fails_at_once():
    with pytest.raises(FixtureError, match="red"):
        _wait([_info(status="red")])


def test_stability_must_be_consecutive():
    # A layout that flaps (4, 5, 4, ...) is a merge in progress, not a restored index.
    clock = iter(range(0, 1000, 10))
    flapping = [_info(), _info(), _info(segments=5)] * 20
    with pytest.raises(FixtureError, match="segments"):
        _wait(flapping, timeout_s=60, monotonic=lambda: next(clock))
