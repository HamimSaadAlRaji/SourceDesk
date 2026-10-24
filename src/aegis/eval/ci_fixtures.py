"""Export and restore the CI evaluation fixtures (design record 16c).

The gate runs against a Qdrant *snapshot* of the served collection rather than an
index rebuilt from vectors, because a rebuilt HNSW graph is neither reproducible
nor comparable to any published number (design record 16b). Two files travel
together as a GitHub release asset: the gzipped snapshot and the gzipped
`chunks.jsonl` that the snapshot's payload ids refer to. `MANIFEST.json` records
their SHA-256 hashes; `verify_fixtures` refuses anything that doesn't match, and
the fingerprint stamped on every gate report is derived from those hashes so a
metric can never be attributed to the wrong corpus.

    python -m aegis.eval.ci_fixtures export  --out ci-fixtures
    python -m aegis.eval.ci_fixtures restore --src ci-fixtures --qdrant-url http://localhost:6333
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from aegis.config import settings

SNAPSHOT_FILE = "collection.snapshot.gz"
CHUNKS_FILE = "chunks.jsonl.gz"
MANIFEST_FILE = "MANIFEST.json"
_HASHED_FILES = (SNAPSHOT_FILE, CHUNKS_FILE)
_TIMEOUT_S = 600.0


class FixtureError(RuntimeError):
    """The fixtures are missing, corrupt, or not what the manifest says."""


@dataclass(frozen=True)
class VerifiedFixtures:
    fingerprint: str
    manifest: dict


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fingerprint(file_hashes: dict[str, str]) -> str:
    joined = "|".join(f"{name}={file_hashes[name]}" for name in _HASHED_FILES)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def build_manifest(directory: Path, *, collection: str, n_chunks: int, dim: int, qdrant_image: str) -> dict:
    hashes = {name: _sha256(directory / name) for name in _HASHED_FILES}
    return {
        "collection": collection, "n_chunks": n_chunks, "dim": dim, "qdrant_image": qdrant_image,
        "files": hashes, "fingerprint": _fingerprint(hashes),
    }


def verify_fixtures(directory: Path) -> VerifiedFixtures:
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.exists():
        raise FixtureError(f"{MANIFEST_FILE} missing from {directory}")
    manifest = json.loads(manifest_path.read_text())
    actual: dict[str, str] = {}
    for name in _HASHED_FILES:
        path = directory / name
        if not path.exists():
            raise FixtureError(f"{name} missing from {directory}")
        actual[name] = _sha256(path)
        if actual[name] != manifest["files"][name]:
            raise FixtureError(f"{name} does not match the manifest (sha256 {actual[name][:12]} != {manifest['files'][name][:12]})")
    fingerprint = _fingerprint(actual)
    if fingerprint != manifest["fingerprint"]:
        raise FixtureError("manifest fingerprint is inconsistent with its own file hashes")
    return VerifiedFixtures(fingerprint=fingerprint, manifest=manifest)


def _gzip_file(src: Path, dst: Path) -> None:
    # mtime=0 so identical content gzips to identical bytes (a stable fingerprint).
    with open(src, "rb") as fin, open(dst, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fout:
        shutil.copyfileobj(fin, fout)


def _gunzip_file(src: Path, dst: Path) -> None:
    with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)


def _qdrant_base() -> str:
    return f"http://{settings.qdrant.host}:{settings.qdrant.port}"


def export_fixtures(out_dir: Path, qdrant_image: str = "qdrant/qdrant:v1.18.3") -> dict:
    """Snapshot the served collection and bundle it with the chunk file. Run once,
    by hand, whenever the corpus or index changes; the result is uploaded as a
    release asset. The server-side snapshot copy is deleted afterwards."""
    collection = settings.qdrant.collection_bge_large
    headers = {"api-key": settings.qdrant.api_key} if settings.qdrant.api_key else {}
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _qdrant_base()
    with httpx.Client(timeout=_TIMEOUT_S, headers=headers) as http, tempfile.TemporaryDirectory() as tmp:
        created = http.post(f"{base}/collections/{collection}/snapshots", params={"wait": "true"})
        created.raise_for_status()
        name = created.json()["result"]["name"]
        raw_snapshot = Path(tmp) / "collection.snapshot"
        try:
            with http.stream("GET", f"{base}/collections/{collection}/snapshots/{name}") as resp, open(raw_snapshot, "wb") as f:
                resp.raise_for_status()
                for block in resp.iter_bytes(1 << 20):
                    f.write(block)
        finally:
            http.delete(f"{base}/collections/{collection}/snapshots/{name}", params={"wait": "true"})
        info = http.get(f"{base}/collections/{collection}").json()["result"]
        _gzip_file(raw_snapshot, out_dir / SNAPSHOT_FILE)
    chunks_src = settings.paths.data_processed / "chunks.jsonl"
    _gzip_file(chunks_src, out_dir / CHUNKS_FILE)
    dim = info["config"]["params"]["vectors"]["dense"]["size"]
    manifest = build_manifest(out_dir, collection=collection, n_chunks=info["points_count"], dim=dim, qdrant_image=qdrant_image)
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return manifest


def restore_fixtures(src_dir: Path, qdrant_url: str | None = None) -> VerifiedFixtures:
    """Verify, then load the snapshot into the (empty, key-less) CI Qdrant and put
    chunks.jsonl where the pipeline expects it. Fails if the restored collection
    isn't green with the manifest's point count."""
    verified = verify_fixtures(src_dir)
    manifest = verified.manifest
    base = qdrant_url or _qdrant_base()
    collection = manifest["collection"]
    with tempfile.TemporaryDirectory() as tmp:
        raw_snapshot = Path(tmp) / "collection.snapshot"
        _gunzip_file(src_dir / SNAPSHOT_FILE, raw_snapshot)
        with httpx.Client(timeout=_TIMEOUT_S) as http, open(raw_snapshot, "rb") as f:
            resp = http.post(
                f"{base}/collections/{collection}/snapshots/upload",
                params={"priority": "snapshot", "wait": "true"},
                files={"snapshot": (raw_snapshot.name, f)},
            )
            resp.raise_for_status()
            info = http.get(f"{base}/collections/{collection}").json()["result"]
    if info["status"] != "green" or info["points_count"] != manifest["n_chunks"]:
        raise FixtureError(f"restored {collection}: status={info['status']} points={info['points_count']}, "
                           f"expected green/{manifest['n_chunks']}")
    dest = settings.paths.data_processed / "chunks.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    _gunzip_file(src_dir / CHUNKS_FILE, dest)
    return verified


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_export = sub.add_parser("export")
    p_export.add_argument("--out", type=Path, required=True)
    p_restore = sub.add_parser("restore")
    p_restore.add_argument("--src", type=Path, required=True)
    p_restore.add_argument("--qdrant-url", default=None)
    args = parser.parse_args(argv)
    if args.cmd == "export":
        manifest = export_fixtures(args.out)
        print(json.dumps(manifest, indent=2))
    else:
        verified = restore_fixtures(args.src, args.qdrant_url)
        print(f"restored {verified.manifest['collection']} ({verified.manifest['n_chunks']} points), fingerprint {verified.fingerprint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
