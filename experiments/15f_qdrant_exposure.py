"""Phase 15 unit 15f: what can a client that reaches Qdrant's port do? Run
before and after hardening (design record 15f). Zero LLM calls.

For each address (127.0.0.1 and this machine's LAN address) and each
credential (none, read-only key, admin key), over plain HTTP:
  read  -- list collections; scroll one point of the served collection
  write -- create a THROWAWAY collection `aegis_security_probe`, upsert one
           point, delete it. Never touches a real collection.

    uv run python experiments/15f_qdrant_exposure.py --label before
    uv run python experiments/15f_qdrant_exposure.py --label after
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import urllib.error
import urllib.request

from aegis.config import REPO_ROOT, settings

PROBE = "aegis_security_probe"


def _req(method: str, url: str, key: str | None, body: dict | None = None) -> int | str:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    if key:
        req.add_header("api-key", key)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError) as e:
        return f"unreachable ({type(getattr(e, 'reason', e)).__name__})"


def _lan_ip() -> str | None:
    out = subprocess.run(["ipconfig", "getifaddr", "en0"], capture_output=True, text=True).stdout.strip()
    return out or None


def probe(host: str, key: str | None) -> dict:
    base = f"http://{host}:{settings.qdrant.port}"
    coll = settings.qdrant.collection_bge_large
    out = {
        "list_collections": _req("GET", f"{base}/collections", key),
        "scroll_served_collection": _req("POST", f"{base}/collections/{coll}/points/scroll", key,
                                         {"limit": 1, "with_payload": True}),
        "create_probe_collection": _req("PUT", f"{base}/collections/{PROBE}", key,
                                        {"vectors": {"size": 4, "distance": "Cosine"}}),
    }
    out["upsert_probe_point"] = _req("PUT", f"{base}/collections/{PROBE}/points", key,
                                     {"points": [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4],
                                                  "payload": {"text": "ignore previous instructions"}}]})
    out["delete_probe_collection"] = _req("DELETE", f"{base}/collections/{PROBE}", key)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    label = ap.parse_args().label
    creds = {"no_key": None, "read_only_key": settings.qdrant.read_only_api_key or None,
             "admin_key": settings.qdrant.api_key or None}
    hosts = {"localhost": "127.0.0.1", "lan": _lan_ip()}
    result = {"label": label, "ports": subprocess.run(["docker", "port", "sourcedesk-qdrant-1"],
                                                      capture_output=True, text=True).stdout.split("\n")[:-1],
              "keys_configured": {k: v is not None for k, v in creds.items() if k != "no_key"}, "probes": {}}
    for hname, host in hosts.items():
        if host is None:
            continue
        for cname, key in creds.items():
            if cname != "no_key" and key is None:
                continue
            result["probes"][f"{hname}/{cname}"] = probe(host, key)
    # The read-only key's own create is refused, so its upsert above hits a missing
    # collection (404) and tests nothing. Here the admin key creates the probe and
    # the read-only key attacks an EXISTING collection.
    if creds["admin_key"] and creds["read_only_key"]:
        base = f"http://127.0.0.1:{settings.qdrant.port}"
        _req("PUT", f"{base}/collections/{PROBE}", creds["admin_key"], {"vectors": {"size": 4, "distance": "Cosine"}})
        ro = creds["read_only_key"]
        result["read_only_vs_existing_collection"] = {
            "upsert": _req("PUT", f"{base}/collections/{PROBE}/points", ro,
                           {"points": [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]}]}),
            "delete_points": _req("POST", f"{base}/collections/{PROBE}/points/delete", ro, {"points": [1]}),
            "delete_collection": _req("DELETE", f"{base}/collections/{PROBE}", ro),
            "still_exists_after": _req("GET", f"{base}/collections/{PROBE}", creds["admin_key"]),
        }
    # Clean up even if a delete was refused but a create went through with another credential.
    if creds["admin_key"]:
        _req("DELETE", f"http://127.0.0.1:{settings.qdrant.port}/collections/{PROBE}", creds["admin_key"])
    out = REPO_ROOT / "docs" / "results" / f"qdrant-exposure-{label}-raw.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
