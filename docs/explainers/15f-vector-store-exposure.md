# 15f — The Vector Store Is Sensitive Data: Locking Qdrant, and What Vectors Leak

**Phase:** 15 · **Date:** 2026-10-18 · **Files:** `docker-compose.yml`, `.env` (two new keys, values not recorded here), `src/aegis/config.py` (`QdrantSettings.api_key`, `read_only_api_key`), `src/aegis/embedding/vector_store.py` (`get_client(read_only=)`), `src/aegis/factory.py`, `src/aegis/baseline.py`, `tests/test_qdrant_client_keys.py`, `experiments/15f_qdrant_exposure.py`, `experiments/15f_vector_leakage.py`, raw `docs/results/{qdrant-exposure-before,qdrant-exposure-after,vector-leakage}-raw.json`
**Topics from the course:** Security — vector reconstruction, database hacking, knowledge-base leakage (Module 5)

## 1. What I built

Qdrant now listens on `127.0.0.1` only and requires an API key. There are two keys: an **admin** key (create/write/delete) for indexing and experiments, and a **read-only** key for the served API. A keyless client gets 401, the read-only key gets 403 on every write, and the LAN address refuses connections.

I also measured how much a chunk's content leaks from its stored vector alone, with no payload text.

## 2. Why this approach

**Every defense in 15b–15e lives in the API process, and Qdrant was a second door that skipped all of them.** Design record 15's A6 showed reads. The "before" probe here showed the worse half: from this machine's LAN address, with no credentials, a client could **create a collection, write a point, and delete the collection** (all 200s). Anyone on the same network could delete the served index (an outage) or overwrite vectors so chosen chunks rank for chosen queries (retrieval tampering), and nothing in the API would see it. The partition (15b) and access control (15c) mean nothing to someone who talks to port 6333.

**Loopback binding closes the network door; keys close the local one.** Either alone leaves a gap:
- **Binding only:** any process on this machine, or a container on the Docker network, still has admin rights.
- **Keys only:** the port stays on the LAN for brute force, and the client sends keys over plain HTTP. That's acceptable only on loopback, which `get_client` documents.

**The served API gets the read-only key, because it only searches.** If the API process were compromised (a dependency, an RCE), the attacker would hold whatever key it holds. With the read-only key they can read what the API could already read, but can't poison or delete the index. `get_client(read_only=True)` **refuses** to fall back to the admin key when only the admin key is configured. A silent fallback would quietly undo the split.

**One mitigation was already in place, found by reading the code.** `DenseIndex` resolves chunk text from the local corpus by `chunk_id` and drops unknown IDs. So a write through Qdrant can't inject *new text* into a prompt; it can only reorder real chunks or remove them. That limits index poisoning to tampering and denial of service, not injection. I didn't verify this by writing to the real collection, because that would be destructive.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
|---|---|---|
| Strip `text` from the payload (the vectors are what's needed for search) | A Qdrant dump would then leak only vectors plus metadata. §5 shows text is the much bigger leak | This corpus is **public** (FastAPI docs + public GitHub issues), and its canonical copy is `data/processed/chunks.jsonl` anyway. Stripping means mutating 19,236 points in the served collection (and 7 ablation collections), and it breaks `eval/retrieve_candidates.py` (Phase 4 labeling), which reads `payload["text"]`. **Required for any private corpus**, and recorded as such. |
| Keys only, keep `0.0.0.0` | Smallest change | The port stays reachable from the LAN, and keys travel over plain HTTP. |
| One key for everything | Simpler config | The served process would hold write and delete rights it never uses. |
| TLS on Qdrant | Encrypts the key in transit | Not needed while the port is loopback-only. Required the moment Qdrant runs on another host (Phase 18), and `get_client` hard-codes `https=False` with a comment saying so. |
| Qdrant JWT / RBAC per collection | Finer-grained (per-tenant collections) | The partition is a payload filter in one collection (15b), so a per-collection token doesn't map onto it. That's revisit-with-real-tenants territory (15b §3). |

## 4. How it works

**`docker-compose.yml`:**

```yaml
ports:
  - "127.0.0.1:6333:6333"
  - "127.0.0.1:6334:6334"
environment:
  QDRANT__SERVICE__API_KEY: ${QDRANT_API_KEY:?set QDRANT_API_KEY in .env}
  QDRANT__SERVICE__READ_ONLY_API_KEY: ${QDRANT_READ_ONLY_API_KEY:?set QDRANT_READ_ONLY_API_KEY in .env}
```

The `:?` makes `docker compose up` fail if a key is missing, instead of starting an open Qdrant. Both keys were generated with `secrets.token_urlsafe(32)` and appended to `.env`. Their values were never printed or recorded. The container was recreated: data lives in the `./qdrant_storage` bind mount, and all 19,236 points and 8 collections were verified afterwards.

**`get_client(read_only=False)`:** the admin key by default, the read-only key when `read_only=True`, `RuntimeError` if only the admin key is set, and no key when neither is set (local dev without auth). `factory.build_pipelines` passes `get_client(read_only=True)` to the served `DenseIndex`. `baseline.py`'s direct `QdrantClient(...)` now goes through `get_client`.

**The leakage probe:** the attacker has the stored 1,024-d vectors and nothing else, and knows the public model (bge-large).
- Vocabulary: 2,168 corpus words that appear in ≥ 20 chunks and ≤ 20% of chunks, embedded with bge-large.
- For each of 100 seeded chunks, rank the vocabulary by cosine to the stolen vector and take the top 10.
- Precision@10 = the share of those 10 words that actually occur in the hidden chunk.
- Hubness correction (added after the first result, §6): `score(w) = cos(v, w) − mean cos(w, 2,000 other stolen vectors)`.

## 5. What the numbers said

`uv run pytest -q`: **699 passed, 1 xfailed** (+4 `test_qdrant_client_keys.py`). The delta-indexing integration test passes against the locked Qdrant with the admin key. A keyless client gets `401 Unauthorized`.

**Exposure, before → after** (`15f_qdrant_exposure.py`, plain HTTP; write probes use a throwaway `aegis_security_probe` collection, never a real one):

| Caller | List / read served collection | Create / write / delete a collection |
|---|---|---|
| Before: LAN address, no key | **200 / 200** | **200 / 200 / 200** |
| Before: localhost, no key | 200 / 200 | 200 / 200 / 200 |
| After: LAN address, any credential | connection refused | connection refused |
| After: localhost, no key | **401 / 401** | **401 / 401 / 401** |
| After: localhost, read-only key | 200 / 200 | **403** create; against an existing collection: upsert **403**, delete points **403**, delete collection **403**; collection still there |
| After: localhost, admin key | 200 / 200 | 200 / 200 / 200 |

Port bindings went from `0.0.0.0:6333-6334` + `[::]:6333-6334` to `127.0.0.1:6333-6334`.

**Real server on the read-only key:** it starts, warm-up retrieval works (137.2 ms), `/health` returns 200, and there are no auth errors in the log.

**Vector leakage** (`15f_vector_leakage.py`, seed 15, 100 chunks, 33.9 s on CPU):

| Guess | Precision@10 | Chunks with ≥ 5/10 correct |
|---|---|---|
| 10 random vocabulary words | 0.015 | — |
| The 10 most frequent vocabulary words (no vector needed) | 0.235 | — |
| From the vector, raw cosine | **0.188** (worse than the free prior) | 3/100 |
| From the vector, hubness-corrected | **0.286** | **15/100** |

Example, `doc::advanced/security/oauth2-scopes.md::heading_aware::11` (corrected, 7/10 correct): `datetime, encoded_jwt, jwterror, timedelta, access_token_expire_minutes, import_from_string, verify_token, get_password_hash, token_data, timezone`.

## 6. What surprised me / what to watch for

- **The naive inversion lost to doing nothing.** Raw cosine scored 0.188, against 0.235 for guessing the most frequent words. The cause showed in the top-word lists: `apirouter`, `openapi` and `api_router` top almost every vector (hubness). I added the standard correction *after* seeing that, so the 0.286 is one diagnosed fix, not a sweep. Both numbers are reported.
- **This is a lower bound, and a crude one.** It's a bag-of-words guess over a fixed vocabulary, with no trained inverter. It still recovers topic plus specific identifiers (`access_token_expire_minutes`). Published trained inversion attacks reportedly reconstruct much more, but I haven't run one, so there's no number for it here. The defensible conclusion: **the vectors themselves are sensitive data, and stripping payload text would not make a Qdrant dump safe** for a private corpus. Access control on the store is what matters.
- **The read-only probe initially tested nothing.** The read-only key's own `create` was refused, so its `upsert` hit a missing collection (404) and proved nothing about writes. The probe now has the admin key create the collection and the read-only key attack an existing one (403 ×3). It's the same lesson as the 15 starvation probe and the 15c starvation rewrite: a security check that can't fail isn't a check.
- **The setup now has a hard dependency on `.env`.** `docker compose up` refuses to start without both keys, and so would anyone cloning the repo (`.env` isn't committed). Phase 18 needs the keys in its secret store, plus `https=True` if Qdrant isn't on the same host.
- **The client warns on every connection** ("Api key is used with an insecure connection"). That's expected on loopback; it would be a real finding anywhere else.
- **The dashboard on 6333 is now loopback-only and keyed too.** Opening it in a browser needs the key.

## 7. Where this fits and what's next

This closes A6 and the "vector reconstruction" item at the design-record level, with a measured lower bound. The partition claim from 15b now holds for the deployment, not just the API: the only routes to chunk text are the API (partitioned, keyed) and the loopback admin key.

What's left in Phase 15:
- **15g:** the real-stack injection before/after, and v3's citation check. This needs LLM quota and the user's go-ahead on the token estimate.
- **15h:** the final `security-assessment.md` and the finalized record 15.
