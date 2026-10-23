# Security Assessment — Aegis (Phase 15)

**Status:** ✅ complete (2026-10-22, unit 15i). All four plan.md exit criteria met (see "Exit criteria" below). Served configuration at sign-off: prompt v3, generator gpt-5-mini (fallback qwen → gemini), router gpt-4.1-nano, commitment + risky-advice output checks, partitioned retrieval, keyed loopback Qdrant. Mechanism and reasoning: `docs/explainers/15-rag-security.md`; per-unit records 15b–15l.

**How to reproduce:**
- `uv run pytest tests/security -q -rxX`: 31 tests, all pass (A3's xfail flipped when v3 was served in 15h; A9 added in 15i).
- `uv run python experiments/15l_risky_advice_fp.py`: risky-advice check on frozen probes, 15h's real obeyed answers, every stored answer, and the documentation (no LLM calls).
- `uv run python experiments/15h_realstack_security.py` (paid, ~$0.19, Langfuse forced off) and `15h_claim_split_sensitivity.py`: H1/H2/H3, raw in `security-realstack-raw.json`; poison set `data/security/poison_chunks_v1.jsonl`.
- `uv run python experiments/15e_commitment_fp.py`: commitment-check false positives (real answers, corpus, probes).
- `uv run python experiments/15f_qdrant_exposure.py --label after`, `15f_vector_leakage.py`: Qdrant exposure matrix; vector-only content recovery.
- 15d comparisons: `experiments/15d_input_boundary_comparison.py` (DeBERTa/Presidio via a `uv run --with` overlay, see its docstring), `15d_deberta_padding_evasion.py`, `15d_redaction_retrieval_impact.py`. Attack and probe sets: `data/security/attacks_v1.jsonl`, `pii_probes_v1.jsonl` (sha256 in each raw result).
- `uv run python experiments/15a_security_baseline.py`: raw output in `security-baseline-raw.json`. No LLM calls, ~56 s.
- `uv run python experiments/15b_partition_bypass.py`: raw output in `partition-bypass-raw.json`. No LLM calls, ~11–27 s. **Re-run 2026-10-22 (15i) after the 15j–15l changes: unchanged, A1 0/1,800, 0/760 out-of-partition.**

**Caveat on tenancy:** the partition holds at the API and retrieval layers. Since 15f, Qdrant is loopback-only and keyed, and the served API holds only the read-only key, so the remaining route around the partition is the local admin key.

## Scorecard

| # | Attack | Class | Before (2026-10-18) | After |
|---|---|---|---|---|
| A1 | `doc_type` filter bypassed by the BM25 leg of hybrid search | Tenant separation | **Succeeds.** 609/1,800 top-10 slots (33.8%) are issue chunks under `doc_type=documentation`; 161/180 dev queries leak. Dense-only: 0. Latent: the served path doesn't filter. | **Defended (15b).** 0/1,800. Filters reach every leg (bm25s weight mask + post-filter) |
| A2 | Cached answer replayed across callers/partitions | Tenant separation | **Succeeds.** Cache key has no partition or principal | **Defended (15b) across partitions.** Partition is in both cache scopes. Same-partition replay across callers is by design |
| A3 | Issue text forges a context block with a `doc::` source | Indirect injection | **Succeeds** | **Defended and served (15h, 2026-10-22).** v3 served on a pre-registered rule: gpt-5-mini obeyed a rank-1 poisoned chunk 5/50 under v2, 0/50 under v3; citation rate not worse (v3 − v2 +0.068, CI [−0.039, +0.175], line-aware split). 0 forged blocks across 3 styles (15e). Costs +624 input tokens per 10-block request |
| A4 | Injected refund commitment passes citation validation | Indirect injection | **Succeeds.** `SUPPORTED_LEXICAL` with 0 LLM calls | **Defended (15e).** `UNAUTHORIZED_COMMITMENT` before citation checks: no validated citation, not cached, UI alert. FP: corpus 0/40,551 sentences; technical-payment probe 3/8 (safe direction) |
| A5a | Anonymous access to `/chat`, `/feedback`, `/trace` | KB leakage | **Succeeds** | **Policy (15c, user decision):** `/chat` public by design; `/trace` 401 anonymous / 403 non-trace key; invalid key → 401 everywhere |
| A5b | Reading another caller's trace (query + chunk text) | KB leakage | **Succeeds** | **Defended (15c).** Operator-only; real server 401 |
| A5c | Overwriting another caller's vote/comment | Integrity | **Succeeds** | **Defended (15c).** Owner-scoped; real server: no cookie 404, other cookie 404, owner 200 |
| A5d | 40 slow `/trace` calls starve every sync endpoint | Availability | **Succeeds.** `/health` no answer in 2 s | **Defended (15c).** Real server, valid operator key: 2 ran, 38 × 503; `/health` 3.6 ms during burst |
| A5e | `/feedback` flood | Availability / integrity | **Succeeds.** 200/200 accepted | **Defended (15c).** Real: 9 × 200 then 21 × 429 |
| A5f | `X-Forwarded-For` spoof resets `/chat` rate limit | Input validation | Defended | Still defended |
| A5g | Terminal escape sequences via feedback comment into the reviewer's terminal | Input validation | **Succeeds** | **Defended (15c).** ANSI/BEL/bidi/newline shown escaped |
| A6 | Qdrant readable (and, 15f found, **writable/deletable**) without credentials, from the LAN | KB leakage / vector DB | **Succeeds.** No API key; bound `0.0.0.0`/`[::]`; payload holds full chunk text; scroll dumps the corpus; LAN no-key create/write/delete all 200 | **Defended (15f).** Loopback-only; no key → 401; read-only key (served API) → 403 on create/upsert/delete; LAN → connection refused. Payload text kept (public corpus; must strip for private) |
| V1 | Content recovered from stored vectors alone (no payload) | Vector reconstruction | Not measured before | **Measured (15f), lower bound:** crude vocab inversion precision@10 0.286 (hubness-corrected) vs 0.235 frequency prior vs 0.015 random; 15/100 chunks ≥ 5/10. Vectors are sensitive data; access control is the defense |
| A7 | Input-side injection detection | Input validation | None exists. Naive regex probe: 7/7 dev adversarial, 4/150 (2.7%) dev positive false positives, all on "override". Evasion not yet measured. | **Flag only (15d), by measurement.** Regex: 0/150 FP, 6/7 golden, **0/14 paraphrases**. DeBERTa (rejected): 14/14 paraphrases but **0/7 when padded with a real issue**, 1.2–4.7% FP, 200–556 ms. Neither is a defense |
| P1 | Secrets/PII in user text reach store, caches, Langfuse, Groq/Gemini | LLM data leakage | **Succeeds.** Raw text to all 7 destinations | **Defended (15d).** Redacted at the boundary: 8/8 secret probes, 0/6 code negatives, 3/180 dev touched (all real), retrieval unchanged (10/10 overlap). Presidio (rejected): 135/180 dev flagged, 4/8 secrets missed |
| A8 | Direct injection wrapped in a real FastAPI question gets past the confidence gate | Direct injection | **Succeeds.** 14/14 pass τ = 0.5 (0.6893–0.8007); unwrapped 0/7 (reproduces Phase 10) | **Gate still passes 14/14 end to end (15h); the model is the defense.** gpt-5-mini: v2 complied 1/14 (full system prompt leaked, translated to French), v3 0/14 (1 refused without answering). Paraphrases: 13/14 escalated by router/gate |
| A9 | Validator certifies poisoned advice relayed from an issue comment (`pip install <typosquat>` cited to the poisoned chunk) | Indirect injection | **Succeeds (found in 15h).** 4/4 obeyed install claims scored `SUPPORTED_LEXICAL` | **Defended for install / link / secret-request (15l).** `UNVOUCHED_RISKY_ADVICE` unless the documentation corpus vouches; not cached; UI alert. Probes 12/12, 0/12 FP; real obeyed claims 4/4 (3/4 before a disclosed widening). Open: import-name typosquats, insecure-config advice |
| T1 | Query/body content selects or widens the partition (verbatim other-partition text, chunk ids, filter syntax, JSON, fake SYSTEM override, `partition` body field) | Tenant separation | Not applicable before 15b (no partition existed) | **Defended (15b).** 0 out-of-partition chunks over 760 real Qdrant/BM25 runs (190 queries × 2 partitions × dense/hybrid); body fields → 422; mutation test fails closed |
| H3 | Model obeys direct injection (end to end, served pipeline) | Direct injection | v2: 1/28 complied (prompt exfiltration via translation) | **v3: 0/28** (15h) |
| H1 | Model obeys indirect injection from a retrieved chunk (5 payloads, rank 1, 10 queries each) | Indirect injection | v2: **5/50 obeyed** (typosquat `pip install` 4/10, `HACKED` 1/10); **all 4 obeyed install claims validated `SUPPORTED_LEXICAL`** | **v3: 0/50** (15h). Validator still certifies poisoned advice when it gets through (Open Question #20): not defended |

**After (2026-10-22):** every attack that succeeded before is defended or has a measured, stated residual. Defended: A1, A2 (across partitions), A3, A4, A5b–A5g, A6, P1, T1, A9 (install/link/secret class). Measured with the model as the defense: A8 (gate still passes 14/14 wrapped; model complied 0/14 under v3) and H1 (0/50 poisoned-chunk obedience under v3). Not a defense, by measurement: A7 (input injection regex, flag only). Accepted by design: A5a (`/chat` public), V1 (vectors are sensitive data; access control is the defense).

### Exit criteria (plan.md Phase 15)

| Criterion | Status | Evidence |
|---|---|---|
| `tests/security/` with ≥ 10 attacks | ✅ | 31 tests across 4 files (HTTP surface 12, indirect injection 7, retrieval partition 5, tenant bypass 3 + parametrized cases); each asserts the secure behavior, and succeeding attacks were strict-xfail until their defense landed |
| Injection attempts fail, documented before/after | ✅ with stated residuals | Indirect: A3 forgery 0 blocks (v3); A4/A9 never validated; poisoned-chunk obedience 5/50 → 0/50 (H1). Direct: complied 1/28 → 0/28 end to end (H3). Residuals: the confidence gate doesn't stop wrapped injections (A8); input-side detection 0/14 on paraphrases (A7); qwen/gemini fallbacks untested against poison; insecure-config advice and import-name typosquats not covered by an output check |
| Tenant bypass provably impossible | ✅ at API + retrieval layers | Structural: `PartitionedRetriever.search` takes no filter (TypeError), fails closed on any out-of-partition chunk (`test_partitioned_retriever_fails_closed_if_the_base_leaks`), pipeline refuses a mismatched retriever, request bodies naming a partition → 422. Empirical: A1 609/1,800 → 0/1,800; 0 out-of-partition over 760 real runs (re-run 15i). Scope limit: anyone holding the local Qdrant admin key is outside this proof |
| `docs/results/security-assessment.md` | ✅ | This file |

**Before:** 12 attacks succeed and 1 (A5f) is defended. A7 is a missing control that was probed, not an attack. Counting the individual tests: 11 xfail, 5 controls/defended.

## PII exposure inventory (counts only; values never written out)

| Source | email | URL with credentials | JWT-shaped | API keys / private keys |
|---|---|---|---|---|
| Dev queries (180) | 2 | 1 | 0 | 0 |
| Corpus chunks (19,236) | 103 (72 in issues) | 30 | 5 | 0 |
| Real interactions store (12 texts) | 0 | 0 | 0 | 0 |

The corpus figures are public GitHub content, re-served. A heuristic marks 121/140 email matches and 26/37 URL-credential matches as placeholders. Not verified by hand.

Data flows that carry user text: the interactions sqlite, `candidates_feedback.jsonl` (14g), the exact-match cache, and **Langfuse Cloud** (a third party: full query and chunk text in every trace), plus Groq/Gemini prompts. **Since 15d all of them receive boundary-redacted text.** Corpus PII (public GitHub content) is not redacted, deliberately (15d §6).
