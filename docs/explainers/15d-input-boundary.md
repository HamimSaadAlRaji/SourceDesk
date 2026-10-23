# 15d — The Input Boundary: Redaction, and Why Injection Detection Is Only a Flag

**Phase:** 15 · **Date:** 2026-10-18 · **Files:** `src/aegis/guardrails/input_boundary.py`, `src/aegis/config.py` (`InputBoundarySettings`), `src/aegis/api/app.py`, `tests/test_input_boundary.py`, `tests/security/test_attack_http_surface.py`, `data/security/attacks_v1.jsonl`, `data/security/pii_probes_v1.jsonl`, `experiments/15d_input_boundary_comparison.py`, `experiments/15d_deberta_padding_evasion.py`, `experiments/15d_redaction_retrieval_impact.py`, raw `docs/results/{input-boundary-comparison,deberta-padding-evasion,redaction-retrieval-impact}-raw.json`
**Topics from the course:** Security — LLM data leakage, input validation (Module 5); guardrails

## 1. What I built

Every user-supplied string goes through `redact()` at the HTTP boundary: the query, each history turn, and the feedback comment. Eight high-precision secret/PII shapes become `[REDACTED:<kind>]` before anything downstream sees them. A regex injection heuristic logs `input.injection_flag` and **never blocks**.

Both choices were measured against the libraries they replace: protectai's DeBERTa injection classifier, and Presidio.

## 2. Why this approach

**Redact at the boundary, once, because the raw text had seven destinations.** Before this unit, a pasted `postgresql://admin:S3cretPass@…` went to:
- the pipeline's caches (both);
- the interactions sqlite;
- `candidates_feedback.jsonl`;
- Langfuse Cloud (third party) in every span's input;
- Groq and Gemini (third parties) in the router, rewriter and generation prompts.

Redacting at storage would have covered two of those seven. Redacting before the pipeline covers all of them with one call, and the measured cost is zero (§5: retrieval unchanged on every dev query it touches).

**Hand-rolled regex beats Presidio *on this corpus* in both directions.** Presidio's NER is built for prose about people. Questions here are code, stack traces and log lines. On the 180 dev questions it flags **135** (score ≥ 0.5): `routing.py`, `app.ge`, `uvicorn.run` and `self.app` as URLs, 97 PERSON spans, "Linux" as a LOCATION. Meanwhile it misses 4 of the 8 secret shapes this system actually receives (GitHub token, private key, bearer token, and a JWT it only matches as a "URL"), and labels the OpenAI and AWS keys "PERSON". The regex flags 3 dev questions, all real (2 emails, 1 credentialed URL), and catches 8/8 secret probes with 0/6 false positives on code-shaped negatives.

**Injection detection is only a flag, because neither detector holds against a cheap evasion.** The plan's build list says "injection heuristics", and the course material treats input filtering as a layer. Measured here:
- the regex misses **every one of 14 paraphrases** (0/14);
- DeBERTa catches all 14, but **misses 7/7 when the injection is appended to a real pasted issue** (0/7). It only reads 512 tokens, and the corpus routinely has 1,000-token questions. Scanning in windows gets 4/7, at 4.7% false positives on legitimate questions and 556 ms per query.

A blocking detector that a paraphrase or a pasted stack trace beats is a false sense of security with a real cost. What has to hold is structural: the router's billing hard-route (Phase 8), the partition (15b), and the output-side checks in 15e. The flag stays because it's free and gives an operator a signal for naive attempts, which are most attempts.

**Decision record:** the user chose "hand-rolled, flag-only, measured against DeBERTa/Presidio" (2026-10-18) *before* these numbers existed. DeBERTa's 14/14 on paraphrases contradicted the expectation, so I stopped and investigated (the padding and windowing tests) before proceeding. The decision stands on the numbers, not on the earlier preference.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here (measured) |
|---|---|---|
| **DeBERTa injection classifier, blocking or escalating** | 14/14 paraphrases, 7/7 golden; `transformers` is already installed (via sentence-transformers), so it's model weights, not a new library | Padding a real issue: **0/7** at the end, **2/7** at the start. Windowed: 2/7 and 4/7, with **7/150 (4.7%)** false positives and **556 ms/query**. Single-pass: 1/150 + 1/23 false positives, one of them a legitimate **Chinese-language** question (0.999), and 194–210 ms mean / 354–370 ms p95 added to a path already at 5.6 s p95 vs a 3 s target. |
| DeBERTa as a flag only, alongside the regex | Better coverage of short paraphrases | ~200 ms per uncached request for a log line, which a padded injection still evades. Possible later as an *offline* scan of corpus chunks (chunks are short, so truncation doesn't apply): 19k chunks × ~200 ms ≈ 64 min of CPU, which is past the 10-minute rule, so it would be a Kaggle job. Noted for 15e. |
| **Presidio** for PII | Mature; finds names, phones, locations the regex deliberately skips | Flags 135/180 legitimate dev questions; misses 4/8 secret shapes; 67.6 ms mean / 211.8 ms p95. Pulls in spaCy + a 400 MB model. The things it uniquely finds (names, phone, city: probes 8–9) are rare in FastAPI questions and are what its false positives are made of. |
| Redact at storage only (store, export) | No change to what the model sees | Leaves Langfuse Cloud, Groq, Gemini and both caches with the raw text: 5 of 7 destinations. |
| Keep `override` in the regex | It caught golden injection #9 ("SYSTEM OVERRIDE") | It caused all 4/150 dev false positives (FastAPI's `dependency_overrides`). As a flag this would only be log noise, but the pattern was frozen *before* the evasion measurement and I didn't re-tune it on the evaluation set. Cost: golden 7/7 → **6/7**. |

## 4. How it works

```python
def redact(text):                       # (redacted_text, {kind: count}), idempotent
    text = _URL_PASSWORD.sub(...)       # FIRST: scheme://user:PASS@host -> scheme://user:[REDACTED:password]@host
    for kind, pattern in _PII:          # private_key_block, jwt, github_token, openai_style_key,
        text = pattern.sub(...)         # aws_access_key, bearer_token, email  (InputBoundarySettings.pii_patterns)
```

- **Order matters, and a test caught it.** With email first, `admin:S3cretPass@db.internal` became `admin:[REDACTED:email]:5432`: the host was eaten and the query no longer said what failed. URL credentials now run first.
- **Only the password is replaced in a credentialed URL**, so "why does my postgres URL fail" survives.
- **Placeholders match no pattern**, so `redact(redact(x)) == redact(x)` (tested).
- **Wiring in `/chat`:** the query and each history turn are redacted before `Query(...)` is built, so the pipeline, caches, store and rewriter only ever see redacted text. The `/feedback` comment is redacted before the store. `input.redacted` logs the kinds and counts, never the values. `input.injection_flag` logs at most 5 matched phrases.

**The attack and probe sets** (`data/security/`, versioned; sha256 recorded in each raw result; frozen at phase end):
- `attacks_v1.jsonl`, 35 rows:
  - 7 references to dev golden injections by ID (text not copied, so they stay comparable with Phase 10);
  - 14 wrapped (both orders, same construction as 15a A8);
  - 14 hand-written paraphrases avoiding the obvious keywords, including one in Spanish and one base64-encoded.
- `pii_probes_v1.jsonl`, 16 rows: 10 with fake but correctly shaped secrets/PII and 6 code-shaped negatives.

## 5. What the numbers said

`uv run pytest -q`: **661 passed, 2 xfailed** (+17 `test_input_boundary.py`, +1 end-to-end boundary test). The 2 xfails are still A3/A4.

**Injection detection** (`15d_input_boundary_comparison.py`, attacks sha256 `d2925bcd…`):

| Set | Regex (flag) | DeBERTa (single pass) |
|---|---|---|
| Dev positives, false positives | **0/150** | 1/150 |
| Other dev negatives, false positives | 0/23 | 1/23 |
| Golden injections | 6/7 | 7/7 |
| Wrapped | 12/14 | 12/14 (misses differ from the regex's) |
| Paraphrase | **0/14** | **14/14** |
| Injection + 984-token real issue, at start / end | 6/7 / 6/7 | **2/7 / 0/7** (windowed: 2/7 / 4/7) |
| Latency | ~0 | 194–210 ms mean, 354–370 ms p95 (two runs); windowed 556 ms |

**PII** (`pii_probes_v1` sha256 `a375eb6c…`; Presidio default `AnalyzerEngine`, `en_core_web_lg` 3.8.0, run in a `uv run --with` overlay, not a project dependency):

| | Regex | Presidio |
|---|---|---|
| Secret probes found (8) | **8/8** | 4/8, and 2 of those mislabeled PERSON |
| Name/phone/location probes (2) | 0/2 (by design) | 2/2 |
| Code-shaped negatives flagged (6) | **0/6** | 4/6 |
| Dev questions with anything flagged (180) | **3** (all real) | **135** at score ≥ 0.5 |
| Latency | ~0 | 67.6 ms mean, 211.8 ms p95 |

**Redaction's cost to retrieval** (`15d_redaction_retrieval_impact.py`, served dense path): `redact` changes 3/180 dev questions. For all 3: **top-10 overlap 10/10, same top-1, ground-truth hits identical** (8/8, 6/6, 8/8). Zero measured cost.

**End to end (TestClient):** a query with a credentialed URL and an email, a history turn with a GitHub token, and a feedback comment with an email. None of the 4 secrets appears in what the pipeline received, in the stored query, answer or comment, and the placeholders are present.

## 6. What surprised me / what to watch for

- **The classifier that looked decisively better failed the cheapest attack this corpus invites.** Pasting a full issue body is normal here: 43 golden queries are over 2,000 characters. The headline "14/14 vs 0/14" alone would have argued for adopting it. The padding test is what the decision rests on.
- **The paraphrase set is mine and small (n=14), and probably in-distribution for DeBERTa**, which was trained on jailbreak-style data. 14/14 is an upper bound, not a robustness estimate.
- **DeBERTa flagged a legitimate Chinese-language question at 0.999.** That's n=1, but if it were blocking, this is exactly the shape of a fairness failure: non-English users escalated more often.
- **Presidio's false positives come from the domain, not from bugs.** Dotted Python paths look like hostnames, and library names look like people. It's tuned for a different text distribution, and the 135/180 says so.
- **The corpus itself still holds PII** (design record 15, A7: 103 chunks with emails, 30 credentialed URLs, mostly placeholders by heuristic). It's public GitHub content, re-served in citations. It isn't redacted, because that would change the indexed text every published number came from. Revisit only if the corpus stops being public data.
- **Redaction changes what the model sees.** A user debugging "why is my token rejected" now sends `[REDACTED:jwt]`. For the 3 dev questions it touches, retrieval didn't move, but answer quality on secret-dependent questions isn't measured (no quota). A question about *the token's contents* can no longer be answered. That's the intended trade.
- **Two caches were left on disk:** the DeBERTa weights (714 MB in `~/.cache/huggingface`) and the Presidio/spaCy overlay in uv's cache. The disk is 97% full. Neither is a project dependency, and both are safe to delete.

## 7. Where this fits and what's next

The input side is now honest about what it does: it removes secrets (measured, free) and *flags* injection (measured, evadable). The remaining exit criterion, "injection attempts fail, documented before/after", therefore has to be met **after** the input, where retrieved text and model output can be checked:

- **15e:** close A3 (forgeable context blocks) and A4 (the validator passing an injected commitment) with unforgeable per-request block delimiters and an output-side commitment check. Possibly prompt v3.
- **15g:** the real-stack before/after run on `attacks_v1.jsonl`, quota-bound.
