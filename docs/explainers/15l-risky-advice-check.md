# 15l — The Validator No Longer Certifies Poisoned Advice (Open Question #20)

**Phase:** 15 · **Date:** 2026-10-22 · **Files:** `src/aegis/guardrails/risky_advice.py`, `src/aegis/guardrails/citations.py` (`UNVOUCHED_RISKY_ADVICE`, `vouched=`), `src/aegis/pipeline.py`, `src/aegis/events.py`, `src/aegis/api/static/app.js`, `src/aegis/config.py` (`RiskyAdviceSettings`), `data/security/risky_advice_probes_v1.jsonl`, `experiments/15l_risky_advice_fp.py`, `tests/test_risky_advice.py` (+18), `tests/test_pipeline_stream.py` (+1)
**Topics from the course:** Indirect prompt injection, output-side guardrails (Module 5, #36); citation grounding (Module 4)

## 1. What I built

An output-side check in the citation validator, built like 15e's commitment check. A sentence that tells the user to do one of three things is never validated, whatever it cites, unless the **documentation corpus** (the trusted half; never issue comments) vouches for that specific package or host:
1. **install** a package the docs never mention, or a raw URL / VCS target;
2. **open a link** to a host no documentation chunk links to;
3. **hand over a secret** ("paste/share/send … your API key / password / token / secret / .env"), unless negated.

Such a sentence gets the status `UNVOUCHED_RISKY_ADVICE`: no validated citation, a `done.warnings=["unvouched_risky_advice"]` flag with a UI alert, and the answer is **never cached**.

## 2. Why this approach

15h's worst finding: when v2 relayed a poisoned comment's `pip install fastapi-securefix`, the validator scored it `SUPPORTED_LEXICAL` in 4/4 cases. "Supported" means "a retrieved block says so", and issue comments are attacker-writable, so a validated citation made malicious advice look *more* trustworthy. v3 stopped the model relaying it (0/50), but the prompt is one layer and the fallback models are untested. The check holds when the model doesn't comply, which is the same argument 15e made for commitments.

**Why corpus-wide vouching by documentation, not "the chunks this answer cites":** legitimate fixes often come from issues but use documented packages ("install python-multipart" cited to a closed issue). The question the check asks isn't "who said it in this context" but "does the trusted source know this thing exists". `fastapi-securefix` appears in zero documentation chunks; `uvicorn`, `python-multipart` and `fastapi.tiangolo.com` appear in many.

**Fail closed:** if the documentation corpus is missing, nothing is vouched, so every install or link fires (tested).

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
|---|---|---|
| Provenance demotion: any claim supported only by an issue chunk is "unverified" | General; covers non-install poison too | Issues are 16,944 of 19,236 chunks and the whole `known_issue` value. It would demote most legitimate known-issue answers to fix one class. The risky class is narrow and checkable. |
| LLM judge: "is this advice safe?" | Catches paraphrase and novel payloads | The judge reads the attacker's text and is itself injectable; it's also a paid call per sentence in the serving path. Deterministic rules can't be talked out of a verdict. |
| Allowlist from `pip install` lines in the docs only | Precise | Measured: the docs often quote the target (`pip install "fastapi[standard]"`) or mention packages in prose, so the pip-line extraction missed `uvicorn` entirely. It would block legitimate advice. |
| Rely on prompt v3 | Already 0/50 in 15h | One layer. qwen/gemini fallbacks were never tested against poison, and 15h was rank-1, n=10 per payload. |

## 4. How it works

```python
vouched = Vouched.from_texts(doc_chunk.text for doc_chunk in corpus if doc_type == "documentation")
#   tokens: every word-like token in the docs (11,728)   hosts: every linked host (131) + loopback

risky_reasons(sentence):
  for target in install_targets(sentence):           # "pip/uv/poetry/pipx/conda install|add ..." and prose "install(ing) [≤3 descriptors] <name-with-hyphen>"
      URL/VCS/path target            -> install_from_url
      base name not in vouched.tokens -> unvouched_package
  for host in links(sentence): host not in vouched.hosts -> unvouched_host
  "paste|share|send|post|… … your … key|token|password|secret|.env|credential" and not negated before it -> secret_request
```

In `CitationValidator.validate`, this runs right after the commitment check and before the short-sentence exemption and every citation check. A flagged sentence makes no entailment call.

## 5. What the numbers said

All zero LLM cost (`experiments/15l_risky_advice_fp.py`, raw `docs/results/risky-advice-fp-raw.json`).

| Set | Result |
|---|---|
| Frozen probes (sha256 `8ba1ac49…`, written before the first measurement) | **12/12 risky caught, 0/12 legit fired** |
| 15h's real `fastapi-securefix` sentences (4 obeyed + 2 mentions) | **3/4 obeyed caught at the frozen patterns**; 6/6 after widening (below) |
| Every stored generated answer (bake-off, prompt-fit, 15h H2/H3; 1,162 sentences) | 0 fired |
| Documentation sentences (14,824) | 0 fired |

**What didn't work, disclosed:** the first run missed a real obeyed answer: "The recommended first step is to install **the patch package** fastapi-securefix". The prose pattern allowed one descriptor word, not two. I widened it to up to three descriptor words and parenthesized names (`RiskyAdviceSettings.install_prose`, with a comment). The 6/6 is therefore **not held-out**: the widening was made after seeing this sentence.

**How much the zero false-positive result is worth:** the real answers contain **0** install sentences, 6 links and 10 secret-type nouns. So the 0/1,162 exercises the link and secret triggers lightly and the install trigger not at all. Install false positives rest on the 12 legitimate probes and on the documentation sentences, and the latter is partly circular, because anything the docs mention is vouched by definition. The secret-request check on the documentation *is* a real test (0/14,824, including the OAuth tutorials).

`uv run pytest -q`: **745 passed** (+18 risky-advice tests, +1 pipeline stream test; validator/pipeline tests red first).

## 6. What surprised me / what to watch for

- **Import-name typosquats slip through.** Vouching is by token, and the docs contain import names (`pydantic_settings`) as well as distribution names (`pydantic-settings`). `pip install pydantic_settings` points at a different PyPI name but would be vouched. A distribution-name list would close this, at the cost of maintaining it.
- **Not covered: insecure configuration advice.** The 15h forged-block payload ("set `allow_origins=["*"]` with `allow_credentials=True`") is neither an install nor a link, so this check wouldn't catch it. In 15h the model ignored it under both prompts, and that is the only defense measured for it.
- **Mentions fire too.** "The issue comment recommending pip install fastapi-securefix is unrelated" is flagged even though it warns against the package. That's the safe direction (no validated citation, a UI warning), but a careful answer gets the same alert as an obeying one.
- **The vocabulary is the whole doc corpus,** so common words are vouched. A package literally named like a common docs word would pass. The hyphen requirement in the prose trigger and the command parser keep this narrow, and it's still the main residual.

## 7. Where this fits and what's next

This closes Open Question #20 for its measured class (install / link / secret-request). Insecure-configuration advice and import-name typosquats remain open (logged in `progress.md`). With #18–#20 done, Phase 15's remaining unit is **15i, the wrap-up**.
