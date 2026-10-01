# 13f — The Audit Trail

**Phase:** 13 · **Date:** 2026-09-30 · **Files:** `src/aegis/observability/audit.py`, `src/aegis/pipeline.py`, `tests/test_audit_trail.py`, `experiments/13j_audit_trail_demo.py`, `docs/results/audit-trail-example.md`
**Topics from the course:** Logging, monitoring & observability (Module 5, #31); Handling hallucinations (#25)

## 1. What I built

`build_audit_trail(observations)` rebuilds, from one Langfuse trace and nothing else, what produced an answer: the query, cache/route decisions, the retrieved chunks with scores, the confidence signals, the prompt version and generation parameters, **which chunks the LLM was actually shown under which `[n]`**, the answer, the validated citations, and — for escalations — the exact reason. `resolve_evidence` attaches the real chunk text and flags whether it has changed since. `render_markdown` writes it as a readable record. `fetch_observations` is the only I/O.

## 2. Why this approach

Air Canada's failure wasn't only a wrong answer; it was that nobody could show what the system had been told or why it said what it said. The worked reconstruction has to answer "what evidence produced this?" from the trace. **My first check found the trace could not**: it recorded the top-10 retrieved chunks and the validated citation ids, but not which chunks were packed into the prompt, their `[n]` numbering, truncation, or how many the token budget dropped. An auditor reading "...[1]" couldn't say what `[1]` was. That gap was fixed before building the reader (below), not worked around.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ------------------ |
| Read everything from the root span's stored `PipelineResult` | One object, trivial | Reconstruction from the root alone can't detect a partial or garbled trace. Per-stage spans are the independent path; the root is used as a cross-check (`inconsistencies`). |
| Store the full prompt text in the trace | "Exact" evidence | ~1-4k tokens per request, and it duplicates the corpus. The prompt is deterministic from `prompt_version` + ordered chunk ids + corpus text; recording ids + `content_hash` gives the same answer plus drift detection. |
| Write hand-built observation dicts for the tests | Simpler tests | Guards nothing. The real risk is format drift; the tests generate observations from a real `AnswerPipeline` run so changing a span's shape breaks a test. |
| Parse the answer's `[n]` back to chunks by re-running the budgeter | No trace change | Not reproducible if retrieval or the corpus changed. Recording the map is the audit-grade answer. |

## 4. How it works

The pipeline now records `ContextEntry(citation_id, chunk_id, content_hash, score, token_count, truncated)` per packed chunk, on both `PipelineResult.context` and the `generation` span's input (with prompt version, temperature/top_p/max_tokens/budget, total tokens, dropped count). Reconstruction:

```python
obs   = fetch_observations(client, trace_id)      # polls; fields="core,basic,io"; JSON strings -> objects
trail = build_audit_trail(obs, trace_id)          # pure
evid  = resolve_evidence(trail, chunk_by_id.get)  # ok | changed | missing | unverified
print(render_markdown(trail, evid))
```

Failure is loud, not silent: an expected stage that's absent, a generation span with no context map, or spans that disagree with the root's recorded result each produce an explicit `warnings` / `inconsistencies` entry. Which stages are "expected" depends on the branch the root result says was taken (cache hit → 1 stage; hard escalation → 3; confidence escalation → 5; answer → 6).

## 5. What the numbers said

`uv run pytest -q`: 496/496 (+10 audit tests, +3 context-recording tests, TDD'd RED-first).

**Live, real end-to-end** (real Qdrant, cross-encoder, Groq router + generation, Langfuse): three queries — an answered one (`golden::issue::604`), a hard-route escalation (`negative::billing_legal::4`), and a confidence escalation (`golden::issue::2689`, `confidence 0.2295 < tau 0.5000`). Each was reconstructed from its stored trace and compared field-by-field to the object the live pipeline returned: **0 warnings, 0 inconsistencies, 0 differences on all three** (cache status, route, outcome, answer text, citations, provider, prompt version, retrieved ids and scores, context map incl. content hashes, confidence). The answered case shows `[1]` resolving to `issue::4842::heading_aware::8` and its real text, and 9 other chunks that were shown but not cited. Output: `docs/results/audit-trail-example.md`; logs joined by trace ID in `audit-trail-example.log.jsonl`.

## 6. What surprised me / what to watch for

**The API hid the data.** Observations returned by Langfuse's v2 API omit `input`/`output` unless you pass `fields="core,basic,io"`, and return them as JSON *strings*. A reconstruction built from the default response would have looked "empty" and silently useless.

**What the trace still does not record** (limits of the audit, not verified otherwise): per-claim validation results (only `claim_count` and the validated citation ids, not which claim each citation supported); the exact prompt text (reconstructable, not stored); the actual model name (the generation span's `model` field is the client class name, `FallbackLLMClient`, not the model that answered — only the provider name is recorded); `tau` is present only inside the escalation reason string.

**Not yet a legal artifact.** Langfuse Cloud's free tier retains data for **30 days**. An audit trail that evaporates after a month is a demo, not a compliance record. Retention/export to durable storage is unbuilt, and is the largest gap between this and the "documentation trail Air Canada lacked."

**Drift is detectable, not prevented.** `content_hash` comparison flags a chunk whose text changed or vanished since the answer (unit-tested with a mutated and a deleted chunk). It has not been exercised against a real re-index.

**Incidental timing, not evidence:** the live answered request took ~1.9 s end to end (routing 573, retrieval 62, confidence 322, generation 642, validation 295 ms), unpaced and hot, n=1. That is under 3 s and doesn't reproduce the ~2.4 s confidence stage from the 70 s-paced batches; it also cuts against the (still-untested) hypothesis that running real generation in-process slows the cross-encoder. One request proves neither.

## 7. Where this fits and what's next

This closes Phase 13's build list (tracing, structured logging, latency attribution, warm-up, audit trail). Exit criteria: every request traced ✓ (`AnswerPipeline`; no HTTP entry point until Phase 14), `latency-budget.md` ✓, worked audit-trail reconstruction ✓, **p95 < 3 s ✗ unmet**. Phase 14's `GET /trace/{id}` is this module behind an endpoint; Phase 16's CI gate needs the p95 question settled or an honest, explained threshold.
