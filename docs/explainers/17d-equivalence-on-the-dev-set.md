# 17d — Is the Port Behavior-Preserving? 180 Queries, Real LLMs, Both Paths

**Phase:** 17 · **Date:** 2026-10-25 · **Files:** `src/aegis/eval/equivalence.py`, `experiments/17b_equivalence.py`, `tests/test_equivalence.py`, `docs/results/orchestration-equivalence-raw.json`, `docs/results/orchestration-equivalence-raw-control.json`
**Topics from the course:** Evaluation strategies (system vs. component, code-based vs. LLM-judge vs. human feedback)

## 1. What I built

An equivalence harness: the 180 dev queries run through `AnswerPipeline` and `LangGraphPipeline` end to end (router → retrieval → confidence → generation → validation → cache), with every field of every `PipelineResult` compared. The direct path runs live and records every LLM call (router, generator, citation entailment) on a tape; the graph replays it.

## 2. Why this approach

A paid temperature-0 model is not bit-reproducible (Open Question #4). Running both live would put model noise into a comparison meant to isolate orchestration. Record/replay removes the noise and makes the question exact: a replayed run can differ only if the graph *asks a different question* (a replay miss, which raises) or *uses the answer differently* (a field diff). The tape also reports entries the graph never used, the mirror image of a miss.

The existing CI gates could not do this job: they call components directly, so none of them executes the orchestration at all. Reranker scores come from the 16i cache (0 misses), and retrieval excludes each query's source issue exactly as the Phase 10 gate does, so the run is also an end-to-end re-measurement of Phase 10 through the real pipeline.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Run both live and compare metrics | Simplest, no tape | Generation text differs run to run; metric equality would need a tolerance that could hide a real divergence. Costs 2× ($0.55). |
| Compare only aggregate metrics | What the plan literally says | Aggregates are blunt: in the control, 59 queries differ while two of the three gate metrics (`air_canada_fn`, negative recall) do not move at all. Per-query field identity is strictly stronger. |
| The 40-query subset | Cheaper | The 180 run cost $0.274; the subset would save ~$0.2 and miss the rarer branches (policy escalation, warnings). |
| Semantic cache on | It is on in production | Hit pattern depends on arrival order, and the queries run on 8 threads; the cache-hit branch is exercised instead by asking every query twice (exact cache). |

## 4. How it works

`Tape.call(kind, args, fn)` keys on `sha256([kind, args])`; `kind` (router / generation / entailment) stops the same text sent to two stages from colliding. Prompt v3 draws a fresh random boundary key per request (Phase 15's injection defence), so two identical requests never match byte for byte; `_without_boundary_key` replaces exactly the `key: <16 hex> ===` tokens before keying and nothing else (tested both ways: key-only differences collapse, content differences do not).

## 5. What the numbers said

Real run (direct live, graph replay), `docs/results/orchestration-equivalence-raw.json`:

| | Pass 1 (cold) | Pass 2 (cache) |
| --- | --- | --- |
| Queries identical, every field | **180 / 180** | **180 / 180** |
| Differing | 0 | 0 |
| Cache statuses (both paths) | 180 miss | 128 exact hit, 52 miss |
| Escalated / answered (both paths) | 51 / 129 | 51 / 129 |
| `air_canada_fn` / negative recall / deflection (both) | 3 / 0.90 / 0.84 | same |
| `cited_answer_rate` (both) | 0.8992 | same |
| Unused tape entries / replay misses | 0 / 0 | |
| Reranker cache | 1260 hits, 0 misses | |

The direct pipeline's end-to-end escalation numbers equal the Phase 10 gate's (3 / 0.90 / 0.84), so the pipeline and the standalone gate agree. Spend **$0.274** against a $0.28 estimate; direct path wall time 79 s on 8 threads (graph's 1.6 s is replay, **not** a latency measurement).

**Negative control** (graph with `tau` +0.1, replaying the same tape, $0): **121 identical, 59 differing**; 19 queries flip answered → escalated, deflection 0.84 → 0.713. `air_canada_fn` and negative recall do not move (3 / 0.90), which is why aggregates alone would not have shown it.

## 6. What surprised me / what to watch for

- Pass 2 shows 128 hits for 129 answers: one answer carries a guardrail warning and is deliberately never cached (15e/15l). The run log has 4 `guardrail.answer_flagged` lines, which is one answer across four runs (two paths x two passes).
- My first control, the retry edge on, could not run: the retry asks new questions, and the reranker miss budget refused (63 > 60 uncached pairs) rather than burning ~3 min of CPU. The budget did its job. Consequence, stated plainly: **the retry edge has been verified by unit tests, not on real data**.
- A first offline wiring run exposed the v3 key problem (129 replay misses) before any paid call.
- Limits: 180 dev queries, one run; replay proves equal *decisions given the same LLM text*, not that the two would agree if both were live (they would differ by model noise alone). That is the intended question, but it is the scope.

## 7. Where this fits and what's next

Exit criterion 1 ("passes the full eval suite, metrics matching within tolerance") is met with tolerance **zero**. 17e measures what the framework costs in time, with the fresh direct p95 the Phase 16 hand-off asked for.
