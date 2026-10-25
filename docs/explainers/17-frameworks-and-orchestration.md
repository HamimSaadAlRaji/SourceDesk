# 17 — Frameworks and Orchestration: What LangGraph Gave and Took

**Phase:** 17 · **Date:** 2026-10-25 · **Files:** `src/aegis/graph/pipeline.py`, `src/aegis/eval/equivalence.py`, `src/aegis/factory.py`, `src/aegis/pipeline.py`, `docs/results/framework-comparison.md`
**Topics from the course:** Agentic RAG (sequential, conditional, iterative, parallel workflows); Latency vs. response quality; Evaluation strategies (system vs. component)
Sub-records: `17a` dependency and switch · `17b` the graph · `17c` selection and trace parity · `17d` equivalence · `17e` latency · `17f` LOC and debuggability.

## 1. What I built

The request flow ported to a LangGraph `StateGraph` and run through the project's own harness against the original. Both are selectable by `ORCHESTRATION_BACKEND`. The graph is a subclass that replaces one seam, so everything except "which stage runs next" is shared code.

## 2. Why this approach

The goal was a defensible opinion, not a rewrite. A port that reimplemented tracing and caching would have been compared on two implementations; sharing the stage methods makes the comparison about orchestration alone, and makes "behavior-preserving" a checkable claim: same fakes through both (differential tests), the whole existing suite on both backends, and 180 real queries with every field compared.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Build on LangGraph from the start | Framework-native | Hides the architecture; the plan chose direct-then-port so the cost of the framework is observable. |
| Rewrite every stage as a node | Purest port | Two copies of tracing/timing/streaming to keep in sync; the equivalence result would test two implementations. |
| Compare aggregate metrics only | What the plan says | Cannot see per-query divergence (17d control: 59 queries differ with two of three metrics unmoved). |
| Run both paths live | No tape | Model noise (OQ #4) would be inside the comparison. |
| Enable the retry edge | Matches the plan | The direct path has no loop; Phase 8 measured it as a regression. Built, gated, off. |

## 4. How it works

See 17b for the graph (10 nodes, 4 conditional edges, one back-edge), 17d for the record/replay tape, and `docs/results/orchestration-graph.mmd` for the generated diagram.

## 5. What the numbers said

| Question | Result |
| --- | --- |
| Behavior preserved? | 180/180 dev queries identical in every field, cold and cached; 862 tests pass on both backends; identical span trees |
| Does the check detect a difference? | Yes: `tau` +0.1 → 59/180 differ |
| Time cost | +0.6 to +1.2 ms per request (stubbed stages); live paired median -22 ms, CI [-162, +124] ms: not detectable |
| Code cost | 61 → 184 lines deciding the flow; 7 → 13 traceback frames |
| Money | $0.274 (equivalence) + $0.264 (latency, of which $0.131 lost to a bug of mine) |

## 6. What surprised me / what to watch for

- The plan's retry loop does not exist in the system; the port can express it but must not enable it.
- Prompt v3's per-request random boundary key broke replay until normalized (129 replay misses in the first offline run), a side effect of a Phase 15 security feature on a Phase 17 test design.
- Langfuse keeps one resource manager per public key; two clients in one process silently share an exporter.
- The 11.8 s direct p95 is the finding with the most consequence, and it is not about LangGraph.

## 7. Where this fits and what's next

Verdict: keep direct as the default (`docs/results/framework-comparison.md`). Open and recorded in `docs/future-work.md`: the CI matrix leg (deferred by decision), the retry edge on real data, live-vs-live noise, and the p95. Phase 18 (deployment) inherits `ORCHESTRATION_BACKEND` defaulting to `direct`.
