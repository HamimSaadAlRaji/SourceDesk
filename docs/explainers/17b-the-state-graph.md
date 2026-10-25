# 17b — The StateGraph: Same Stages, Different Control Flow

**Phase:** 17 · **Date:** 2026-10-25 · **Files:** `src/aegis/graph/pipeline.py`, `src/aegis/pipeline.py` (two seams), `tests/test_graph_pipeline.py`, `tests/test_pipeline.py` (backend switch)
**Topics from the course:** Agentic RAG (sequential, conditional, iterative, parallel workflows)

## 1. What I built

`LangGraphPipeline`, a subclass of `AnswerPipeline` whose request flow is a compiled `StateGraph`: ten nodes (`rewrite`, `cache`, `serve_cached`, `route`, `escalate_route`, `retrieve`, `confidence`, `reformulate`, `escalate_confidence`, `generate`) and four conditional edges (after start, cache, route, confidence). Every node calls the stage method the direct path already has.

## 2. Why this approach

The risk of a port is not the graph, it is a second copy of the logic that drifts. So only the *orchestration* moves. `AnswerPipeline._answer` was split into two seams (`_orchestrate`, `_serve_cached`); the graph overrides the first and reuses the rest, which means spans, timings, streamed events, cache scope and the audit trail are the **same code** under both backends. What can differ is only *which stage runs next*, and that is what 17d measures.

Per-run objects (the timings dict, the event sink) go through `config["configurable"]`, not graph state. State is what a checkpointer would persist; a callback is not state.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Reimplement each stage as a node (no subclass) | A "pure" LangGraph app | A second copy of tracing, timing and streaming that has to be kept in sync with the first; the equivalence run would then be testing two implementations, not two orchestrations. |
| Graph replaces the whole `_run` | Rewrite becomes an ordinary node either way | `_run` owns the root span, logging, the `done` event and `answer_stream`'s thread hand-off; moving them into nodes risks the OTel same-thread rule (14a). |
| `Command`/`Send` dynamic routing | Idiomatic in newer LangGraph | Routing here is a pure function of state; conditional edges name the topology in one place, which is the readability the port is supposed to buy. |
| Put `emit` and `timings` in state | One mechanism | They are not serializable state, and would be persisted by any checkpointer. |

## 4. How it works

```python
g.add_conditional_edges(START, self._after_start, {"rewrite": "rewrite", "cache": "cache"})
g.add_conditional_edges("cache", self._after_cache, {"serve_cached": "serve_cached", "route": "route"})
g.add_conditional_edges("route", self._after_route,
    {"escalate_route": "escalate_route", "retrieve": "retrieve", "done": END})
g.add_edge("retrieve", "confidence")
g.add_conditional_edges("confidence", self._after_confidence,
    {"reformulate": "reformulate", "escalate_confidence": "escalate_confidence", "generate": "generate"})
g.add_edge("reformulate", "retrieve")      # the retry loop: the only back-edge
```

`_after_confidence` is the only place the retry decision lives: generate if confident, else `reformulate` if the flag is on and `retries < max_retries`, else escalate. A retry changes only the *retrieval* query (`retrieval_query`); the answer, the cache key and the escalation payload still use the user's question (`effective`). `_accumulating` makes a stage that runs twice report its total.

## 5. What the numbers said

Built test-first (`tests/test_graph_pipeline.py`, 19 tests): the same fakes through both classes, comparing the whole `PipelineResult` (minus trace id and timings), the span sequence, the cache writes and the streamed events, across 7 scenarios (exact hit, semantic hit, policy escalation, router unavailable, low confidence, generated, generation unavailable). Then the entire pre-existing suite was re-run against the graph with `AEGIS_TEST_BACKEND=langgraph`:

```
$ uv run pytest -q                                   862 passed
$ AEGIS_TEST_BACKEND=langgraph uv run pytest -q      862 passed
```

Mutation checks (tests must be able to fail): retry cap `<` → `<=` fails the cap test; generating with the retry's query instead of the user's fails the retry-answer test; removing the cache-hit edge fails 3 differential tests.

## 6. What surprised me / what to watch for

- The tests passed on the first green run, which is only worth anything because of the mutation checks above.
- The retry edge exposed a design choice the plan did not mention: *which* query the answer and the cache use after a reformulation. If the cache were keyed on the reformulated text, the user's own repeat question would never hit.
- Not yet checked on a real run: a retry on real data. It is covered by unit tests only (see 17d).

## 7. Where this fits and what's next

17c makes it selectable and checks that tracing survives; 17d runs it against the dev set.
