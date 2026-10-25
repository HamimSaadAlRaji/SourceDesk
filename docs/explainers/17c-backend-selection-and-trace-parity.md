# 17c — Selecting the Backend, and Whether the Trace Survives

**Phase:** 17 · **Date:** 2026-10-25 · **Files:** `src/aegis/factory.py` (`pipeline_class`), `experiments/17a_trace_parity.py`, `tests/test_graph_pipeline.py`
**Topics from the course:** Logging, monitoring & observability

## 1. What I built

`factory.pipeline_class(backend)` and a `backend` argument on `build_pipeline(s)`, so the API server (which builds through the factory) and every experiment pick the orchestration from `ORCHESTRATION_BACKEND`. And an experiment that runs both pipelines under a real Langfuse/OpenTelemetry pipeline with an in-memory exporter, to compare span trees.

## 2. Why this approach

The unit tests use a `NullTracer` that records names. That would pass even if LangGraph ran nodes on another thread and the real OTel context did not propagate, producing orphan spans. A port that silently breaks Phase 13's tracing is not behavior-preserving, so it needed a check against the real tracer, not a fake.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Assert on the `NullTracer` only | Already there, free | Cannot see context propagation, which is the actual risk. |
| Send real traces to the local Langfuse | The most realistic | Needs the Langfuse stack up and leaves test data in the project; an in-memory exporter on the same SDK answers the parenting question without it. |
| A backend flag per request | Compare live in production | A flag on the request path is a way to ship the experimental path to users; it is a deployment-level setting. |

## 4. How it works

`Langfuse(..., tracer_provider=TracerProvider(+InMemorySpanExporter))` makes the SDK emit real OTel spans into memory. The script builds each pipeline with a client keyed `pk-<ClassName>` (see §6) and prints every span with its parent.

## 5. What the numbers said

```
AnswerPipeline traces: 1                LangGraphPipeline traces: 1
   cache_lookup <- answer_query            cache_lookup <- answer_query
   routing      <- answer_query            routing      <- answer_query
   retrieval    <- answer_query            retrieval    <- answer_query
   confidence   <- answer_query            confidence   <- answer_query
   generation   <- answer_query            generation   <- answer_query
   validation   <- answer_query            validation   <- answer_query
   answer_query <- None                    answer_query <- None
identical span trees: True
```

LangGraph adds **no spans of its own** and keeps one trace with every stage a direct child of the root. (Nodes are not spans here; the stage methods open them. That is a choice of this port, not something LangGraph does for free: a LangChain callback handler would have produced its own, differently shaped, spans.)

## 6. What surprised me / what to watch for

The first run reported `LangGraphPipeline traces: 0`. Not a LangGraph problem: Langfuse keeps **one resource manager per public key**, so the second client reused the first one's exporter and its own never saw a span. Distinct keys fixed it. Anyone comparing two pipelines against two Langfuse clients in one process hits this.

## 7. Where this fits and what's next

Together with 17b this covers "same flow, same trace". 17d is the equivalence on real data.
