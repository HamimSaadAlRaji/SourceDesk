# 17a — The LangGraph Dependency and the Orchestration Switch

**Phase:** 17 · **Date:** 2026-10-25 · **Files:** `pyproject.toml`, `uv.lock`, `src/aegis/config.py` (`OrchestrationSettings`)
**Topics from the course:** Agentic RAG (sequential, conditional, iterative, parallel workflows)

## 1. What I built

A pinned `langgraph>=1.2.2,<2` dependency and an `OrchestrationSettings` group: `backend` (`direct` | `langgraph`, default `direct`), `retry_on_low_confidence` (default `False`) and `max_retries` (1).

## 2. Why this approach

`CLAUDE.md` §2 says a new dependency needs a justification here. The justification is the phase's purpose: framework fluency is a screening filter, and a *measured* port is a stronger answer than a preference. LangGraph was already installed in the uv environment at 1.2.2 as a transitive of the LangChain packages, but was **not declared**; relying on it undeclared would have broken a clean checkout (the CI rehearsal in 16a uses one). The upper bound `<2` is there because LangGraph's `StateGraph` API has changed between majors.

The switch defaults to `direct` so that nothing changes for the API, CI or any experiment until it is asked to. The retry flag defaults off for a reason found while reading the plan: `plan.md` says the retry loop on low confidence "maps onto a graph naturally", but **the direct pipeline has no retry loop**. Iterative retrieval was built in Phase 8, measured at −13.4% NDCG@5 and shipped off. A port that turned it on would change behavior, and the equivalence run could not then show that the port preserved it.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Rely on the transitive install | No `pyproject` change | Undeclared; a fresh `uv sync` from the lockfile would only work by accident of the LangChain pins. |
| Enable the retry edge by default | Matches the plan's wording | Changes behavior; measured as a regression when it was a retriever wrapper (Phase 8). |
| Leave the retry edge out entirely | Smallest port | Loses the one thing the plan says a graph expresses well; a config-gated edge costs ~10 lines and is unit-tested. |
| LangChain's `RunnableSequence` / agent executors | More "standard" | The flow has conditional exits at four points; a linear chain would hide them in `if`s, which is what the direct path already is. |

## 4. How it works

`settings.orchestration.backend` is read at call time by `factory.pipeline_class()` (17c), so an environment override (`ORCHESTRATION_BACKEND=langgraph`) or a test patch takes effect without a rebuild.

## 5. What the numbers said

Nothing measured in this unit; the suite stayed at 830 passed after the change (`uv run pytest -q`). The first measurements are 17b (tests) and 17d (equivalence).

## 6. What surprised me / what to watch for

The plan's description of the retry loop did not match the code. This is worth saying in the interview: the claim "the retry loop maps naturally" is true of the *graph*, and untrue of the *system*, which deliberately has no such loop.

## 7. Where this fits and what's next

17b builds the graph that reads these settings. The retry edge stays off through every comparison in 17d–17e.
