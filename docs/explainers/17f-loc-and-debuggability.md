# 17f — Lines of Code and Debuggability

**Phase:** 17 · **Date:** 2026-10-25 · **Files:** `experiments/17e_loc.py`, `experiments/17f_debuggability.py`, `docs/results/orchestration-loc.json`, `docs/results/orchestration-graph.mmd`
**Topics from the course:** Agentic RAG (sequential, conditional, iterative, parallel workflows)

## 1. What I built

Two small measurements of the non-latency costs: code lines (docstrings, comments and blanks excluded, counted by `ast`/`tokenize`) and what a developer sees when a stage fails (same injected fault in both pipelines).

## 2. Why this approach

"Lines of code" is easy to flatter either way. Counting only code tokens removes commenting style from the comparison; counting only what *decides the flow* removes the stage methods both paths share. For debuggability, the one thing measurable without a debugger session is how many foreign frames sit between a fault and the project's own code.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Raw `wc -l` | Zero effort | The graph module's docstring is long on purpose; it would charge the graph for explaining itself. |
| Count the shared stage methods too | "Total system size" | Identical in both, so it adds a constant to both and hides the difference. |
| Subjective "readability" scoring | What readers care about | Not a number I can defend; the qualitative judgement is in the comparison doc and labeled as opinion. |

## 4. How it works

The direct orchestration is `_orchestrate`, `_serve_cached` and `_answer` in `pipeline.py`. The graph's is its whole module. A forced `Boom("qdrant timed out")` in the retriever is run through `answer()` on each class and the traceback frames are split into project files and everything else.

## 5. What the numbers said

| | Direct | Graph |
| --- | --- | --- |
| Code lines deciding the flow | **61** (`_answer` 38, `_serve_cached` 15, `_orchestrate` 8) | **184** module / 131 in functions (`_build_graph` 28, `__init__` 13, `_n_route` 13, ...) |
| of which the retry loop | 0 | 8 (`_n_reformulate`) + 1 edge |
| Exception type and message preserved | yes | yes (`Boom: qdrant timed out`) |
| Traceback frames (project / foreign) | 7 / 0 | 13 / 6 (all `langgraph`) |
| Topology as data | no | yes: `graph.get_graph().draw_mermaid()` printed the 10-node, 4-conditional-edge diagram, saved as `docs/results/orchestration-graph.mmd` |

About **3x the code** for the same behavior, and 6 more frames to read through.

## 6. What surprised me / what to watch for

- The 3x understates the effect in the *favourable* direction for the graph, because the nodes are thin wrappers over stage methods that already existed. A port that reimplemented each stage as a node would be larger. This is the price of the "shared stage methods" design in 17b, paid in the graph's favour.
- The direct `_answer` reads top to bottom in one screen; the graph's equivalent logic is spread across `_build_graph` (what connects), four edge functions (when) and ten nodes (what runs). That is the readability cost and the readability benefit at once: the topology is one place, the sequence is not.
- Not measured: the pull of additional transitive dependencies, and any checkpointing, interrupt or parallel-branch feature, none of which this system uses.

## 7. Where this fits and what's next

Feeds the verdict in `docs/results/framework-comparison.md`.
