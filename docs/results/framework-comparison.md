# Direct pipeline vs LangGraph — what the framework cost and what it bought

Phase 17. The request flow (cache → route → retrieve → confidence → generate-or-escalate, plus follow-up
rewriting) exists twice behind one interface, selected by `ORCHESTRATION_BACKEND=direct|langgraph`.
The direct version is the one I built first; the LangGraph version is a port of it, held to the direct one's
behavior by the project's own eval harness. Every number below was produced in this repository
(`docs/results/orchestration-*.json`, `framework-overhead-raw.json`; design records 17a-17f).

## Verdict

**For this system, keep the direct pipeline as the default. The port works and costs almost nothing at
runtime, but it buys nothing this system uses.** It is 3x the code and 6 more stack frames for a flow with
one process, four exits and no persistence. I would reach for LangGraph when the flow needs what it is built
for (below), and I would not reach for it here.

## The comparison

| | Direct | LangGraph | Source |
|---|---|---|---|
| Behavior on the 180 dev queries (real router, generator, entailment) | reference | **180 / 180 identical in every field**, cold pass and cache pass | 17d |
| Same, with `tau` shifted 0.1 (does the harness see a difference?) | | 59 / 180 differ, 19 answered → escalated | 17d control |
| Whole pre-existing test suite | 862 passed | 862 passed (`AEGIS_TEST_BACKEND=langgraph`) | 17b |
| Span tree under real Langfuse/OTel | 1 trace, 7 spans | identical | 17c |
| Code lines that decide the flow | 61 | 184 (131 in functions) | 17f |
| Framework cost per request, stubbed stages | 26-182 µs | 634-1,352 µs (**+0.6 to +1.2 ms**) | 17e |
| Live paired wall time, 40 queries | p50 3,577 / p95 11,799 ms | p50 3,406 / p95 10,718 ms; paired median **-22 ms, 95% CI [-162, +124]** | 17e |
| Traceback frames for a failing stage | 7 | 13 (6 are LangGraph) | 17f |
| Topology introspectable | no | yes, `draw_mermaid()` | 17f |
| New declared dependencies | 0 | 1: `langgraph>=1.2.2,<2` (the lockfile gained 2 lines; its checkpoint, prebuilt and sdk packages were already installed through the LangChain packages) | 17a |

## What state management bought, and what control it cost

**Bought.** The flow's shape is now data. The four exits and the one loop are named in one place
(`_build_graph`), the diagram is generated rather than drawn, and the retry loop on low confidence is
one back-edge plus an 8-line node, capped by a counter in state. Adding a branch means adding a node and an
edge, not threading a flag through a 38-line method.

**Cost.** The sequence is no longer readable top to bottom: the direct `_answer` fits one screen; the same
logic is spread over a graph builder, four routing functions and ten nodes. Everything that used to be a
local variable became a state field (`verdict`, `retrieval_query`, `retries`). Things that are not state
(the event sink, the timings dict) had to go through `config`, which is a second mechanism to learn. A
failure arrives through six extra frames, though the exception type and message survive intact.

**Not bought, because this system does not use it.** Checkpointing and resume, human-in-the-loop interrupts,
parallel branches, multi-agent handoffs, and LangGraph's streaming modes. The system streams through its own
event queue, and the escalation to a person is a returned result, not a paused graph.

## Three things the port taught me

1. **The plan's description did not match the code.** The plan says the low-confidence retry loop "maps
   naturally" to a graph. It does, but the direct pipeline has no such loop: iterative retrieval was measured
   at -13.4% NDCG@5 in Phase 8 and shipped off. I built the edge, config-gated and off, so the port stayed
   behavior-preserving. The edge is verified by unit tests only; on real data it needs reranker pairs that are
   not cached (the attempt was refused at 63 > 60 uncached pairs).
2. **The size of the port depends on what it reuses.** 184 lines is small because the nodes call stage
   methods that already existed, so tracing, timing, streaming, caching and the audit trail are the same code
   under both backends. A framework-native rewrite would be larger and would have two copies of each of those
   to keep in sync. The honest framing is "orchestration only", not "an application on LangGraph".
3. **The live latency run could not see the framework, and the isolated one could.** Provider variance is
   seconds; the framework is a millisecond. Only removing the network showed it.

## When I would, and would not, use it

Use LangGraph when the flow has durable state (resume after a crash or a human's decision), long-running or
interruptible steps, parallel fan-out, or several cooperating agents, or when a team already shares its
vocabulary and tooling. Do not use it for a single-process request/response flow whose branching is a handful
of `if`s that a new reader can follow in one screen, which is what this is.

## Limits of this comparison

- One implementation of each, by one person; the direct one came first, so it is the better-understood.
- Equivalence is proven under LLM replay (same model text to both). Two live runs would also differ by model
  noise alone; that noise was not measured.
- The live latency sample is 40 queries per path (35 on the generate path): enough to show no visible
  difference, not enough to measure a tail.
- The retry edge has not run on real data (above).
- Lines of code were counted on code tokens only, and only for the code that decides the flow.

## An unrelated finding this measurement produced

The fresh direct p95 on the current stack is **11.8 s** (n = 40) against a 3 s target. The last number on
file was 5.6 s from the qwen-era stack. On the generate path generation is 53% of mean time (p95 7.1 s) and
validation 9% (p95 2.7 s, because citation entailment is now an LLM call). This is the baseline the Phase 16
hand-off asked for; it is not a result of the port (graph p95 was 10.7 s in the same run) and it is unfixed.
Recorded in `docs/future-work.md`.
