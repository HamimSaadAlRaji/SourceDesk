# CLAUDE.md — Working Agreement for Aegis

This project is optimized for **getting the user hired as an AI/RAG engineer**, not for teaching them RAG. They have completed all five modules in `RAG_Topics_Learned.md` and understand the concepts. Do not re-teach them.

Read this fully before any work. It governs every session.

---

## 1. The Prime Directive

**Every significant claim this project makes must be backed by a number the user produced, and every significant decision must have a written record of what was rejected and why.**

That is what separates a hireable portfolio project from a tutorial with a nice README. An interviewer's questions are not "what is an embedding" — they are "why that chunk size", "how do you know hybrid helped", "what happens when it's wrong". This repository must answer those.

Practical consequences:

- A component without a measurement is not finished.
- A decision without a rejected alternative is not a decision, it's a default.
- "This is standard practice" is never a justification. Either measure it or explain the mechanism.
- Honest negative results are more valuable than tidy positive ones. A technique that didn't pay for itself, reported clearly, demonstrates judgment. Hiding it demonstrates nothing.

---

**Never Commit anything to git. You can not write in git. You can read it but not write.**

## 2. Build vs Import

`plan.md` §1 is the authoritative split. The governing test:

> **Does a mature library already do this well, and would writing it myself prove anything?**

Import the solved problems: BM25 (`bm25s`), embeddings, HNSW, the reranker, retrieval metrics (`ranx`), RAGAS, Langfuse, FastAPI.

Build what's genuinely the project's own: the custom analyzer, the heading-aware chunker, RRF fusion, the confidence scorer, the citation validator, the routing policy, the eval harness, the semantic cache.

**Adding a dependency requires justification in the design record.** So does hand-rolling something a library does well — that direction is now the one that needs defending, not the reverse. Reimplementing commodity infrastructure costs days and reads as not knowing the ecosystem.

---

## 3. The Design Record Protocol — non-negotiable

**After every implementation unit, write a design record.** Not after every phase. After every unit a person could reasonably ask "why did you do it that way?" about.

Location: `docs/explainers/NN-topic-name.md`, numbered to match phases. Sub-units get letters: `05a-`, `05b-`.

These are **engineering design records, not tutorials.** The difference matters:

| Not this                                     | This                                                                                                              |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| "HNSW is a graph-based ANN index that…"      | "`ef_search=128` because recall plateaus at 0.97 above it while p95 keeps climbing — here's the curve"            |
| "Cross-encoders score query and doc jointly" | "Reranking top-50 gains 0.04 NDCG@5 over top-20 and costs 210 ms, which doesn't fit the budget, so we rerank 20"  |
| "RRF combines multiple rankings"             | "RRF beat linear fusion by 3.1 Recall@10 on this corpus because BM25 and dense scores are on incomparable scales" |

Concept explanation is fine where it's _load-bearing for the decision_. It is not the point.

### Required structure

```markdown
# NN — <Title>

**Phase:** <n> · **Date:** YYYY-MM-DD · **Files:** `path/one.py`, `path/two.py`
**Topics from the course:** <exact names from RAG_Topics_Learned.md>

## 1. What I built

Two or three sentences. What goes in, what comes out, what it's for.

## 2. Why this approach

The actual argument for this design over another. Tie it to a constraint that
exists in this project: this corpus, this latency budget, this measurement,
this observed failure. Not general best practice.

## 3. Alternatives rejected

At least two, in a table. "Worse" is not a reason. "Adds 200 ms to a stage
already at 340 ms of a 3 s budget" is a reason.

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ----------------- |

## 4. How it works

The mechanism, with the real annotated code for the part carrying the idea.
Formulas written out with each term explained. Data structures shown with a
concrete example.

## 5. What the numbers said

The measurement. Before/after, or the sweep, or the comparison. If nothing is
measured yet, say "not yet measured — Phase N" and link forward. Never invent
a number, never use an illustrative one.

## 6. What surprised me / what to watch for

Non-obvious behavior, sharp edges, things that will bite later. Where a result
contradicted expectation. Often the most valuable section — write it even when
it feels like there's nothing to say.

## 7. Where this fits and what's next

What this unlocks, what depends on it, what comes next and why that ordering.
```

### Rules

- **Real numbers from this project only.** No illustrative values, ever.
- **Name the course topic when one is exercised**, and connect it to the decision — that's the bridge from "studied HNSW" to "knows what `ef_search` costs."
- **Report what didn't work.** Techniques that underperformed, measurements that contradicted expectations, approaches abandoned mid-implementation.
- Length follows substance. Padding is worse than brevity.

---

## 4. Session Protocol

### Starting

1. Read `progress.md` — it is the state of the world.
2. Read the current phase's exit criteria in `plan.md`.
3. State plainly: current phase, what's done, what's next, carried-over blockers.

### During

- Work in units. After each: explain in chat → write the design record → update `progress.md`.
- **Never batch multiple components before explaining.**
- When a decision has real tradeoffs (not a routine judgment call), present options with a recommendation and ask. Don't silently pick and mention it later.
- When a measurement contradicts an expectation, stop and investigate before proceeding. That contradiction is the most valuable thing available and burying it wastes it.

### Ending

1. Update `progress.md` — phase status, decisions, numbers, open questions.
2. State what the next session picks up.
3. Leave `make test` passing.

---

## 5. Verification Discipline

**Never claim something works without running it and showing the output.**

- "Tests pass" requires pasted `pytest` output.
- "Recall improved to 0.84" requires the command and its output.
- "The endpoint streams" requires an actual request against it.

If it wasn't run: "I've written this but haven't run it yet — running now."

If it fails, report the real error. Don't paper over it, don't describe partial results as complete, don't move on with a known failure behind you.

**Above all: never invent a metric value.** If the eval hasn't run, the number does not exist.

---

## 6. Code Standards

**Style.** Type hints everywhere. Docstrings explain _why_, not _what_. Small focused modules — past ~300 lines a file is probably doing two things.

**Configuration.** Every tunable in `config.py`. No magic numbers in logic. This isn't fastidiousness: ablation studies are impossible when parameters are scattered, and this project is mostly ablation studies.

**Testing.** TDD the components where a silent bug corrupts every downstream measurement: the analyzer, the chunkers, RRF, the confidence scorer, the citation validator. Write the failing test first. I/O-bound components get integration tests instead.

**Reproducibility.** Seeds fixed. Eval runs keyed by config hash. Golden set versioned and frozen. A result that can't be reproduced isn't a result.

---

## 7. Progress Tracking

`progress.md` is updated **at the end of every phase without exception**, and mid-phase when something significant lands.

It must always answer at a glance: what phase, what's done, what's measured (with numbers), what was decided (with alternatives), what's blocked, what's next.

It is the recovery mechanism. If the user returns after two weeks or a session ends abruptly, it must be sufficient to reload full context. Treat it that way.

---

## 8. Communication Style

- **Direct.** Answer first, reasoning second.
- **No flattery.** Don't open with "great question."
- **Disagree when warranted.** If a proposal will produce a worse system or a weaker portfolio, say so with the reason, then follow their call if they reaffirm it. Agreeing with a bad idea to be pleasant is a failure of this agreement.
- **Flag uncertainty explicitly.** "I believe X but haven't verified it" is useful. Confident wrongness is not.
- **Minimal self-correction.** Fix it, note it in one line if it changes their understanding, move on.

---

## 9. Scope Control

`plan.md` is the contract.

- Ideas not in the plan go to `docs/future-work.md`. They don't get built.
- Scope changes are proposed explicitly with the cost in days, and require agreement.
- `plan.md` §7 defines what gets cut under time pressure and in what order. Follow it rather than improvising.
- **Phases 4, 6, 10, 11 (judge validation), 16, and 19 are never cut.** They are the evidentiary spine. Without them this is a demo with unbacked claims — which is what every other candidate has.

---

## 10. Project Context

**Corpus:** FastAPI documentation + closed GitHub issues.
**Hardware:** Apple M1 Pro, 16 GB. MPS for local inference; batch conservatively during embedding runs.
**Python:** 3.12 via `uv` — not the system 3.13, parts of the ML stack lag it.
**Vector store:** Qdrant in Docker, named dense + sparse vectors.
**Docker:** installed.

**Compute budget:** If a run is projected to take more than 10 minutes of this machine's CPU time, it does not run locally — it moves to Kaggle (free GPU notebooks) instead. Benchmark the real per-unit cost on real project data (not short placeholder text) before projecting total runtime, so the 10-minute call is made on a real number, not a guess. Kaggle is not connected to this session: shipping a job there means preparing a self-contained script plus any data export it needs, and then **explicitly telling the user they must paste the script into a Kaggle notebook and run it themselves** — this is a manual handoff, never something to imply happens automatically. Say so plainly every time a job is handed off this way.

**The recursion is intentional:** this system is built with FastAPI and answers questions about FastAPI. When retrieval is wrong the user can often tell immediately from their own knowledge — a genuine evaluation advantage, and part of why this corpus was chosen.

**The North Star:** the Air Canada failure mode. Every design decision should trace back to preventing a confident, uncited, unescalated wrong answer. When a tradeoff is unclear, ask: _which choice makes it harder for this system to confidently mislead a user?_
