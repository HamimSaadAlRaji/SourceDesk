# 00 — The Baseline That Fails

**Phase:** 0 · **Date:** 2026-07-29 · **Files:** `src/aegis/baseline.py`, `src/aegis/config.py`, `src/aegis/types.py`
**Topics from the course:** What RAG is & why LLMs need it; The RAG loop & its advantages; Why not put everything in the prompt

## 1. What I built

A deliberately bad RAG pipeline: 85 FastAPI tutorial/advanced doc pages, chunked into
non-overlapping 500-character windows, embedded with `bge-small-en-v1.5` (fastembed),
indexed in Qdrant, retrieved dense-only top-3, stuffed into a prompt with no citation
requirement and no way to abstain. It answers five FastAPI questions I know the answers
to. In, a question; out, a fluent answer with no indication of whether it's grounded.
This is the architecture most RAG tutorials ship as the finished product — it's the
starting point specifically because it's wrong in instructive ways.

## 2. Why this approach

The goal of Phase 0 isn't a good system, it's a _measured bad one_, so every later# 00 — The Baseline That Fails

**Phase:** 0 · **Date:** 2026-07-29 · **Files:** `src/aegis/baseline.py`, `src/aegis/config.py`, `src/aegis/types.py`
**Topics from the course:** What RAG is & why LLMs need it; The RAG loop & its advantages; Why not put everything in the prompt

## 1. What I built

A deliberately bad RAG pipeline: 85 FastAPI tutorial/advanced doc pages, chunked into
non-overlapping 500-character windows, embedded with `bge-small-en-v1.5` (fastembed),
indexed in Qdrant, retrieved dense-only top-3, stuffed into a prompt with no citation
requirement and no way to abstain. It answers five FastAPI questions I know the answers
to. In, a question; out, a fluent answer with no indication of whether it's grounded.
This is the architecture most RAG tutorials ship as the finished product — it's the
starting point specifically because it's wrong in instructive ways.

## 2. Why this approach

The goal of Phase 0 isn't a good system, it's a _measured bad one_, so every later
phase has a concrete failure to point at and a number to beat. Building the naive
version first, on purpose, and running it against real questions rather than reasoning
abstractly about where RAG can go wrong, surfaced two distinct and unrelated failure
mechanisms in the very first run — not hypothetical ones.

## 3. Alternatives rejected

| Alternative                                                     | The case for it                                                            | Why rejected here                                                                                                                                                                                                                                                                                                                                                                                  |
| --------------------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Whole-document embedding (no chunking)                          | Simpler; avoids the "which chunk boundary" problem entirely                | Average doc in the subset is 5,720 chars (486,214 chars / 85 docs). A single embedding vector for a doc that long dilutes past the point of being useful for a specific question — this is the well-known long-document embedding degradation, not a hypothesis.                                                                                                                                   |
| BM25-only (keyword) as the naive baseline instead of dense-only | Also a legitimate "naive" architecture; cheaper, no embedding model needed | Dense-only is the pattern actually shown in most introductory RAG tutorials and is the one that fails in the specific way this project's abstention system (Phase 10) needs to catch: _semantically plausible but wrong_ retrieval, not _zero-overlap_ retrieval. BM25's failure mode (no keyword match → empty results) is easier to detect and less relevant to the calibrated-abstention story. |
| Hosted embeddings API (OpenAI/Cohere) for the baseline          | One less local dependency to install                                       | Adds a second paid, opaque API for a component that gets thrown away at the end of this phase. `bge-small` via fastembed (ONNX, no torch/MPS setup) indexed 1,015 chunks in well under a minute locally, so there's no latency or cost reason to reach for a hosted API this early. The real encoder bake-off (bge-large vs BGE-M3) is Phase 3.                                                    |

## 4. How it works

Naive fixed-size chunking — no sentence, heading, or code-block awareness:

```python
def fixed_chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size) if text[i : i + size].strip()]
```

Retrieval is dense-only, top-3, and the prompt has no instruction to check groundedness
or decline to answer:

```python
NAIVE_SYSTEM_PROMPT = """You are a helpful assistant answering questions about FastAPI. \
Use the context below to answer the user's question directly and confidently."""
```

That system prompt is the whole guardrail budget. There is no citation requirement, no
confidence score, no "say you don't know if the context doesn't cover it." Whatever the
model produces goes straight back to the user.

## 5. What the numbers said

**Corpus size vs. the "why not just use a long-context model" question.** The 85-doc
subset used here is 486,214 chars (~121,553 tokens at ~4 chars/token). The _full_
English FastAPI docs tree (155 files, before Phase 1 adds ~2,000–3,000 GitHub issues)
is 1,453,680 chars — **~363,420 tokens**. `llama-3.3-70b-versatile` on Groq has a
128,000-token context window ([Groq pricing page](https://groq.com/pricing), fetched
2026-07-29). The full docs corpus alone is **2.84× over that window** — stuffing it
into a single prompt isn't expensive, it's impossible, before a single issue is added.

Even the reduced 85-doc subset, which does technically fit at 121,553 tokens, would
cost real money on every query: at Groq's $0.59/M input tokens, stuffing the whole
subset costs **$0.0717/query** in input tokens alone. The actual retrieval-based
baseline sent an average of ~340 tokens of context per query (measured from the saved
transcript, `docs/results/00-baseline-transcript.json`), costing **~$0.0002/query** —
retrieval is roughly **300× cheaper** than long-context stuffing at this corpus size,
and the gap only widens once issues are added in Phase 1.

**The hallucinations.** Five questions, two runs (temperature=0.0, but see §6). Full
transcripts with retrieved chunks: `docs/results/00-baseline-transcript.json`.

_Q: "How do you define a path parameter with a regex constraint in FastAPI?"_
Retrieved context (top-3, scores 0.796/0.792/0.772) was about boolean query parameter
parsing and path-containing-a-path — **zero overlap** with regex/pattern validation.
The model answered anyway:

> To define a path parameter with a regex constraint... `item_id: str = re.path("{item_id}", r"\d+")`

`re.path` does not exist in Python's `re` module — fully fabricated. A second run
produced `Path(..., regex="^\d{1,5}$")` and claimed a non-matching value "will return a
404 error." Both are wrong: current FastAPI uses `pattern=`, not the deprecated
`regex=` (confirmed in `docs_src/query_params_str_validations/tutorial004_an_py310.py`,
which the naive ingestion never reached — see §6), and validation failures return
**422 Unprocessable Entity**, confirmed via the `422`/`"Validation Error"` schema in
`docs/en/docs/advanced/additional-responses.md`, not 404.

_Q: "What decorator should I use today to run code when the app shuts down?"_
Retrieval this time found the right file (`advanced/events.md`, score 0.745) — but the
top chunk starts mid-sentence: `"l events, not both.\n\nYou can probably skip this
part...\n\nThere's an alternative way..."`. The model answered:

> you should use the `@app.on_event("shutdown")` decorator... for most use cases `@app.on_event("shutdown")` is the way to go.

The actual doc heading for that section, one sentence before the retrieved chunk
starts, is `## Alternative Events (deprecated) { #alternative-events-deprecated }`. The
current recommended mechanism — `lifespan`, covered earlier in the same file — never
appears in the retrieved chunks at all. A second run fabricated a non-existent
`@app.shutdown_event` decorator alongside the deprecated one.

_Q: "What is the default request timeout in FastAPI?"_ One run hedged correctly and
deferred to the ASGI server. A second run recommended the `slowapi` library — a rate
limiter, not a timeout mechanism — with a broken import (`_rate_limit_exceeded`; the
real name is `_rate_limit_exceeded_handler`), confidently conflating two unrelated
concepts.

Two questions ("does it have rate limiting", "max path operations") got honest,
correctly-hedged "no" answers both runs — the baseline isn't uniformly bad, which is
itself the point (§6).

## 6. What surprised me / what to watch for

**Two unrelated failure mechanisms showed up in the same five questions**, not one:

1. **Missing content, not just bad chunking.** FastAPI's docs pull code examples in via
   `{* ../../docs_src/... *}` include directives — the markdown files never contain the
   actual code, just a reference to a separate `.py` file. The naive ingestion (raw
   `.read_text()`, no include-resolution) never followed these, so the correct
   `pattern=` usage for path/query regex constraints **was never in the index at all**.
   The model didn't retrieve the wrong chunk — there was no right chunk to retrieve.
   This is exactly the risk plan.md flags for Phase 1 ("FastAPI uses MkDocs Material
   syntax — handle its `///` tab blocks and `{* ... *}` includes or you'll silently
   mangle content"), and now there's a concrete, reproduced instance of it before
   Phase 1 has even started.

2. **Retrieval can succeed and chunking can still throw away the one word that
   mattered.** For the shutdown-decorator question, retrieval found the correct file.
   Fixed-size chunking split the heading `## Alternative Events (deprecated)` from its
   own body text — the retrieved chunk starts one sentence _after_ the heading, so the
   single word that would have told the model (and the user) "this is outdated" never
   made it into context. This is a sharper, more specific case than "chunking loses
   context" as a generality — it's losing a specific disambiguating token at a specific
   boundary, which is precisely what `heading_path` in `Chunk` (already in `types.py`)
   and Phase 2's heading-aware chunker are for.

**Temperature=0.0 was not fully deterministic.** Two runs against the same index
produced different fabrications for the same questions (`re.path()` vs. a `slowapi`
recommendation for different runs of the same prompt). This is consistent with known
non-associativity in batched inference on providers like Groq — floating-point
summation order isn't fixed across batches, so temp=0 narrows but doesn't eliminate
run-to-run variance. Worth flagging now: the eval harness (Phase 6) and judge
validation (Phase 11) need to either accept this variance explicitly or pin
generation-side reproducibility some other way (e.g. caching, or accepting it as a
property of the system under test rather than trying to eliminate it).

**Not every question triggered a hallucination.** The two questions with genuinely
negative answers ("no, X doesn't exist in FastAPI") were answered correctly and
honestly both runs. The failure is conditional on retrieval quality, not a blanket
property of "the LLM lies" — which is the whole reason a _confidence/abstention_
mechanism (Phase 10) is the right fix, rather than e.g. a stricter prompt. The model
already knows how to say "I don't know" when the gap is obvious; the dangerous case is
when it isn't obvious to the model that its context is insufficient or missing.

## 7. Where this fits and what's next

Both mechanisms found here trace directly to specific later phases rather than being
generic RAG folklore:

- Mechanism 1 (missing content from unresolved includes) → **Phase 1**, which builds
  structure-preserving markdown parsing that explicitly has to handle `{* ... *}`
  includes and `///` admonition blocks.
- Mechanism 2 (heading severed from body at a chunk boundary) → **Phase 2**, the
  chunking laboratory, specifically the heading-aware strategy that keeps
  `heading_path` attached to every chunk.
- Both mechanisms together → **Phase 9/10**, grounded generation with citations and a
  calibrated confidence/abstention threshold, so that when retrieval is empty,
  irrelevant, or missing the disambiguating context, the system declines instead of
  fabricating fluently. That's the Air Canada failure mode this whole project is built
  around, and this phase produced two independent, reproduced instances of it before
  any of the real system exists.
- The token/cost math above is the first hard number that rules out "just use a bigger
  context window" as a fix — it's revisited with real numbers again once Phase 1 adds
  the issue corpus and the true corpus size is known.

phase has a concrete failure to point at and a number to beat. Building the naive
version first, on purpose, and running it against real questions rather than reasoning
abstractly about where RAG can go wrong, surfaced two distinct and unrelated failure
mechanisms in the very first run — not hypothetical ones.

## 3. Alternatives rejected

| Alternative                                                     | The case for it                                                            | Why rejected here                                                                                                                                                                                                                                                                                                                                                                                  |
| --------------------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Whole-document embedding (no chunking)                          | Simpler; avoids the "which chunk boundary" problem entirely                | Average doc in the subset is 5,720 chars (486,214 chars / 85 docs). A single embedding vector for a doc that long dilutes past the point of being useful for a specific question — this is the well-known long-document embedding degradation, not a hypothesis.                                                                                                                                   |
| BM25-only (keyword) as the naive baseline instead of dense-only | Also a legitimate "naive" architecture; cheaper, no embedding model needed | Dense-only is the pattern actually shown in most introductory RAG tutorials and is the one that fails in the specific way this project's abstention system (Phase 10) needs to catch: _semantically plausible but wrong_ retrieval, not _zero-overlap_ retrieval. BM25's failure mode (no keyword match → empty results) is easier to detect and less relevant to the calibrated-abstention story. |
| Hosted embeddings API (OpenAI/Cohere) for the baseline          | One less local dependency to install                                       | Adds a second paid, opaque API for a component that gets thrown away at the end of this phase. `bge-small` via fastembed (ONNX, no torch/MPS setup) indexed 1,015 chunks in well under a minute locally, so there's no latency or cost reason to reach for a hosted API this early. The real encoder bake-off (bge-large vs BGE-M3) is Phase 3.                                                    |

## 4. How it works

Naive fixed-size chunking — no sentence, heading, or code-block awareness:

```python
def fixed_chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size) if text[i : i + size].strip()]
```

Retrieval is dense-only, top-3, and the prompt has no instruction to check groundedness
or decline to answer:

```python
NAIVE_SYSTEM_PROMPT = """You are a helpful assistant answering questions about FastAPI. \
Use the context below to answer the user's question directly and confidently."""
```

That system prompt is the whole guardrail budget. There is no citation requirement, no
confidence score, no "say you don't know if the context doesn't cover it." Whatever the
model produces goes straight back to the user.

## 5. What the numbers said

**Corpus size vs. the "why not just use a long-context model" question.** The 85-doc
subset used here is 486,214 chars (~121,553 tokens at ~4 chars/token). The _full_
English FastAPI docs tree (155 files, before Phase 1 adds ~2,000–3,000 GitHub issues)
is 1,453,680 chars — **~363,420 tokens**. `llama-3.3-70b-versatile` on Groq has a
128,000-token context window ([Groq pricing page](https://groq.com/pricing), fetched
2026-07-29). The full docs corpus alone is **2.84× over that window** — stuffing it
into a single prompt isn't expensive, it's impossible, before a single issue is added.

Even the reduced 85-doc subset, which does technically fit at 121,553 tokens, would
cost real money on every query: at Groq's $0.59/M input tokens, stuffing the whole
subset costs **$0.0717/query** in input tokens alone. The actual retrieval-based
baseline sent an average of ~340 tokens of context per query (measured from the saved
transcript, `docs/results/00-baseline-transcript.json`), costing **~$0.0002/query** —
retrieval is roughly **300× cheaper** than long-context stuffing at this corpus size,
and the gap only widens once issues are added in Phase 1.

**The hallucinations.** Five questions, two runs (temperature=0.0, but see §6). Full
transcripts with retrieved chunks: `docs/results/00-baseline-transcript.json`.

_Q: "How do you define a path parameter with a regex constraint in FastAPI?"_
Retrieved context (top-3, scores 0.796/0.792/0.772) was about boolean query parameter
parsing and path-containing-a-path — **zero overlap** with regex/pattern validation.
The model answered anyway:

> To define a path parameter with a regex constraint... `item_id: str = re.path("{item_id}", r"\d+")`

`re.path` does not exist in Python's `re` module — fully fabricated. A second run
produced `Path(..., regex="^\d{1,5}$")` and claimed a non-matching value "will return a
404 error." Both are wrong: current FastAPI uses `pattern=`, not the deprecated
`regex=` (confirmed in `docs_src/query_params_str_validations/tutorial004_an_py310.py`,
which the naive ingestion never reached — see §6), and validation failures return
**422 Unprocessable Entity**, confirmed via the `422`/`"Validation Error"` schema in
`docs/en/docs/advanced/additional-responses.md`, not 404.

_Q: "What decorator should I use today to run code when the app shuts down?"_
Retrieval this time found the right file (`advanced/events.md`, score 0.745) — but the
top chunk starts mid-sentence: `"l events, not both.\n\nYou can probably skip this
part...\n\nThere's an alternative way..."`. The model answered:

> you should use the `@app.on_event("shutdown")` decorator... for most use cases `@app.on_event("shutdown")` is the way to go.

The actual doc heading for that section, one sentence before the retrieved chunk
starts, is `## Alternative Events (deprecated) { #alternative-events-deprecated }`. The
current recommended mechanism — `lifespan`, covered earlier in the same file — never
appears in the retrieved chunks at all. A second run fabricated a non-existent
`@app.shutdown_event` decorator alongside the deprecated one.

_Q: "What is the default request timeout in FastAPI?"_ One run hedged correctly and
deferred to the ASGI server. A second run recommended the `slowapi` library — a rate
limiter, not a timeout mechanism — with a broken import (`_rate_limit_exceeded`; the
real name is `_rate_limit_exceeded_handler`), confidently conflating two unrelated
concepts.

Two questions ("does it have rate limiting", "max path operations") got honest,
correctly-hedged "no" answers both runs — the baseline isn't uniformly bad, which is
itself the point (§6).

## 6. What surprised me / what to watch for

**Two unrelated failure mechanisms showed up in the same five questions**, not one:

1. **Missing content, not just bad chunking.** FastAPI's docs pull code examples in via
   `{* ../../docs_src/... *}` include directives — the markdown files never contain the
   actual code, just a reference to a separate `.py` file. The naive ingestion (raw
   `.read_text()`, no include-resolution) never followed these, so the correct
   `pattern=` usage for path/query regex constraints **was never in the index at all**.
   The model didn't retrieve the wrong chunk — there was no right chunk to retrieve.
   This is exactly the risk plan.md flags for Phase 1 ("FastAPI uses MkDocs Material
   syntax — handle its `///` tab blocks and `{* ... *}` includes or you'll silently
   mangle content"), and now there's a concrete, reproduced instance of it before
   Phase 1 has even started.

2. **Retrieval can succeed and chunking can still throw away the one word that
   mattered.** For the shutdown-decorator question, retrieval found the correct file.
   Fixed-size chunking split the heading `## Alternative Events (deprecated)` from its
   own body text — the retrieved chunk starts one sentence _after_ the heading, so the
   single word that would have told the model (and the user) "this is outdated" never
   made it into context. This is a sharper, more specific case than "chunking loses
   context" as a generality — it's losing a specific disambiguating token at a specific
   boundary, which is precisely what `heading_path` in `Chunk` (already in `types.py`)
   and Phase 2's heading-aware chunker are for.

**Temperature=0.0 was not fully deterministic.** Two runs against the same index
produced different fabrications for the same questions (`re.path()` vs. a `slowapi`
recommendation for different runs of the same prompt). This is consistent with known
non-associativity in batched inference on providers like Groq — floating-point
summation order isn't fixed across batches, so temp=0 narrows but doesn't eliminate
run-to-run variance. Worth flagging now: the eval harness (Phase 6) and judge
validation (Phase 11) need to either accept this variance explicitly or pin
generation-side reproducibility some other way (e.g. caching, or accepting it as a
property of the system under test rather than trying to eliminate it).

**Not every question triggered a hallucination.** The two questions with genuinely
negative answers ("no, X doesn't exist in FastAPI") were answered correctly and
honestly both runs. The failure is conditional on retrieval quality, not a blanket
property of "the LLM lies" — which is the whole reason a _confidence/abstention_
mechanism (Phase 10) is the right fix, rather than e.g. a stricter prompt. The model
already knows how to say "I don't know" when the gap is obvious; the dangerous case is
when it isn't obvious to the model that its context is insufficient or missing.

## 7. Where this fits and what's next

Both mechanisms found here trace directly to specific later phases rather than being
generic RAG folklore:

- Mechanism 1 (missing content from unresolved includes) → **Phase 1**, which builds
  structure-preserving markdown parsing that explicitly has to handle `{* ... *}`
  includes and `///` admonition blocks.
- Mechanism 2 (heading severed from body at a chunk boundary) → **Phase 2**, the
  chunking laboratory, specifically the heading-aware strategy that keeps
  `heading_path` attached to every chunk.
- Both mechanisms together → **Phase 9/10**, grounded generation with citations and a
  calibrated confidence/abstention threshold, so that when retrieval is empty,
  irrelevant, or missing the disambiguating context, the system declines instead of
  fabricating fluently. That's the Air Canada failure mode this whole project is built
  around, and this phase produced two independent, reproduced instances of it before
  any of the real system exists.
- The token/cost math above is the first hard number that rules out "just use a bigger
  context window" as a fix — it's revisited with real numbers again once Phase 1 adds
  the issue corpus and the true corpus size is known.
