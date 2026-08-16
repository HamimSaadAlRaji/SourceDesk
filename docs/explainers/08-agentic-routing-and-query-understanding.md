# 08 — Agentic Routing & Query Understanding

**Phase:** 8 · **Date:** 2026-08-16 · **Files:** `src/aegis/config.py`, `src/aegis/router/classifier.py`, `src/aegis/router/policy.py`, `src/aegis/query/rewrite.py`, `src/aegis/query/expand.py`, `src/aegis/query/hyde.py`, `src/aegis/query/iterative.py`, `experiments/08a_router_eval.py`, `experiments/08b_generate_rewrite_eval_set.py`, `experiments/08c_query_understanding_ablation.py`, `docs/results/routing-confusion-matrix.md`, `docs/results/query-understanding-ablation.md`
**Topics from the course:** Query parsing (query rewriting, NER, HyDE); Advanced prompting & reasoning (in-context learning, Chain-of-Thought, reasoning models, context pruning); Agentic RAG (sequential, conditional, iterative, parallel workflows)

## 1. What I built

Two things. First, a router (`RouterClassifier` + `router/policy.py`): a small LLM classifies every query into `docs_howto`, `known_issue`, or `billing_account_legal` before any retrieval happens, and a hard-coded (not prompted) policy function makes `billing_account_legal` structurally unable to reach a generation call. Measured against 58 labeled golden queries: overall accuracy 68.97%, but the number that actually matters — recall on `billing_account_legal` — is 1.0000.

Second, four query-understanding techniques (rewriting, multi-query expansion, HyDE, iterative retrieval), each built as a toggleable wrapper around Phase 6's winning dense retriever, each A/B'd against it. One measurably helps (rewriting). Three measurably hurt, for three different, independently-investigated reasons, and are off by default.

## 2. Why this approach

**The router's safety property is a code path, not a prompt, on purpose** — this is plan.md's own framing and CLAUDE.md's North Star made concrete: "which choice makes it harder for this system to confidently mislead a user?" `RouterClassifier.classify()` can fail three ways — malformed JSON, an out-of-vocabulary label, a raw Groq `APIError` (all three happened for real during development, see §6) — and all three are wired to the *same* fallback: `Route.BILLING_ACCOUNT_LEGAL`. Not `docs_howto`. A classifier failure is far more likely to be noise than an actual billing query, but the asymmetry in cost (an unnecessary human handoff vs. a possible Air Canada-shaped miss) means the fail-safe direction isn't a coin flip. `router/policy.py::handle()` then enforces this structurally: it is the *only* function that turns a route into either `generate_fn` or `escalate_fn`, and it cannot call both. `tests/test_router_policy.py::test_generate_fn_that_raises_on_call_proves_billing_never_invokes_it` proves this by passing a `generate_fn` that raises `AssertionError` if it's ever invoked — the test doesn't just check the outcome, it makes the forbidden path detonate if reachable.

**Every query-understanding technique is a `.search()`-compatible wrapper, composed, not built into `HybridRetriever`.** Same shape as Phase 7's `RerankedRetriever`: `ExpandedRetriever`, `HydeRetriever`, `IterativeRetriever` all take a base retriever and an `enabled` flag, and degrade to exactly the base retriever's behavior when off. This is what makes the A/B honest — the "off" condition isn't a separate code path that could silently drift from the "on" condition, it's the identical base retriever untouched.

**Router ground truth reuses `strata.py`'s existing classification rather than a fresh labeling pass** (decided with the user, 2026-08-12): `how_to` dev positives → `docs_howto`, `error_debugging` dev positives → `known_issue`, `billing_legal` negatives → `billing_account_legal`. This isn't a repurposed signal — `strata.py`'s how_to/error_debugging cues (`"how do i"`, `"is it possible to"` vs. `"error"`, `"crash"`, `"not working"`) are *already* a rule for how a query is phrased, which is exactly the router's own distinction. Reusing it is reusing the same signal under a new name, not stretching an unrelated one.

**Query rewriting's eval set is LLM-generated, not hand-written** (decided with the user, 2026-08-12, reversing my own initial recommendation) — this corpus has no real multi-turn conversation data (FastAPI docs and closed GitHub issues are both single-turn artifacts), so there is no way around synthesizing it. The generated `(turn1, turn2)` pairs are anchored to a real golden query's real `ground_truth_chunk_ids`, so even though the *conversation* is synthetic, the *scoring* is not.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| ----------- | --------------- | ------------------ |
| Router ground truth: a doc_type-composition heuristic (label `docs_howto` if a query's ground-truth chunks are mostly `doc::`, `known_issue` if mostly `issue::`) | Uses data already in the golden set, no new labeling logic | Measured first: checked doc_type composition of ground truth across all 6 strata — every stratum sits at 46-58% doc-chunks, none separates cleanly (a side effect of Phase 4's broad auto-labeling, same root cause as §6 below). This axis has no discriminating signal to reuse. |
| Router ground truth: fresh LLM-based auto-labeling of all 6 strata | Covers the full query population, not just 2 of 6 strata | Same "using an LLM to grade an LLM" circularity the router classifier itself has — would weaken the eval's credibility rather than strengthen its coverage. User explicitly picked strata-reuse over this. |
| Fail-open on router parse failure (default to `docs_howto`) | Fewer unnecessary escalations, better deflection-rate optics | Directly the Air Canada failure mode CLAUDE.md's North Star exists to prevent. An unnecessary escalation costs a human ticket; a silently-mishandled billing/legal query is the one failure mode this whole project is built around avoiding. |
| Query rewriting eval: hand-write the synthetic conversations myself | Full traceability, no LLM-generation-quality risk (my own initial recommendation) | User picked LLM-generation instead (2026-08-12) — faster, and spot-checking 3 of 24 against their real source issue text (§5) confirmed the fidelity was there anyway, including one correctly-translated Chinese-language item. |
| Fix multi-query expansion's RRF-fusion weighting or paraphrase-quality filtering after finding the failure mechanism (§6) | Might turn a −49.4% regression into a real win | Out of Phase 8's scope as specified — plan.md asks to A/B and report honestly, not to iterate each technique to convergence. The measured "off" decision is already actionable; a fix attempt is future-work territory, not a blocker here. |
| Iterative retrieval's confidence signal: wait for Phase 10's real confidence scorer before measuring anything | More accurate trigger, avoids measuring a signal known to be provisional | Phase 8 is explicitly sequenced before Phase 10 in plan.md; the point of this phase's measurement is whether the *retry pattern* helps at all with *any* cheap signal available now, not to pre-empt Phase 10. Documented as provisional in both `config.py` and the results doc so it's revisited, not forgotten. |

## 4. How it works

**The four agentic workflow types (course topic 27), each with a real component in this phase:**

- **Conditional** — the router. `router/policy.py::decide()` branches on the classified route; only one of two outcomes (`generate_fn` or `escalate_fn`) is ever reached per query, enforced in `handle()`.
- **Parallel** — multi-query expansion. `ExpandedRetriever` fires independent retrieval calls for the original query and up to 3 LLM-generated paraphrases, then combines all of them with RRF — several attempts made concurrently in principle (sequentially here, since there's no async plumbing yet, but the *pattern* is parallel: none of the retrieval calls depends on another's result).
- **Sequential** — HyDE and rewriting are linear chains where each step's output feeds the next: `QueryRewriter.rewrite(history, query) → dense_index.search(rewritten)`, or `HydeGenerator.generate(query) → dense_index.search(hypothetical)`. One dependent step after another, not a branch or a loop.
- **Iterative** — `IterativeRetriever`. Retrieve, check a stopping condition (confidence proxy), and conditionally repeat with a modified input — capped at `max_retries=1` per plan.md's explicit instruction, tested in `tests/test_iterative.py::test_retry_is_capped_even_if_still_low_confidence`.

(Course topic 17 also names NER as a query-parsing technique. Not built as a separate component here: this corpus's entity-extraction need is already covered by Phase 1's keyword-based `product_area` matching, and no Phase 8 technique had an unmet need a dedicated NER step would solve — same design-record-only treatment Phase 7 gave ColBERT, for the same reason: a technique whose lesson is already covered by something built for a different reason doesn't need its own implementation to prove the point.)

**Why HyDE works, mechanically (course topic 24's advanced-prompting/reasoning angle, applied):** dense retrieval's cosine similarity measures how close two pieces of text sit in embedding space, and a *question* and its *answer* are different genres of text even when topically identical. Two answers to the same question sit much closer together than a question sits to its own answer — so embedding a plausible-sounding hypothetical answer instead of the raw question searches from the answer's neighborhood, not the question's:

```python
# aegis/query/hyde.py — the whole idea in three lines
hypothetical = self.generator.generate(query)   # LLM invents a plausible answer
return self.base_retriever.search(hypothetical, top_k=top_k, **kwargs)  # embed & search THAT
```

No interface change was needed for this — `DenseIndex.search(query: str, ...)` already embeds whatever string it's given internally, so passing a hypothetical answer instead of a real query is transparent to every downstream consumer.

**Why the router's fail-safe direction is asymmetric, mechanically:** `RouterClassifier.classify()` parses the LLM's JSON response and maps failures — `json.JSONDecodeError`, `KeyError`, `ValueError` (out-of-vocabulary route string), `TypeError` — to one outcome:

```python
# aegis/router/classifier.py
except (json.JSONDecodeError, KeyError, ValueError, TypeError):
    return Route.BILLING_ACCOUNT_LEGAL
```

Combined with `_default_classify_fn` catching `groq.APIError` and returning `""` (which then hits the same except-branch), every failure mode — bad output, bad label, network/API failure — collapses to the one safe outcome, rather than three different failure paths each needing their own reasoning about safety.

## 5. What the numbers said

**Routing confusion matrix** (`docs/results/routing-confusion-matrix.md`, 58 labeled queries, `llama-3.1-8b-instant`, few-shot, JSON-mode):

| Route | Precision | Recall | TP | FN | FP |
|---|---|---|---|---|---|
| docs_howto | 0.6452 | 0.8000 | 20 | 5 | 11 |
| known_issue | 0.7500 | 0.4800 | 12 | 13 | 4 |
| billing_account_legal | 0.7273 | **1.0000** | 8 | 0 | 3 |

Overall accuracy 68.97% (40/58). **The number that matters is `billing_account_legal` recall: 1.0000** — every single billing/legal query in this eval set was routed correctly, with zero false negatives. The 3 false positives (2 `known_issue` queries and 1 `docs_howto` query misrouted *into* `billing_account_legal`) are the safe direction of error: an unnecessary escalation, not a missed one.

**Query-understanding ablation** (`docs/results/query-understanding-ablation.md`, 150-query dev set + 24-item synthetic set):

| Technique | NDCG@5 Δ | Recall@10 Δ | Latency (mean/p95) | Decision |
|---|---|---|---|---|
| Multi-query expansion | **−49.4%** | −51.8% | 5,526ms / 13,457ms | Off |
| HyDE | −23.6% | −28.3% | 1,597ms / 6,727ms | Off |
| Iterative retrieval | −13.4% | −15.8% | 574ms / 3,092ms | Off |
| Query rewriting | +52.7% (vs. context-free) | +34.2% | 1,542ms / 2,463ms | **On** |

Three of four regress; one — rewriting, measured on the problem it actually solves (context-free follow-ups) rather than against an already-strong single-turn baseline — clears its bar. `QuerySettings.rewrite_enabled = True`; the other three stay `False`.

## 6. What surprised me / what to watch for

**The router's known_issue recall (0.48) looked like a router accuracy problem and turned out to be partly a ground-truth artifact.** Investigated the false negatives directly rather than accepting the number: `golden::issue::12965` ("Will FastAPI support QUERY http method?") is labeled `known_issue` by `strata.py` purely because the word "error" appears once, deep in the issue body, describing a Swagger UI symptom of what is fundamentally a feature-support question — the router (correctly, in my reading) called it `docs_howto`. `strata.py`'s classification runs over title + body + *resolution comment* combined, using first-match-wins priority order over generic keyword cues; a query genuinely can carry two real signals at once, and the stratum a query lands in reflects "which cue matched first," not "this query is unambiguously that category." Reusing that as router ground truth (§2's decision) inherits this same fuzziness. This doesn't undermine the one number that actually matters here (`billing_account_legal` recall = 1.0000, unaffected by this ambiguity), but it means the 68.97% overall accuracy number is a floor, not a precise measurement of the router's real-world behavior — worth remembering before quoting it without context.

**Multi-query expansion's regression (−49.4% NDCG@5) is the worst measured anywhere in this project, and it demanded real investigation, not a shrug** (CLAUDE.md: a result this far outside expectation gets investigated, not just reported). Found two compounding, independently-verifiable mechanisms (full detail and a concrete example in `docs/results/query-understanding-ablation.md` §1): the small paraphrase model (`llama-3.1-8b-instant`, chosen for cost) sometimes ignores its own "do not answer the question" instruction and generates confident, wrong, invented answers instead of rephrasing the question — and even when paraphrases are reasonable, unweighted RRF fusion across 4 lists structurally lets 3 mutually-reinforcing-but-wrong lists outvote 1 accurate one, since Phase 6's `rrf_k=60` was tuned for exactly 2 sources of comparable reliability, never validated for N-way fusion where most sources can be wrong simultaneously. Neither mechanism is specific to this corpus — both would recur on any corpus, which is exactly why this is worth documenting mechanically rather than filing as "just didn't work here."

**HyDE and Phase 7's reranker failed for the same underlying reason, discovered independently, four phases apart.** HyDE's one clear regression case (`golden::issue::2342`) wasn't hallucinated garbage — it was a specific, accurate, FastAPI-literate hypothetical answer that simply committed to one narrow angle of a question whose golden-set ground truth spans a broader, "on-topic" standard (Phase 4's token-overlap auto-labeling, median 17 ground-truth chunks across 12 documents per query). That is the *exact* mechanism `docs/explainers/07-reranking-and-encoder-architectures.md` §6 found for the cross-encoder reranker. Two different techniques, built in different phases, independently discovered to be penalized by the same labeling-methodology property — strong enough of a pattern now (this is the second confirmation) that it should shape how much weight Phase 11's *absolute* numbers can bear, not just be treated as a one-off excuse each time it recurs.

**I repeated a mistake this project already has two decision-log entries about.** `experiments/08c_query_understanding_ablation.py`'s first run hung for 17+ minutes at 0% CPU with ~3GB swap in use — the identical MPS-under-memory-pressure signature Phase 3 and Phase 7 both hit and documented. I built the encoder without overriding `device`, defaulting to the configured `mps`, despite Phase 7's decision log describing this exact failure in this exact codebase four days earlier. Fixed the same way Phase 7 did (force `device="cpu"`), and the run completed normally at 58% CPU. The lesson isn't new — it's that a documented lesson doesn't automatically transfer to a new script unless the new script's author (me, this phase) actually re-reads the prior decision log before writing code that touches the same resource, rather than after it hangs.

**A small, un-investigated discrepancy is worth naming rather than smoothing over.** This phase's baseline (NDCG@5 0.7587, recall@10 0.4435) doesn't exactly match Phase 7's reported no-rerank baseline on the identical config (0.7550, 0.4324) — same query set, same collection. Not chased to ground (doesn't change any conclusion in this phase), but flagged in `query-understanding-ablation.md` rather than silently reconciled or ignored, per CLAUDE.md's rule that a contradiction is the most valuable thing available and shouldn't be buried.

## 7. Where this fits and what's next

The router is real and measured, but not yet wired into the actual request path — that's an integration step, not a Phase 8 deliverable (plan.md's Phase 8 build list stops at classify + policy + eval; end-to-end assembly happens incrementally as later phases add the pieces it would call). Query rewriting is the one technique carried forward into the live pipeline (`QuerySettings.rewrite_enabled = True`); expansion, HyDE, and iterative retrieval remain fully built, tested, and toggleable — reversible decisions, not deleted code, exactly like Phase 6's enrichment and Phase 7's reranking. Iterative retrieval's "off" result is the most explicitly provisional of the three: it used a cheap proxy confidence signal by necessity (Phase 10 doesn't exist yet), and should be re-measured once `guardrails/confidence.py`'s real scorer (reranker score + rank gap + inter-chunk agreement + score-distribution shape) exists — this phase's number is evidence about *this proxy*, not a closed question about the retry pattern itself. Phase 9 (grounded generation) is now the first phase that can actually call a real LLM to answer instead of just classify/rewrite/expand — everything upstream of it (router, retrieval, now with rewriting) exists specifically to hand it a well-formed query and the right chunks before it commits to an answer.
