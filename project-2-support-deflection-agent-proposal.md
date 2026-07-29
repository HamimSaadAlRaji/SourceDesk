# Business & Technical Proposal
## Aegis: An Agentic RAG Customer Support Deflection Platform

**Prepared as a portfolio project proposal — full problem statement, solution design, and technical architecture**

---

## 1. Executive Summary

Customer support is one of the largest controllable cost centers in any product company, and today it runs on a broken economic model: humans manually re-answer the same few hundred questions thousands of times a day, at a fully-loaded cost that industry benchmarks place at **$18–$35 per ticket for SaaS companies and $30–$60 per ticket for B2B/technical support** ([Lorikeet, 2026](https://www.lorikeetcx.ai/articles/customer-service-cost-per-ticket); [Supportbench, 2026](https://www.supportbench.com/calculate-real-cost-per-ticket-justify-headcount/)). Meanwhile, generic AI chatbots — the "fix" many companies have already tried — have made things *worse* in visible, expensive ways when they answer confidently without being grounded in real source-of-truth documentation, as the [Moffatt v. Air Canada](https://www.forbes.com/sites/marisagarcia/2024/02/19/what-air-canada-lost-in-remarkable-lying-ai-chatbot-case/) case demonstrates in painful, litigated detail (Section 2.2).

This document proposes **Aegis**, a production-grade Retrieval-Augmented Generation (RAG) system that answers customer support questions by retrieving from real product documentation and historical resolved tickets, cites every claim to a source, and — critically — knows when *not* to answer and hands off to a human instead. It is not a generic chatbot wrapper. It is an **agentic retrieval pipeline** with query routing, hybrid dense+lexical search, cross-encoder reranking, a calibrated escalation policy, and a continuously-running evaluation harness that proves the system's accuracy rather than asserting it.

The business case: independent industry data compiled by Fullview/Customer Experience Update puts the average cost of an AI-handled support interaction at roughly **$0.50 versus $6.00 for a human-handled interaction — a 12x cost differential** ([Unthread, 2026](https://unthread.io/blog/support-ticket-resolution-statistics/)), and Freshworks' 2025 Customer Service Benchmark Report found AI-powered tooling drove a measured **55% reduction in average first-response time** across CX teams it studied ([Unthread, 2026](https://unthread.io/blog/support-ticket-resolution-statistics/)). The opportunity is real. The risk of doing it badly — as Air Canada learned in a Canadian small-claims tribunal — is also real and legally binding. Aegis is designed specifically to capture the upside while engineering out the failure mode that got Air Canada sued.

---

## 2. The Problem

### 2.1 The economics, in plain terms

Every support ticket that reaches a human agent costs money whether or not it needed a human. Benchmarks vary by vertical, but the pattern is consistent across every source: **SaaS support runs $18–$35 per ticket, B2B/technical support runs $30–$60 per ticket**, driven by handle time, escalation chains, and the fact that a meaningful share of "resolved" issues actually take multiple contacts to close — one industry analysis puts the average at 2.3 contacts per issue, meaning the *real* cost per resolved issue is 2.3x the headline cost-per-contact figure ([Lorikeet, 2026](https://www.lorikeetcx.ai/articles/customer-service-cost-per-ticket)).

Here's the part that matters for a RAG system specifically: **a large fraction of that volume is repetitive.** Anyone who has worked adjacent to a support team knows the shape of the ticket queue — a small number of distinct underlying questions ("how do I reset my API key," "why is my webhook failing," "how do I upgrade my plan") account for a disproportionate share of total volume. This is precisely the situation retrieval-augmented generation was built for: the answers already exist, scattered across documentation, changelogs, and the resolution history of thousands of past tickets. The problem isn't a lack of information. It's that no human can hold that entire corpus in their head, search it in under a second, and produce a correctly-cited answer 500 times before lunch. A machine can — *if* it is grounded in the real corpus rather than improvising from a language model's parametric memory.

### 2.2 The real-life example: what happens when you skip the "grounded" part

In November 2022, a Vancouver resident named Jake Moffatt visited Air Canada's website following the death of his grandmother and asked the airline's support chatbot about bereavement fares. The chatbot told him he could book a normal-price ticket and apply for the bereavement discount retroactively, within 90 days. He did exactly that. When he later filed the claim, Air Canada rejected it — the airline's actual policy, published elsewhere on its own site, required the discount to be requested *before* travel, with no retroactive refunds ([Forbes, 2024](https://www.forbes.com/sites/marisagarcia/2024/02/19/what-air-canada-lost-in-remarkable-lying-ai-chatbot-case/)).

Moffatt took Air Canada to Canada's Civil Resolution Tribunal. Air Canada's defense was, essentially, that the chatbot was a separate entity and the airline couldn't be held liable for what it said. The tribunal rejected that argument outright, ruling that a chatbot is simply part of a company's website like any other page, and found Air Canada liable for **negligent misrepresentation** for failing to take reasonable care that its chatbot's outputs were accurate ([McCarthy Tétrault, 2024](https://www.mccarthy.ca/en/insights/blogs/techlex/moffatt-v-air-canada-misrepresentation-ai-chatbot)). Air Canada was ordered to pay the fare difference plus tribunal fees — a modest CAD $812.02 in this instance ([AiBusiness, 2024](https://aibusiness.com/nlp/air-canada-held-responsible-for-chatbot-s-hallucinations-)) — but the precedent was the real cost: companies are now on notice that an AI support agent's output carries the same legal weight as a page on their own website, whether or not the underlying answer was actually retrieved from a real policy document.

**This is not a hypothetical edge case for this project — it's the exact failure mode the architecture in Section 4 is designed to prevent.** Air Canada's chatbot didn't fail because AI is inherently untrustworthy. It failed because it generated an answer instead of retrieving one, with no citation trail, no confidence threshold, and no mechanism to say "I'm not certain — let me connect you with someone who can confirm this." Those three missing pieces — grounding, citation, and calibrated abstention — are the core engineering contribution of a properly built RAG system, and they are exactly what a naive LLM wrapper doesn't have.

### 2.3 Why RAG, specifically, and not the alternatives

- **Why not a static FAQ / keyword search?** Real user questions don't match documentation phrasing. A user typing "it doesn't work" or "webhook keeps timing out on retries" needs semantic understanding to be routed to the correct doc section, something keyword search alone cannot reliably do — although, as shown below, keyword (lexical) matching is still essential *alongside* semantic search for exact identifiers like error codes and config keys.
- **Why not fine-tuning an LLM on support transcripts?** Documentation changes constantly — pricing pages, API contracts, feature deprecations. A fine-tuned model bakes a snapshot of "truth" into its weights and goes stale the moment anything changes, with no way to audit *which* fact it learned from *which* source. It also cannot cite its answer, which is precisely the property that would have protected Air Canada.
- **Why not a plain LLM with a long system prompt ("just paste the docs into context")?** This works at small scale and fails as the corpus grows past what fits in a context window, and it still doesn't solve retrieval precision, version-awareness, or auditability — you'd be re-deriving RAG badly instead of building it properly.
- **Why RAG is the right fit:** the correct answer is version-specific, policy-specific, and must be *provably* sourced from a document that a human explicitly published as current and correct. RAG retrieves that document, cites it, and — this is the differentiator this proposal leans on hardest — refuses to answer when nothing in the corpus supports a confident response.

---

## 3. Who Has This Problem — Target Users & Market

### 3.1 Primary user segments

| Segment | Who they are | What they need |
|---|---|---|
| **End customers** | Users of a SaaS/API product hitting a support question outside business hours or wanting an instant answer | A correct answer in seconds, with the honesty to say "I don't know, let me get you a human" rather than confidently guessing |
| **Support agents (Tier 1/2)** | Human agents currently drowning in repetitive tickets | A system that deflects the repetitive 60–70% so they can focus on genuinely hard, high-empathy, or account-specific cases |
| **Support Ops / CX leadership** | The people who own the cost-per-ticket and CSAT metrics | Auditable deflection numbers, an escalation policy they can trust legally and reputationally, and a system whose accuracy is *measured*, not asserted |
| **Engineering / Platform teams** | Whoever owns the documentation and the product | A feedback loop showing which questions the system can't answer — which is also a live signal of documentation gaps |

### 3.2 Realistic scale

Any company running a documented product with an active user base and a public issue tracker or discussion forum has this exact problem. For a portfolio build, the natural free-data analog is a large, actively maintained open-source project (its documentation stands in for product docs; its closed GitHub Issues/Discussions stand in for resolved support tickets) — which conveniently gives a corpus in the tens of thousands of real questions with real, verifiable ground-truth resolutions, at zero cost and with no proprietary data risk.

---

## 4. Proposed Solution — System Overview

Aegis is not a single retrieval call bolted to an LLM. It is a small **agentic pipeline**: incoming queries are first classified and routed, then handled by one of several specialized paths, each with its own retrieval strategy, before a final grounded-generation and citation-validation step decides whether to answer or escalate.

```
┌─────────────┐    ┌──────────────┐    ┌────────────────────────────────┐
│ User Query  │ →  │ Query Router │ →  │  Route A: Docs/How-To           │
│ (+ history) │    │ (classifier) │    │  Route B: Known Issue / Bug     │
└─────────────┘    └──────────────┘    │  Route C: Billing/Account       │
                                        │           → immediate escalate │
                                        └────────────────────────────────┘
                                                       │
                          ┌────────────────────────────┴─────────────────────────┐
                          ▼                                                      ▼
              ┌───────────────────────┐                              ┌───────────────────────┐
              │ Query Rewriting /      │                              │ Semantic + Exact Cache │
              │ Multi-Query Expansion  │                              │ (short-circuit on hit) │
              └───────────┬────────────┘                              └───────────┬────────────┘
                          ▼                                                        │
              ┌───────────────────────┐                                           │
              │ Hybrid Retrieval       │                                          │
              │ (Dense + BM25, RRF)    │                                          │
              └───────────┬────────────┘                                          │
                          ▼                                                        │
              ┌───────────────────────┐                                           │
              │ Cross-Encoder Reranker │                                          │
              └───────────┬────────────┘                                          │
                          ▼                                                        │
              ┌───────────────────────┐                                           │
              │ Confidence Scoring &   │                                          │
              │ Abstention Threshold   │                                          │
              └───────────┬────────────┘                                          │
                    ┌──────┴───────┐                                              │
                    ▼              ▼                                              │
        ┌────────────────┐  ┌────────────────┐                                   │
        │ Grounded        │  │ Escalate to     │                                  │
        │ Generation +    │  │ Human, with     │                                  │
        │ Citation Check  │  │ Auto-Summary    │                                  │
        └────────┬────────┘  └────────┬────────┘                                 │
                  └────────────────────┴──────────────────────────────────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │ Response to User + Trace   │
                          │ logged to Observability     │
                          └───────────────────────────┘
```

The single most important design decision in the whole system is the **confidence scoring and abstention threshold** step. Everything upstream of it exists to make retrieval good enough that the threshold rarely has to fire; everything downstream exists to make sure that when it does fire, the handoff to a human is smooth rather than a dead end.

---

## 5. Technical Architecture (Deep Dive)

### 5.1 Corpus and ingestion

**Sources:** product/API documentation (Markdown/HTML), release notes and changelogs, and a historical archive of resolved support tickets or closed GitHub Issues/Discussions, which double as a labeled dataset — each closed issue already has a known "correct" resolution, which is unusually valuable for building an evaluation set later.

**Parsing:** documentation is parsed with structure preserved (headings, code blocks, tables) using a layout-aware parser (e.g., Docling) rather than a naive text-strip, because code blocks and their surrounding explanatory prose must never be separated during chunking.

**Content-hash change detection:** every source document is hashed at ingestion time; re-ingestion only re-processes documents whose hash has changed, which is the foundation of the delta-indexing strategy described in 5.9.

### 5.2 Chunking strategy

Fixed-size chunking is used only as a deliberate baseline to measure against — it is not the shipped strategy. The production strategy is **heading-aware, structure-preserving chunking**: split on Markdown/HTML heading boundaries, keep code blocks atomic and attached to their explanatory text, and cap chunk size with an overlap only when a section exceeds the model's effective context. Each chunk is enriched before embedding with a short prepended context string — its document title, heading path, and product version — a technique generally referred to as **contextual retrieval**, which exists specifically to solve the problem of a chunk being semantically meaningless once separated from the document it came from (a paragraph that says "set this to 30 seconds" is useless without knowing it's describing a webhook retry timeout, not a session timeout).

A/B evidence is required, not assumed: the deliverable includes a table comparing fixed-size, recursive, and heading-aware chunking against the golden evaluation set (Section 5.8), reporting Recall@10 and MRR for each, with the winning strategy shipped and the reasoning documented.

### 5.3 Embedding and hybrid retrieval

**Embedding model:** an open-weight sentence embedding model (e.g., BGE-large or BGE-M3) run locally at zero marginal cost, benchmarked against at least one alternative on the project's own golden set rather than trusted from a public leaderboard, since leaderboard rankings on general benchmarks do not reliably predict performance on a specific, narrow support corpus.

**Why hybrid, specifically, and not dense-only:** support queries are full of tokens that dense embeddings routinely blur — exact error codes, config key names, HTTP status codes, SDK method names, version numbers. A user searching for `ECONNREFUSED` or `webhook_retry_max` needs exact lexical matching (BM25 or Qdrant's native sparse vectors), fused with dense semantic retrieval via **Reciprocal Rank Fusion (RRF)**. This is not a theoretical justification — it is validated empirically against the golden set, with the fusion weighting tuned rather than left at defaults, and the resulting recall lift over dense-only retrieval reported explicitly.

**Vector store:** Qdrant, using named dense + sparse vectors on the same collection, filterable by metadata (`product_area`, `doc_version`, `doc_type`) so retrieval can be scoped before similarity search runs, not after.

### 5.4 Query understanding and agentic routing

Real support queries are frequently short, ambiguous, and stripped of context ("it's broken again"). Before retrieval, the query passes through:

- **Classification/routing** into one of several handling paths — documentation/how-to questions, known-bug/issue-tracker questions, or account/billing questions that should escalate immediately without ever attempting retrieval (because giving automated financial commitments is exactly the category of risk Air Canada's case illustrates).
- **Query rewriting**, incorporating recent conversation turns so a follow-up like "what about on the free tier?" is correctly expanded using prior context.
- **Multi-query expansion**, generating 2–3 paraphrased variants of the query and retrieving for each, then merging — a technique that measurably improves recall on vague queries and is evaluated with a before/after comparison, not assumed to help.

This routing layer is what makes the system *agentic* rather than a single retrieve-then-generate call: it is making a decision about *how* to handle the request before committing to a retrieval strategy, and a billing question never touches the RAG pipeline at all.

### 5.5 Reranking

The top-k candidates from hybrid retrieval (typically 20–50) are passed through a cross-encoder reranker (e.g., a BGE reranker) to reorder by true relevance to the specific query, which dense/sparse retrieval alone cannot achieve because it scores query and document independently rather than jointly. Reranking is expensive per-pair, so the design explicitly measures the **precision gain versus the added latency** of reranking at different candidate-pool sizes (rerank top-10 vs. top-50) and selects a pool size that stays within the latency budget defined in Section 5.10.

### 5.6 Confidence scoring, guardrails, and calibrated abstention

This is the component that did not exist in Air Canada's chatbot and is the core differentiator of this proposal. After reranking, the system computes a confidence signal from the reranker's top score and the semantic agreement across retrieved chunks. If confidence falls below a threshold tuned against the golden set, the system does not attempt an answer — it escalates immediately, with an auto-generated summary of the user's question and the (insufficient) evidence it found, handed to a human agent.

Additional guardrails:
- The model is instructed, and separately validated, to **never state a policy, refund, or commitment that isn't explicitly present in a retrieved chunk.**
- Every factual claim in a generated answer is required to carry an inline citation to a specific chunk ID; a **post-generation citation validator** checks that cited chunks actually support the claim made, and strips or flags any claim that doesn't pass.
- Billing, legal, and account-specific queries are routed to immediate human escalation by policy, never answered generatively, regardless of retrieval confidence — a direct, deliberate design response to the Air Canada precedent.

The single most valuable artifact this project produces is a **confusion matrix**: correctly-answered / incorrectly-answered / correctly-escalated / incorrectly-escalated, measured against the golden set. Publishing that table with real numbers is what proves the escalation policy is calibrated rather than guessed.

### 5.7 Generation

Generation uses a fast, free-tier-hosted LLM (e.g., Groq-hosted Llama or an equivalent open-weight model, with a fallback provider) with a system prompt that requires numbered citations to retrieved chunk IDs and forbids answering from outside the provided context. Responses are streamed to the user via server-sent events for perceived latency, while the citation validator runs on the complete response before it's finalized.

### 5.8 Evaluation

- **Golden set (100–300 queries):** built from real closed support tickets / resolved GitHub issues, where the ground-truth resolution is already known — giving free, high-quality labels rather than requiring manual answer-writing from scratch.
- **Retrieval metrics:** Recall@k, MRR, and NDCG@k, computed per chunking strategy and per retrieval configuration (dense-only vs. hybrid vs. hybrid+rerank), reported as a comparison table.
- **Generation metrics:** faithfulness, answer relevancy, and context precision via RAGAS, plus an **LLM-as-judge** for answer helpfulness that is itself validated by checking its agreement rate against a sample of human-labeled judgments — a step frequently skipped, and one worth explicitly reporting.
- **The escalation confusion matrix** described in 5.6, which is the single most business-relevant number in the whole evaluation suite.
- **Regression gate:** a CI job (GitHub Actions) that re-runs the full evaluation suite on every change to the pipeline and fails the build if any metric drops below its threshold, preventing silent regressions of the kind that plague RAG systems in production when someone tweaks a prompt and doesn't realize retrieval quality dropped.

### 5.9 Caching and delta indexing

Support queries are highly repetitive by nature, which makes caching unusually high-leverage here specifically. Two layers are implemented:

- **Exact-match cache**, short-circuiting identical repeated queries.
- **Semantic cache**, serving a cached answer when a new query embeds above a similarity threshold to a previously-answered query — with the cache key including a corpus-version hash so a documentation update automatically invalidates stale cached answers rather than serving outdated information.

The deliverable reports a measured **cache hit rate and the resulting cost/latency reduction**, since this is a directly quantifiable unit-economics story and a compelling one in an interview.

**Delta indexing:** a scheduled job detects changed source documents via content hash, re-chunks and re-embeds only what changed, and purges chunks belonging to deleted or superseded documents — the classic failure mode where a deleted or outdated policy page continues to be retrievable and cited long after it stopped being true (the exact category of risk that sank Air Canada) is explicitly tested against.

### 5.10 Observability and latency budget

Every request is traced end-to-end (via Langfuse or Arize Phoenix) — routing decision, retrieved chunks with scores, rerank scores, confidence score, generation, citation validation result, and final latency broken down by stage. This is both a debugging tool and, in production, an audit log: if a customer disputes an answer, the exact evidence the system retrieved and the confidence it assigned is reconstructable after the fact, which is precisely the documentation trail Air Canada lacked. Target latency: p95 end-to-end under 3 seconds for the retrieval path, with billing/account queries escalating near-instantly since they skip retrieval entirely.

### 5.11 Deployment

FastAPI backend with streaming responses, containerized, deployed on a free-tier host (e.g., Hugging Face Spaces or Render); vector store self-hosted or on a free-tier managed instance; a lightweight chat UI exposing citations and, critically, a visible "escalated to human" state so the demo can show the abstention behavior live rather than only the happy path.

---

## 6. Business Value and KPIs

| Metric | What it measures | Why leadership cares |
|---|---|---|
| **Deflection rate** | % of queries resolved without human escalation | Directly maps to the $18–$60/ticket cost avoided ([Lorikeet](https://www.lorikeetcx.ai/articles/customer-service-cost-per-ticket)) |
| **Escalation precision/recall** | Confusion matrix from Section 5.6 | Proves the system escalates *correctly*, not just often — the trust metric |
| **Cost per resolved query** | Blended AI + human cost | Grounded in the ~12x cost gap between AI and human interactions reported by Fullview/CX Update data ([Unthread, 2026](https://unthread.io/blog/support-ticket-resolution-statistics/)) |
| **First response time** | Time to first useful response | Freshworks' 2025 benchmark data found a 55% reduction attributable to AI tooling in the teams it studied ([Unthread, 2026](https://unthread.io/blog/support-ticket-resolution-statistics/)) |
| **Faithfulness / hallucination rate** | RAGAS faithfulness score | The metric that, unmeasured, is exactly what got Air Canada sued |
| **Cache hit rate** | Repeated-query cost savings | Directly reduces LLM inference spend without touching accuracy |

*Note on the figures above: they are third-party industry benchmarks, not measurements of this specific system. They establish the size of the opportunity; the actual deflection rate, cost-per-query, and hallucination rate this system achieves must be measured against its own golden set and reported honestly, including where it underperforms.*

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| System confidently gives a wrong answer (the Air Canada failure mode) | Mandatory citation grounding + post-generation citation validation + calibrated abstention threshold tuned on the golden set |
| Escalation threshold set wrong (too eager → no cost savings; too lax → wrong answers reach users) | Confusion matrix evaluation explicitly optimizes and reports this tradeoff rather than picking a threshold by feel |
| Documentation changes and the index goes stale | Content-hash delta indexing with a defined freshness SLO |
| Cache serves an outdated answer after a docs update | Cache keys include a corpus-version hash, forcing invalidation on change |
| LLM provider free-tier rate limits under load | Multi-provider fallback (e.g., Groq primary, Gemini or local Ollama fallback); note that free-tier limits and terms change frequently and should be reverified against the provider's current pricing page before relying on them |
| Billing/legal questions answered automatically | Hard-routed to human escalation by policy, bypassing generation entirely, regardless of retrieval confidence |

---

## 8. MVP Scope and Stretch Goals

**MVP (2–3 weeks):** one product's documentation + ~1,000 resolved support tickets/issues as the corpus; three-way query router; hybrid retrieval + reranking with a published A/B table; semantic + exact caching; calibrated escalation with a published confusion matrix; a 150–200 query golden set built from real closed issues; end-to-end tracing; a deployed chat UI showing citations and the escalation state live.

**Stretch goals:** multi-turn conversational memory with history-aware query rewriting; a feedback loop where user thumbs-down responses feed into a review queue and become new golden-set cases over time; automatic ticket-summary drafting handed to the human agent on escalation, so the handoff isn't a cold start.

---

## 9. Definition of Done

The system is complete when it can demonstrate, with real published numbers rather than assertions:

1. A measured Recall@10 and MRR improvement from baseline (naive dense, fixed-size chunking) to final (hybrid, heading-aware, reranked) configuration.
2. RAGAS faithfulness ≥ 0.90 on the golden set.
3. A published escalation confusion matrix with a stated, justified precision/recall tradeoff.
4. A cache hit rate with the corresponding measured cost/latency reduction.
5. A CI evaluation gate that demonstrably fails when the pipeline is deliberately regressed.
6. p95 end-to-end latency under 3 seconds on the retrieval path.
7. A documented failure-analysis section listing real queries the system still gets wrong, categorized by cause.

---

## Sources

- Lorikeet, "Cost Per Support Ticket: Benchmarks and How to Reduce It" (2026) — https://www.lorikeetcx.ai/articles/customer-service-cost-per-ticket
- Supportbench, "How do you calculate the real cost per ticket?" (2026) — https://www.supportbench.com/calculate-real-cost-per-ticket-justify-headcount/
- Unthread, "23 Support Ticket Resolution Statistics by Complexity" (2026) — https://unthread.io/blog/support-ticket-resolution-statistics/
- Forbes, "What Air Canada Lost In 'Remarkable' Lying AI Chatbot Case" (Feb 2024) — https://www.forbes.com/sites/marisagarcia/2024/02/19/what-air-canada-lost-in-remarkable-lying-ai-chatbot-case/
- McCarthy Tétrault, "Moffatt v. Air Canada: A Misrepresentation by an AI Chatbot" (2024) — https://www.mccarthy.ca/en/insights/blogs/techlex/moffatt-v-air-canada-misrepresentation-ai-chatbot
- AiBusiness, "Air Canada Held Responsible for Chatbot's Hallucinations" (Feb 2024) — https://aibusiness.com/nlp/air-canada-held-responsible-for-chatbot-s-hallucinations-

**A note on rigor:** the cost-per-ticket and deflection figures above are third-party industry benchmarks compiled by vendor and research sites (Lorikeet, LiveChatAI, Unthread, Fullview), not independently audited data — treat them as directional and verify against primary sources (e.g., Zendesk, Gartner, SaaS Capital reports) before citing them in a resume or interview as fact. The Air Canada case, by contrast, is a matter of public legal record and is reported consistently across Forbes, a law firm's own analysis, and the original tribunal reasoning, so it's on much firmer ground as a citable example.
