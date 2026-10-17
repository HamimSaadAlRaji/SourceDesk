# 14e — Static UI (vanilla JS over the SSE contract)

**Phase:** 14 · **Date:** 2026-10-01 · **Files:** `src/aegis/api/static/{index.html,app.js,sse.js,style.css}`, `src/aegis/api/app.py`, `tests/test_ui_static.py`
**Topics from the course:** Latency vs. response quality (Module 5, #35); Guardrails / escalation UX (Module 5)

## 1. What I built

A no-build page served by the API at `/`: a chat thread that streams the answer as `token` events arrive, turns validated `[n]` markers into chips that expand the real source chunk, renders escalation as a different kind of card, sends the conversation as `history`, shows "Searched for: …" when a follow-up was rewritten, and has 👍/👎 (+ comment on 👎) wired to `/feedback`. Also shows time-to-first-text and total time per answer.

## 2. Why this approach

- **Plain HTML/JS, no framework or build step** (plan.md Phase 14 step 4): the UI's whole job is to render five event types; a bundler would be the largest thing in the repo.
- **`fetch` + stream reader, not `EventSource`:** `EventSource` is GET-only and `/chat` is a POST with a body. The SSE framing is parsed by a ~20-line incremental parser (`sse.js`) that keeps a partial trailing frame between network reads; that is the one piece of real logic, so it has a node-run test with a frame split across chunks.
- **Citations as trust UX, not decoration:** a `[n]` the validator confirmed is a chip that opens the actual retrieved chunk text; one it did *not* confirm is struck through with a tooltip. An unverified citation can't look like a verified one.
- **Escalation is a different component, not a styled answer:** amber dashed card, "Escalated to a human — no answer was generated", a one-line why (hard-routed vs. not confident enough after retrieval). It shares no CSS with an answer, so the abstention path is unmistakable.
- **Untrusted text:** answers and source chunks (GitHub issue text) only enter the DOM via `textContent`/`createElement`. A test asserts the JS contains no `innerHTML`/`insertAdjacentHTML`. Inline markdown (`**bold**`, `` `code` ``) is rendered from DOM nodes for the same reason.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| React/Vite | Components, ecosystem | Build toolchain for five event types; plan.md asks for static HTML + vanilla JS. |
| `EventSource` | Native reconnect | GET-only; can't send the query/history body. |
| `marked` + `DOMPurify` for full markdown | Complete rendering | Two dependencies and an HTML-sanitising surface, to fix a cosmetic gap. Only bold/code are rendered; lists stay as text. |
| Show all citations as plain `[n]` text | Zero work | Makes verified and unverified citations look identical, which defeats the point of validation. |

## 4. How it works

`ask()` POSTs `{query, history}`, feeds decoded chunks to the parser, and `handle()`s each event: `token` appends text, `citation` stores the chunk by id, `escalation` swaps the answer card for the escalation card, `done` re-renders the answer (now with chips), adds the feedback row, and pushes the exchange (the assistant text, or the escalation message) into `history`, capped at the server's `max_history_messages`. `error` appends an error box and says whether the visible answer is incomplete.

## 5. What the numbers said

`uv run pytest -q`: **542 passed** (+3: static serving, no-HTML-injection, split-frame SSE parse under node).

Real session in a real browser (Playwright-driven Chromium against `uvicorn`, real Groq/Gemini/Qdrant, throwaway caches), n=1 per case:

- **Streaming answer + citations:** rendered with 15 clickable chips on the "follow-up" answer; clicking `[3]` expanded the source `doc::tutorial/body.md::heading_aware::9` with its real text and heading path. 0 unverified citations in that answer.
- **Escalation:** "I was charged twice on my subscription, please refund me" rendered the amber escalation card (viewed in a screenshot), with the router's hard-escalation reason, no answer text.
- **Multi-turn:** after the earlier turns, "what about making it required?" showed "Searched for: 'How do I make a query parameter required in FastAPI?'".
- **Feedback:** 👎 → "Thanks — recorded."; the sqlite row held rating `down`, the original and rewritten query, `semantic_hit`, and the answer text.
- **Timing of the first (cold) request:** first text **22,719 ms**, done 29,815 ms. Server stages: cache_lookup **9,304**, routing 479, retrieval 66, confidence 2,033, generation **17,916**, validation 2; provider was **gemini**.

## 6. What surprised me / what to watch for

- **A real answer-quality problem showed up in the follow-up.** The rewritten question "How do I make a query parameter required?" was served as a **semantic cache hit** on the earlier generic "How do I declare a query parameter?" answer. The semantic threshold (0.88, Phase 12, tuned for zero false positives on 15 hard negatives) did not separate these two questions. The user got the generic answer, not one aimed at "required". I did not check whether that answer happens to cover it. This weakens the multi-turn story from Unit 4 and is Phase 12's known risk showing up in a real session.
- **The cold first request is far outside any latency budget:** 9.3 s in `cache_lookup` (a first-use cost the warm-up doesn't cover: warm-up skips the caches by design) and 17.9 s generating on Gemini. Why Groq wasn't used is not established; the fallback leaves no log line saying it failed over. Both are open items.
- **Answers are markdown; only bold and inline code render.** Lists and headings show as raw text with the model's `*`; source chunks show their raw markdown too (honest, but unpolished).
- **Dark mode, mobile layout and accessibility were not checked.** Only the light theme at desktop width was viewed.
- **Escalation reason text is the router's internal sentence** ("billing_account_legal always hard-escalates…"), which is developer-facing, not user-facing wording.
- **Verified in the browser in this order only**; no concurrent users, no mid-stream provider failure in the browser, no slow-network behavior.

## 7. Where this fits and what's next

Exit criteria status: streaming in a browser ✔ (n=1 session), citations clickable to real source text ✔, escalation visibly different ✔, feedback persists ✔, **multi-turn resolves correctly: not established** (wiring works; the example got a generic cached answer; regime-C eval still owed). Remaining Phase 14: review queue / golden-candidate export, rate limiting + graceful provider-failure handling, a real escalation handler. Then the p95/cache-cold findings above need an owner.
