# 14g — Feedback review queue and golden-set candidates

**Phase:** 14 · **Date:** 2026-10-17 · **Files:** `src/aegis/eval/review_queue.py`, `src/aegis/api/store.py`, `src/aegis/eval/negatives.py`, `tests/test_review_queue.py`
**Topics from the course:** Building a golden set / evaluation data (Module 4); What makes production challenging (Module 5, #29)

## 1. What I built

The second half of plan.md Phase 14 step 5: "Thumbs up/down → feedback store → review queue → golden-set candidates." Voted interactions (already persisted since 14c) form a queue, 👎 first. A person reviews each one in a terminal session and marks it a positive candidate (with a stratum), a negative candidate (with a category), or a reject. `--export` writes every promoted item to `data/golden/candidates_feedback.jsonl` in the golden-set schema, with `split="candidate"`, no ground-truth chunks, and full provenance.

## 2. Why this approach

**A vote is not a label.** The real run in §5 shows it. A user 👎'd a billing question that was correctly hard-escalated ("I just wanted to know who to contact"). The system did exactly what policy requires; the vote measures the user's satisfaction, not system correctness. Auto-promoting 👎s as "failures" would have put a correct behavior into the eval set as a defect. So every item gets a human decision. The queue only orders them: 👎 first, because that's where real failures are most likely and the golden set (built from harvested GitHub issues) has none from real use.

**Candidates, never edits.** `golden_set_v1.jsonl` is frozen and hashed (Phase 4; CLAUDE.md §6 reproducibility). The export is a separate file. A v2 golden set would label these through the same Phase 4 process and freeze under a new version, so every published number stays tied to the set it was measured on.

**Two guards against corrupting future measurements:**
- **No system citations in the export.** Design record 12e found the golden set's labels are already biased toward the live index that surfaced the candidates: live HNSW and exact cosine agree on 57.8% of top-10, and ground-truth chunks found only by the live index outnumber those found only by brute force 348 to 95. Pre-filling labels from the answering system's own citations would score the system against its own output. Positive candidates are exported with `ground_truth_chunk_ids=[]`, and a test asserts no citation data appears in a row.
- **Duplicate and holdout check.** Each candidate's normalized text is checked against every existing golden query, *holdout included*, and flagged with `duplicate_of: {id, split}`. Anyone can type a holdout question into the chat; promoting it into dev would leak the holdout without anyone noticing.

**Follow-ups export their standalone text.** "what about making it required?" is meaningless without its history. The candidate's `text` is the rewritten query (what the system actually searched for), and the user's words are kept in `feedback.original_query`.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Review page served by the API (`/review`) | Nicer than a terminal; same UI stack as 14e | The API has no auth (Phase 15). A review page would expose every user's query and answer, plus a write path into evaluation data, to anyone who can reach the port. The terminal tool matches `label_cli` and has no such surface. |
| Auto-promote 👎 to candidates (no human step) | Zero reviewer time; "closes the loop" literally | §5's real billing case: a 👎 on a correct escalation. A vote says the user was unhappy, not that the system was wrong. |
| Append promoted items straight into the golden set | One file, immediately measured | Breaks the frozen, hashed set every published number was measured on (CLAUDE.md §6). |
| Pre-fill ground truth from the answer's validated citations | Saves the labeler the most time | Self-grading: see the 12e label-bias finding above. |
| Stratum heuristic as an Enter-to-accept default | Faster review | Measured: on query text alone it agrees with the frozen strata on **138/180** positives (76.7%). It was wrong on the first real holdout query tried (`golden::issue::11134`: suggested `error_debugging`, frozen `how_to`). Shown as a hint; the reviewer must type a stratum. |

## 4. How it works

`InteractionStore` gains a `reviews` table: `trace_id` (FK to `feedback`), `decision ∈ {positive, negative, reject}`, `stratum`, `negative_category`, `note`, `reviewed_at`. The schema enforces the decision through a `CHECK`, and the code enforces the conditional fields: a positive needs a stratum, a negative needs a category.

```python
def pending_review(self):            # voted AND not reviewed; 👎 first, then oldest first
    ... "WHERE f.trace_id NOT IN (SELECT trace_id FROM reviews) ORDER BY f.rating = 'up', f.created_at"
```

`export_candidates` rebuilds the whole file from `reviews` in `trace_id` order (deterministic and idempotent, tested). Each row is validated by constructing a real `GoldenQuery`. Negatives take `expected_behavior` from the same map Phase 4's negatives use (`negatives.EXPECTED_BEHAVIOR`, now public), so a promoted billing question gets `hard_escalate`, exactly like the hand-written ones. One exported row:

```json
{"id": "feedback::256d…31af", "text": "Can I get a refund on my enterprise support invoice?", "is_negative": true,
 "negative_category": "billing_legal", "expected_behavior": "hard_escalate", "split": "candidate",
 "ground_truth_chunk_ids": [],
 "feedback": {"rating": "down", "comment": "I just wanted to know who to contact", "original_query": "…",
              "answer_shown": "This request has been escalated to a human agent.", "escalated": true, "…": "…"}}
```

`review_session(store, input_fn, print_fn)` is the interactive loop, with injectable I/O so tests can script it. Every decision is written immediately; `s` skips and `q` quits, leaving items pending.

## 5. What the numbers said

`uv run pytest -q` → **576 passed** (+13 this unit).

**Real end-to-end** (real server on a throwaway interactions DB, real HTTP, n=2 real votes + 1 injected item):

- `/chat` "how do I set a default value for a query parameter" → answered (exact-cache hit), voted 👍.
- `/chat` "Can I get a refund on my enterprise support invoice?" → `escalation` with `cause=policy`, voted 👎 with the comment "I just wanted to know who to contact".
- The first real holdout query (`golden::issue::11134`) was recorded and voted 👎 directly, to test duplicate detection against the real frozen files.
- `pending_review()` ordered the two 👎 first. A scripted review promoted all three, and the export wrote 3 rows. The holdout item came out with `duplicate_of: {"id": "golden::issue::11134", "split": "holdout"}`, and the billing item with `expected_behavior: hard_escalate`.

**Stratum heuristic agreement:** `classify_stratum(text, "", "")` vs the frozen stratum, all 180 positives (dev + holdout): **138/180**.

The real feedback store currently holds 6 interactions and **0 votes**. There are no real items to review yet; the loop is built and verified, not yet fed.

## 6. What surprised me / what to watch for

- **The first real 👎 was on a correct behavior.** That's n=1, so not a rate, but it's exactly the case that rules out auto-promotion.
- **The stratum heuristic is weaker here than at harvest** because it has no issue resolution to read. 23% disagreement on its own home data means per-stratum numbers for any v2 depend on the reviewer, not the heuristic.
- **Duplicate detection is exact-after-normalization only.** A holdout question paraphrased by a user won't be flagged. A semantic check (the Phase 12 encoder at the cache's 0.88 threshold) would catch more, but Phase 12 already measured real overlap between paraphrase and different-question similarity on this corpus, so it would also produce false flags. Left exact; the reviewer sees the text.
- **User text lands in evaluation data.** Candidates carry real user queries and comments. Fine for a single-user portfolio deployment, a PII question for anything real (Phase 15).
- Positive candidates still need chunk labels before they can be measured. `label_cli` hard-codes Phase 4's file paths, so labeling a v2 needs a small change there. Not done; nothing to label yet.

## 7. Where this fits and what's next

Closes plan.md Phase 14 step 5: feedback persists (14c), and now it has a path to evaluation data. All six Phase 14 build items are built. Remaining Phase 14 work is evidence, not features: the semantic-cache collision found in 14e (a follow-up served the earlier generic answer), the 9.3 s cold `cache_lookup`, and the regime-C multi-turn eval that the last exit criterion ("multi-turn follow-ups resolve correctly") depends on.
