# 02 — Chunking Strategies

**Phase:** 2 · **Date:** 2026-07-30 · **Files:** `src/aegis/chunking/{base,tokens,segments,merge,fixed,recursive,heading_aware,semantic,factory}.py`, `experiments/02_chunking_stats.py`
**Topics from the course:** Chunking strategies (fixed, overlapping, recursive, structure-specific, semantic, LLM-based, context-aware)

## 1. What I built

Four chunkers behind one `Chunker` protocol (`chunk(document: Document) -> list[Chunk]`), config-swappable via `build_chunker(strategy)`: **fixed-size** (token sliding window, the naive baseline), **recursive** (paragraph → sentence → word descent), **heading-aware** (splits on Markdown headings, keeps fenced code blocks atomic), and **semantic** (embeds sentences, cuts at cosine-distance peaks). Plus **contextual enrichment**, a toggleable prefix (`{title} > {heading_path} > {version}`) applied to a separate `embedding_text` field so citations still show the reader real source text. Measured against the real Phase 1 corpus: 3,659 documents, all four strategies compared on chunk count, token distribution, and code-block integrity rate.

Not built: **LLM-based chunking** (the seventh kind in the course list) — same treatment as ColBERT in the Phase 7 plan, a design-record topic only. An LLM call per boundary decision across 3,659 documents (many issues alone run 10-20+ sections) is a real cost for a technique whose main selling point — better judgment at ambiguous boundaries — is hard to distinguish from heading-aware's structural approach on a corpus that already has Markdown headings and fenced code as strong, free structural signal. Worth revisiting only if Phase 6's ablation shows all four strategies plateau below what the golden set demands.

## 2. Why this approach

Phase 0's baseline produced a reproduced, concrete failure: a raw 500-char slice severed a `(deprecated)` heading from the body explaining it, and the model recommended the deprecated API with full confidence. The four strategies here exist specifically to produce **measured** answers to "how often does that happen, and what stops it" — fixed/recursive are the naive/moderate baselines that can still fail this way; heading-aware and semantic are built so it's structurally impossible for a fenced code block to be split, with heading-aware additionally attaching each chunk to the exact Markdown section it came from (a real citation-quality difference the other three don't have: only heading-aware populates `heading_path` in a way that reflects the document's actual structure).

The corpus itself set two hard requirements before any code was written: Phase 1 found `release-notes.md` (686,041 chars) and issue #2266 (193,590 chars, a pasted build log) as legitimate long-tail content, not parsing bugs — any chunker has to have a defined, non-crashing behavior for a single document an order of magnitude larger than the rest of the corpus. And the corpus mixes clean, git-authored Markdown (docs) with raw, unstructured, occasionally malformed user-pasted text (issues) — the two behave differently enough that several real bugs (§6) only ever showed up in the issues half.

## 3. Alternatives rejected

| Alternative | The case for it | Why rejected here |
| --- | --- | --- |
| Give fixed and recursive the same code-block atomicity guarantee as heading-aware/semantic | Fewer severed code blocks everywhere, strictly "safer" output | Would erase the exact signal Phase 6's ablation grid needs to see: fixed and recursive are supposed to be the naive/moderate baselines, and their code-integrity rate (84.1% / 85.3% at `max_tokens=400` — see §5) is itself a real, useful measurement, not a defect to paper over. |
| spaCy/nltk sentence segmentation for the semantic chunker | Linguistically correct sentence boundaries, handles abbreviations (`e.g.`, `Dr.`) correctly | A regex splitter (`(?<=[.!?])\s+(?=[A-Z0-9"'`])`) is enough to test whether semantic chunking pays for itself at all here, and plan.md already flags semantic chunking as the most likely strategy to be cut (§7 cut-lines). A heavier NLP dependency for a technique that might not survive Phase 6 isn't justified yet — reconsider only if semantic chunking is kept. |
| Reuse `greedy_merge`'s generic pack-with-overlap for the fixed-size chunker's sliding window | One packing implementation instead of two | `greedy_merge` assumes pieces are individually much smaller than `max_tokens`; fixed-size windows are deliberately *at* the budget, so adding an overlap prefix on top would (and in an early version, did) push the merged result over `max_tokens`. Fixed-size keeps its own bespoke loop with `nearest_safe_boundary` snapping instead. |
| Per-heading-section boundaries for semantic chunking (bound sentence grouping within each Markdown section, like heading-aware) | Keeps `heading_path` populated, more consistent with the rest of the pipeline | Chose the "purer" whole-document sentence stream instead, which is the textbook technique and gives an honest, uncorrelated third data point against heading-aware. Real, measured cost: every semantic chunk has `heading_path == []` (0.0% in §5's table) — a genuine strike against the strategy that Phase 6 should weigh against whatever retrieval-quality gain it shows, not something to quietly avoid measuring. |
| `tiktoken`'s `cl100k_base` for token counting vs. loading the actual embedding model's tokenizer (WordPiece/SentencePiece — bge-large and BGE-M3 aren't even chosen until Phase 3) | Exact token-length parity with what will actually be embedded | Chunk boundaries only need a stable, fast, dependency-light count that's consistent with itself across strategies — not parity with a tokenizer that doesn't exist yet in this project. Loading an ONNX model just to count tokens would mean every chunker pays fastembed's startup cost. `max_tokens=400` is set with headroom below any real encoder's 512-token limit specifically to absorb this approximation plus the enrichment prefix (§4). |
| A strict, CommonMark-compliant closing-fence match (closing line must be *only* backticks, optional trailing whitespace) | "Correct" per the Markdown spec | Real corpus text violates it twice over (§6) — CRLF line endings and a closing fence sharing its line with trailing HTML. A strict match's failure mode is catastrophic (swallows the rest of the document into one "code" segment); a lenient match's failure mode is a handful of documents with genuinely unbalanced backtick counts still misparsing (§6) — the lenient failure mode is far cheaper. |

## 4. How it works

**Code-block atomicity** starts from one shared primitive (`segments.py`) that both heading-aware and semantic build on — split text into alternating text/code `Segment`s, fenced blocks kept whole:

```python
_FENCE_RE = re.compile(r"^```[^\n]*\n.*?(?:\n```[^\n]*|\Z)", re.MULTILINE | re.DOTALL)
```

The closing alternative matches a line that *starts* with `` ``` ``, not a line that *is only* `` ``` `` — see §6 for why the stricter version silently swallowed real content.

**Packing is verify-then-accept, not estimate-then-hope.** Every one of the non-fixed strategies eventually hands a list of already-small pieces to `greedy_merge`, which decides whether the *next* piece still fits by actually joining and re-counting, not by summing each piece's own token count:

```python
if current and _join_tokens(current + [piece]) > max_tokens:
    flushed = _JOIN.join(current).strip()
    ...
```

This matters because BPE token counts aren't additive across concatenation — the join separator itself costs tokens, and two pieces that individually measure under budget can measure over budget once actually joined (§6). Re-encoding on every piece is cheap here specifically because an accumulator never exceeds `max_tokens` (~400 tokens), so it's a few hundred characters, not the whole document.

**Heading-aware section-building** scans for `#{1,6} ` lines only inside *text* segments (never inside code — a Python `# comment` can't be mistaken for a heading), maintaining a heading-level stack so nested sections get the full breadcrumb and sibling sections don't inherit each other's children:

```python
level, title = len(match.group(1)), match.group(2).strip()
while heading_stack and heading_stack[-1][0] >= level:
    heading_stack.pop()
heading_stack.append((level, title))
```

**Semantic breakpoint detection** embeds each sentence (via the same `bge-small-en-v1.5`/fastembed pairing Phase 0's baseline used — a real bake-off model would be overkill for drawing split points on a strategy this project already expects might not survive Phase 6), computes consecutive-sentence cosine distance, and cuts wherever a gap exceeds the configured percentile (default 95th) of that document's own distance distribution — a *relative*, per-document threshold, not an absolute one.

**Contextual enrichment** is a separate field, not a text mutation:

```python
def make_embedding_text(text, document, heading_path, enrichment):
    if not enrichment:
        return text
    prefix = enrichment_prefix(document, heading_path)
    return f"{prefix}\n\n{text}" if prefix else text
```

`Chunk.text` stays the literal source text for citations; `Chunk.embedding_text` is what Phase 3 actually embeds.

## 5. What the numbers said

Full run: `docs/results/chunking-stats.md` (129 docs + a seeded 50-issue sample + issue #2266, `max_tokens=400`, `overlap_tokens=50` shared across all four so the comparison is chunking logic, not size budgets):

| Strategy | Chunks | Chunks/doc | Token p50 | Chunks w/ `heading_path` | Code-block integrity | Wall time |
| --- | --- | --- | --- | --- | --- | --- |
| fixed | 1646 | 9.14 | 400 | 0.0% | 84.1% (880 blocks) | 0.5s |
| recursive | 1791 | 9.95 | 377 | 0.0% | 85.3% (880 blocks) | 2.7s |
| heading_aware | 2682 | 14.9 | 168 | 99.9% | 98.4% (880 blocks) | 1.3s |
| semantic | 1787 | 9.93 | 364 | 0.0% | 99.1% (880 blocks) | **303.2s** |

Full-corpus sweep (all 3,659 documents, no sampling, fast strategies only — semantic wasn't re-run at this scale, see below): **fixed and recursive: 0 chunks exceed `max_tokens` out of 8,984 and 9,561 respectively.** Heading-aware: 19,236 chunks, 427 over budget — 420 are the documented atomic-oversized-code-block exception (§6 has the concrete example), 7 are a separate, accepted residual limitation (also §6).

**Semantic chunking's operational cost is real and large, independent of retrieval quality.** Embedding `release-notes.md`'s 3,690 sentences alone took 211-248s (confirmed compute-bound, not a per-call-overhead artifact: one batched call vs. 32 per-segment calls differed by only ~15%). That's why the measurement above uses 180 documents, not the full 3,659 — extrapolating, a full-corpus run is on the order of hours, for a strategy plan.md's own cut-line analysis already flags as the least likely to earn its keep on structured technical docs.

**The stress-case documents Phase 1 flagged turned out not to be the stress case for this phase.** `release-notes.md` and issue #2266 were pinned into every measurement expecting they'd trigger the atomic-oversized-code exception — instead, both split cleanly with zero oversized chunks at `max_tokens=400` (see the spotlight table in `docs/results/chunking-stats.md`). The actual example: `doc::advanced/security/oauth2-scopes.md`, whose full 184-line example script (1,352 tokens) is included via a Phase 1 `{* *}` directive and repeated across 8 of the page's 27 sections, each one legitimately oversized and correctly left unsplit.

## 6. What surprised me / what to watch for

**Three real bugs, all found by measuring the full corpus, not by unit tests — because unit tests use small, clean inputs by construction.**

1. *Double-counted separators.* `recursive_split_pieces` re-attached each piece's original separator (`"\n\n"`, `". "`, etc.) as a suffix, and `greedy_merge` *also* inserted its own `"\n\n"` when joining — every merged chunk got the separator twice. Found via a real 443-token chunk (issue #15000) against a 400-token budget. Fix: pieces carry content only; `greedy_merge` is the sole source of inter-piece spacing.
2. *Sum-of-parts is not the same number as tokens-of-the-join.* Even after fix 1, a semantic-chunking test still produced a 26-token chunk against a 25-token budget — `greedy_merge`'s running total summed each piece's own token count, which never accounted for the join separator's own token cost. Fixed by making the packing decision verify-then-accept (join the actual candidate text, then count) instead of estimate-then-hope (§4). This is the more general lesson: BPE token counts are not additive across string concatenation, full stop, and any packing logic that assumes otherwise will eventually be wrong by a small but real margin.
3. *A regex fence scanner is only as good as its assumption that closing fences look like closing fences.* CRLF line endings (issue #15000: closing line is literally `` ```\r\n ``) and a closing fence sharing its line with trailing HTML (issue #9780: `` ```</div> ``) both broke a strict "closing line is only backticks" pattern, silently classifying everything after as one giant unclosed code block — which hides every heading after that point from `_build_sections`, since headings are only ever scanned inside text segments. issue #15000 went from 2 detected sections to the correct 9 once fixed.

**A residual limitation, deliberately not chased further.** After all three fixes, 7 of 19,236 heading-aware chunks (0.036%) still exceed `max_tokens` for a reason that isn't a parser bug: the raw issue text itself has an *odd* number of literal `` ``` `` occurrences (`issue::4548` has exactly one, in the whole document) — a genuinely unclosed fence in the source, which no stateless open/close regex can pair correctly by construction. All 7 affected documents are issues (user-generated, unstructured); zero are docs (clean, git-authored). This is the same category of call as Phase 1's decision to stop cataloging canned maintainer replies: bounded, quantified, confined to the noisiest part of the corpus, and only worth a real CommonMark parser if it ever proves load-bearing.

**Naive chunking's code-block blind spot is smaller than expected — at this budget.** Fixed-size chunking is "supposed to" shred code blocks indiscriminately, but at `max_tokens=400` it still gets 84.1% integrity, because many of this corpus's illustrative code examples are simply short enough to fit inside one window by chance, not by design. This is a real, budget-dependent result, not a property of the strategy — the prediction worth testing in Phase 6 is that this number should fall sharply at a smaller `max_tokens`, while heading-aware and semantic stay near 100% regardless, since their guarantee is structural, not incidental.

## 7. Where this fits and what's next

Phase 3 embeds `Chunk.embedding_text` (not `.text`) into Qdrant — contextual enrichment is already wired through every strategy, off by default, toggleable per the exit criteria. Phase 6's ablation grid sweeps `strategy` via `build_chunker()` and `contextual_enrichment` as its two chunking-related axes; both are already config-driven, nothing further to wire up there. Semantic chunking's measured ~250s-per-large-document cost is now a concrete number for that ablation to weigh against whatever Recall@10 it buys — if it doesn't clear a meaningful bar, plan.md's own cut-line reasoning (§7) already anticipated this and the design record here is the evidence for that call either way. The 7-chunk residual fence-parsing limitation is not expected to matter for retrieval quality given its size, but is worth a second look if Phase 8's router shows anomalous behavior specifically on issues with heavy inline code.
