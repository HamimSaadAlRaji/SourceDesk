"""Phase 2 measurement: chunk count, token distribution, and code-block
integrity rate per strategy, run against the real Phase 1 corpus.

Semantic chunking embeds every sentence. Measured directly on this machine:
embedding `release-notes.md`'s 3,690 sentences alone takes ~211s with
`bge-small-en-v1.5` (ONNX/fastembed) — confirmed compute-bound, not a
per-call-overhead artifact (32 separate per-segment calls vs. one batched
call differ by only ~15%, 248s vs 211s). Running that over the full
3,530-issue corpus would turn a per-phase measurement script into a
multi-hour job for no benefit — Phase 6's ablation grid re-measures
retrieval quality on the frozen golden set, not raw chunk stats, so this
only needs to be representative, not exhaustive. Measured against: all 129
documentation pages (full corpus — already includes `release-notes.md`,
the corpus's largest document), plus a fixed, seeded random sample of 50
closed issues, plus issue #2266 pinned in deliberately — the second corpus
outlier Phase 1 flagged as a concrete stress case for chunk-size decisions.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from aegis.chunking.factory import build_chunker
from aegis.chunking.segments import split_code_segments
from aegis.config import settings
from aegis.types import Document

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_PATH = settings.paths.data_processed / "documents.jsonl"
OUTPUT_PATH = REPO_ROOT / "docs" / "results" / "chunking-stats.md"

STRATEGIES = ["fixed", "recursive", "heading_aware", "semantic"]
ISSUE_SAMPLE_SIZE = 50
RANDOM_SEED = 20260729
SPOTLIGHT_IDS = ["doc::release-notes.md", "issue::2266"]


def load_documents() -> list[Document]:
    documents = []
    with DOCUMENTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            documents.append(Document(**json.loads(line)))
    return documents


def select_measurement_set(documents: list[Document]) -> list[Document]:
    by_id = {d.id: d for d in documents}
    docs = [d for d in documents if d.doc_type == "documentation"]
    issues = [d for d in documents if d.doc_type == "issue"]

    rng = random.Random(RANDOM_SEED)
    sampled_issues = rng.sample(issues, min(ISSUE_SAMPLE_SIZE, len(issues)))

    measurement_set = {d.id: d for d in docs + sampled_issues}
    for spotlight_id in SPOTLIGHT_IDS:
        if spotlight_id in by_id:
            measurement_set[spotlight_id] = by_id[spotlight_id]
    return list(measurement_set.values())


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    if lo == hi:
        return values[lo]
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo))


def code_integrity(document: Document, chunk_texts: list[str]) -> tuple[int, int]:
    code_blocks = [s.content for s in split_code_segments(document.text) if s.kind == "code"]
    if not code_blocks:
        return 0, 0
    intact = sum(1 for block in code_blocks if any(block in text for text in chunk_texts))
    return intact, len(code_blocks)


def measure_strategy(name: str, documents: list[Document]) -> tuple[dict, dict[str, dict]]:
    """Returns (aggregate stats, per-document stats for SPOTLIGHT_IDS only)
    — spotlight docs are measured as part of this same pass rather than
    re-chunked separately. Semantic chunking's per-sentence embedding cost
    means `release-notes.md` alone takes ~3.5 minutes (measured); reprocessing
    it a second time for a separate spotlight table would double that for no
    reason.
    """
    chunker = build_chunker(name)
    token_counts: list[int] = []
    heading_path_present = 0
    code_intact_total = 0
    code_blocks_total = 0
    chunk_total = 0
    spotlight: dict[str, dict] = {}

    start = time.time()
    for doc in documents:
        chunks = chunker.chunk(doc)
        chunk_total += len(chunks)
        token_counts.extend(c.token_count for c in chunks)
        heading_path_present += sum(1 for c in chunks if c.heading_path)
        intact, total = code_integrity(doc, [c.text for c in chunks])
        code_intact_total += intact
        code_blocks_total += total

        if doc.id in SPOTLIGHT_IDS:
            doc_token_counts = [c.token_count for c in chunks]
            spotlight[doc.id] = {
                "chunks": len(chunks),
                "token_max": max(doc_token_counts) if doc_token_counts else 0,
                "oversized_chunks": sum(1 for t in doc_token_counts if t > settings.chunking.max_tokens),
                "code_integrity_pct": round(100 * intact / total, 1) if total else None,
                "code_blocks_total": total,
            }
    elapsed = time.time() - start

    aggregate = {
        "strategy": name,
        "documents": len(documents),
        "chunks": chunk_total,
        "chunks_per_doc": round(chunk_total / len(documents), 2) if documents else 0,
        "token_p50": percentile(token_counts, 0.50),
        "token_p95": percentile(token_counts, 0.95),
        "token_max": max(token_counts) if token_counts else 0,
        "token_min": min(token_counts) if token_counts else 0,
        "chunks_with_heading_path_pct": round(100 * heading_path_present / chunk_total, 1) if chunk_total else 0.0,
        "code_integrity_pct": round(100 * code_intact_total / code_blocks_total, 1) if code_blocks_total else None,
        "code_blocks_total": code_blocks_total,
        "oversized_chunks": sum(1 for t in token_counts if t > settings.chunking.max_tokens),
        "wall_seconds": round(elapsed, 1),
    }
    return aggregate, spotlight


def render_markdown(
    results: list[dict],
    spotlight: dict[str, dict[str, dict]],
    n_docs: int,
    n_issues_sampled: int,
    n_issues_total: int,
) -> str:
    lines = [
        "# Chunking Strategy Comparison — Phase 2",
        "",
        f"Measured against {n_docs} documentation pages (full corpus) + "
        f"{n_issues_sampled} of {n_issues_total} closed issues "
        f"(seeded random sample, seed={RANDOM_SEED}, plus `release-notes.md` "
        "and issue #2266 pinned in explicitly — see spotlight table below). "
        "No winner is declared here; that needs the golden set (Phase 4) and "
        "the retrieval ablation grid (Phase 6).",
        "",
        f"`max_tokens={settings.chunking.max_tokens}`, `overlap_tokens={settings.chunking.overlap_tokens}` "
        "(shared across all four strategies, so this compares chunking logic, not size budgets).",
        "",
        "| Strategy | Chunks | Chunks/doc | Token p50 | Token p95 | Token max | Token min | "
        "Chunks w/ heading_path | Code-block integrity | Oversized chunks | Wall time |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        integrity = (
            f"{r['code_integrity_pct']}% ({r['code_blocks_total']} blocks)"
            if r["code_integrity_pct"] is not None
            else "n/a"
        )
        lines.append(
            f"| {r['strategy']} | {r['chunks']} | {r['chunks_per_doc']} | {r['token_p50']} | "
            f"{r['token_p95']} | {r['token_max']} | {r['token_min']} | "
            f"{r['chunks_with_heading_path_pct']}% | {integrity} | {r['oversized_chunks']} | {r['wall_seconds']}s |"
        )

    lines += [
        "",
        "## Stress-case spotlight",
        "",
        "The two corpus outliers Phase 1 flagged by hand: `release-notes.md` "
        "(686,041 chars, FastAPI's full changelog) and issue #2266 (193,590 "
        "chars, a user's pasted PyInstaller build log). Measured result: "
        "neither actually triggers the atomic-oversized-code-block exception "
        "at `max_tokens=400` — both split cleanly into many in-budget chunks. "
        "That contradicts the assumption these two would be the exception's "
        "concrete example; see the design record for what does trigger it "
        "(`doc::advanced/security/oauth2-scopes.md`, a 184-line example "
        "script repeated across 8 sections).",
        "",
        "| Document | Strategy | Chunks | Token max | Oversized chunks | Code-block integrity |",
        "|---|---|---|---|---|---|",
    ]
    for doc_id, per_strategy in spotlight.items():
        for strategy, s in per_strategy.items():
            integrity = f"{s['code_integrity_pct']}% ({s['code_blocks_total']} blocks)" if s["code_integrity_pct"] is not None else "n/a"
            lines.append(
                f"| {doc_id} | {strategy} | {s['chunks']} | {s['token_max']} | "
                f"{s['oversized_chunks']} | {integrity} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    all_documents = load_documents()
    measurement_set = select_measurement_set(all_documents)
    documents_by_id = {d.id: d for d in all_documents}

    n_docs = sum(1 for d in measurement_set if d.doc_type == "documentation")
    n_issues_sampled = sum(1 for d in measurement_set if d.doc_type == "issue")
    n_issues_total = sum(1 for d in all_documents if d.doc_type == "issue")

    results = []
    spotlight: dict[str, dict[str, dict]] = {doc_id: {} for doc_id in SPOTLIGHT_IDS if doc_id in documents_by_id}
    for strategy in STRATEGIES:
        print(f"Measuring {strategy} over {len(measurement_set)} documents...")
        aggregate, doc_spotlight = measure_strategy(strategy, measurement_set)
        results.append(aggregate)
        for doc_id, stats in doc_spotlight.items():
            spotlight[doc_id][strategy] = stats
        print(f"  done in {aggregate['wall_seconds']}s")

    markdown = render_markdown(results, spotlight, n_docs, n_issues_sampled, n_issues_total)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(markdown)
    print(f"\nWrote {OUTPUT_PATH}\n")
    print(markdown)


if __name__ == "__main__":
    main()
