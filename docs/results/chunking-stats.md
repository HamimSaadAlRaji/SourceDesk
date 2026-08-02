# Chunking Strategy Comparison — Phase 2

Measured against 129 documentation pages (full corpus) + 51 of 3530 closed issues (seeded random sample, seed=20260729, plus `release-notes.md` and issue #2266 pinned in explicitly — see spotlight table below). No winner is declared here; that needs the golden set (Phase 4) and the retrieval ablation grid (Phase 6).

`max_tokens=400`, `overlap_tokens=50` (shared across all four strategies, so this compares chunking logic, not size budgets).

| Strategy | Chunks | Chunks/doc | Token p50 | Token p95 | Token max | Token min | Chunks w/ heading_path | Code-block integrity | Oversized chunks | Wall time |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 1646 | 9.14 | 400 | 400 | 400 | 8 | 0.0% | 84.1% (880 blocks) | 0 | 0.5s |
| recursive | 1791 | 9.95 | 377 | 399 | 400 | 8 | 0.0% | 85.3% (880 blocks) | 0 | 2.7s |
| heading_aware | 2682 | 14.9 | 168 | 390 | 2781 | 1 | 99.9% | 98.4% (880 blocks) | 60 | 1.3s |
| semantic | 1787 | 9.93 | 364 | 399 | 2781 | 8 | 0.0% | 99.1% (880 blocks) | 60 | 303.2s |

## Stress-case spotlight

The two corpus outliers Phase 1 flagged by hand: `release-notes.md` (686,041 chars, FastAPI's full changelog) and issue #2266 (193,590 chars, a user's pasted PyInstaller build log). Measured result: neither actually triggers the atomic-oversized-code-block exception at `max_tokens=400` — both split cleanly into many in-budget chunks. That contradicts the assumption these two would be the exception's concrete example; see the design record for what does trigger it (`doc::advanced/security/oauth2-scopes.md`, a 184-line example script repeated across 8 sections).

| Document | Strategy | Chunks | Token max | Oversized chunks | Code-block integrity |
|---|---|---|---|---|---|
| doc::release-notes.md | fixed | 589 | 400 | 0 | 80.6% (31 blocks) |
| doc::release-notes.md | recursive | 671 | 400 | 0 | 96.8% (31 blocks) |
| doc::release-notes.md | heading_aware | 1011 | 400 | 0 | 100.0% (31 blocks) |
| doc::release-notes.md | semantic | 676 | 400 | 0 | 100.0% (31 blocks) |
| issue::2266 | fixed | 166 | 400 | 0 | n/a |
| issue::2266 | recursive | 180 | 400 | 0 | n/a |
| issue::2266 | heading_aware | 180 | 400 | 0 | n/a |
| issue::2266 | semantic | 180 | 400 | 0 | n/a |
