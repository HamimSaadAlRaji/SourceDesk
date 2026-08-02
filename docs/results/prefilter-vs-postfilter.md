# Pre-filter vs. Post-filter Metadata Filtering — Phase 3

Encoder: `BAAI/bge-small-en-v1.5`. For each `product_area` category, 30 queries sampled from within that category (or all available members if fewer). Ground truth: exact top-k *within the category* via brute-force KNN restricted to that category's vectors. "Post-filter" = the common naive implementation — search top-10 unfiltered, then discard results that don't match the category. "Pre-filter" = Qdrant's native `query_filter`.

| Category | Corpus size | Post-filter recall | Post-filter short of k on | Pre-filter recall |
|---|---|---|---|---|
| general | 5081 | 0.4033 | 90% of queries | 1.0000 |
| cors | 300 | 0.1667 | 100% of queries | 1.0000 |
| background-tasks | 68 | 0.1767 | 100% of queries | 1.0000 |
| how-to | 42 | 0.2900 | 100% of queries | 1.0000 |
| graphql | 56 | 0.3000 | 100% of queries | 1.0000 |

## Conclusion

Naive post-filtering loses 60-83% of recall depending on category, **even for
`general`, the single largest category at 26% of the corpus** — a top-10
unfiltered search still comes back short of 10 category-matching results on
90% of queries, because the corpus's other 74% crowds out the pool. Native
pre-filtering is perfect (1.0000) across every category tested, narrow or
broad, because Qdrant searches *within* the filtered subset directly rather
than filtering after the fact. Decision: retrieval-layer metadata filters
(`product_area`, `doc_type`, `doc_version`) always use `query_filter`, never
a post-hoc discard step — confirmed necessary here, not assumed.
