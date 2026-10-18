# Sampling Sweep — Temperature x top_p vs. Faithfulness Proxy

Phase 9, step 5. 20 sampled dev-positive golden queries, Phase 6's winning retrieval config (dense, heading-aware, bge-large, no enrichment), prompt v1, Groq `qwen/qwen3.8-27b` only (single-provider, to isolate the sampling-parameter effect from provider choice — that's step 6).

Faithfulness proxy: `CitationValidator.valid_citation_rate` — fraction of citation-requiring claims that end up supported (lexically or via entailment). Not RAGAS faithfulness (Phase 11); named as a proxy deliberately.

| temperature | top_p | mean valid_citation_rate | mean latency (ms) | claim status counts |
|---|---|---|---|---|
| 0.0 | 0.5 | 0.7296 | 23847 | exempt=10, missing_citation=22, supported_entailment=14, supported_lexical=59, unsupported=6 |
| 0.0 | 1.0 | 0.7296 | 26411 | exempt=10, missing_citation=22, supported_entailment=14, supported_lexical=59, unsupported=6 |
| 0.5 | 1.0 | 0.7350 | 25402 | exempt=7, missing_citation=13, supported_entailment=13, supported_lexical=55, unsupported=11 |
| 1.0 | 1.0 | 0.6546 | 25293 | exempt=3, missing_citation=14, supported_entailment=8, supported_lexical=41, unsupported=7 |
