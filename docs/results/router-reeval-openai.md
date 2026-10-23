# Router re-evaluation (Phase 15 step 2) -- generated

58 labeled dev queries (Phase 8 labels, `08a_router_eval.load_labeled_queries`), served router prompt, temperature 0, JSON mode, max_tokens 50. Phase 8 baseline (llama-3.1-8b-instant, retired): 0.6897 (40/58), billing recall 8/8. Analysis: design record 15g4.

## gpt-4.1-nano: accuracy 0.6379 (37/58), unparseable 0

| true \ predicted | docs_howto | known_issue | billing_account_legal | UNPARSEABLE |
|---|---|---|---|---|
| **docs_howto** | 24 | 1 | 0 | 0 |
| **known_issue** | 20 | 5 | 0 | 0 |
| **billing_account_legal** | 0 | 0 | 8 | 0 |

| Route | Precision | Recall | TP | support |
|---|---|---|---|---|
| docs_howto | 0.5455 | 0.9600 | 24 | 25 |
| known_issue | 0.8333 | 0.2000 | 5 | 25 |
| billing_account_legal | 1.0000 | 1.0000 | 8 | 8 |

## gpt-4.1-mini: accuracy 0.7759 (45/58), unparseable 0

| true \ predicted | docs_howto | known_issue | billing_account_legal | UNPARSEABLE |
|---|---|---|---|---|
| **docs_howto** | 21 | 4 | 0 | 0 |
| **known_issue** | 9 | 16 | 0 | 0 |
| **billing_account_legal** | 0 | 0 | 8 | 0 |

| Route | Precision | Recall | TP | support |
|---|---|---|---|---|
| docs_howto | 0.7000 | 0.8400 | 21 | 25 |
| known_issue | 0.8000 | 0.6400 | 16 | 25 |
| billing_account_legal | 1.0000 | 1.0000 | 8 | 8 |

## qwen/qwen3.8-27b: accuracy 0.7931 (46/58), unparseable 0

| true \ predicted | docs_howto | known_issue | billing_account_legal | UNPARSEABLE |
|---|---|---|---|---|
| **docs_howto** | 21 | 4 | 0 | 0 |
| **known_issue** | 8 | 17 | 0 | 0 |
| **billing_account_legal** | 0 | 0 | 8 | 0 |

| Route | Precision | Recall | TP | support |
|---|---|---|---|---|
| docs_howto | 0.7241 | 0.8400 | 21 | 25 |
| known_issue | 0.8095 | 0.6800 | 17 | 25 |
| billing_account_legal | 1.0000 | 1.0000 | 8 | 8 |
