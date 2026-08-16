# Routing confusion matrix

**Phase:** 8 · **Model:** `llama-3.1-8b-instant` (few-shot, JSON-mode, temperature=0.0)

58 labeled queries: 25 `how_to` dev positives -> `docs_howto`, 25 `error_debugging` dev positives -> `known_issue`, 8 `billing_legal` negatives (of 10 total -- the other 2 are sealed in the Phase 4 holdout split) -> `billing_account_legal`. See the experiment script's docstring for the labeling-methodology decision.

## Overall accuracy: 0.6897 (40/58)

## Confusion matrix (rows = true route, columns = predicted route)

| true \ predicted | docs_howto | known_issue | billing_account_legal |
|---|---|---|---|
| **docs_howto** | 20 | 4 | 1 |
| **known_issue** | 11 | 12 | 2 |
| **billing_account_legal** | 0 | 0 | 8 |

## Per-class precision / recall

| Route | Precision | Recall | TP | FN | FP |
|---|---|---|---|---|---|
| docs_howto | 0.6452 | 0.8000 | 20 | 5 | 11 |
| known_issue | 0.7500 | 0.4800 | 12 | 13 | 4 |
| billing_account_legal | 0.7273 | 1.0000 | 8 | 0 | 3 |
