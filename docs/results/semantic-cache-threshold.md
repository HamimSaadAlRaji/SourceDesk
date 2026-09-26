# Semantic Cache Threshold — Real Sweep (Phase 12 Unit 2)

Zero LLM cost. Two real, measured populations against `bge-large` (the same encoder used for retrieval): the 15 highest-similarity pairs among genuinely *different* real dev-positive golden queries (real negatives -- the actual hardest near-miss cases this corpus produces), and 15 hand-written paraphrases of real golden queries (positives -- a real, disclosed limitation: one person's paraphrases, not a sample of real user phrasing variance).

## The real negatives (highest-similarity distinct-question pairs)

| similarity | query A | query B |
|---|---|---|
| 0.8694 | Generated OpenAPI schema does not reflect pydantic dynamic models.  ## | OpenAPI schema generation sometimes fails  **Describe the bug** FastA |
| 0.8457 | OpenApi docs for subclass of list of pydantic BaseModel  ```Python fr | Support OpenAPI example[s] attribute  Hello everyone. Today I tried t |
| 0.8385 | BrokenResourceError  ### First Check  - [X] I added a very descripti | Upgrading from fastapi 0.60.1 -> 0.63.0 and uvicorn 0.11.6 -> 0.13.2 w |
| 0.8341 | Tags metadata is missing from docs  ### First check  * [x] I added a | Upgrading from fastapi 0.60.1 -> 0.63.0 and uvicorn 0.11.6 -> 0.13.2 w |
| 0.8326 | OpenAPI schema generation sometimes fails  **Describe the bug** FastA | Support OpenAPI example[s] attribute  Hello everyone. Today I tried t |
| 0.8279 | Generated OpenAPI schema does not reflect pydantic dynamic models.  ## | Support OpenAPI example[s] attribute  Hello everyone. Today I tried t |
| 0.8245 | How is the request body processed with pydantic?  I'm trying to unders | One or more objects in request body?  **Description**  I'd like to h |
| 0.8242 | performance latency after updating to Fastapi > 0.82.0  ### First Chec | Performance regression of resolving dependencies  ### First Check  - [ |
| 0.8142 | Double validation pydantic model when used response_model  ### Example | Support OpenAPI example[s] attribute  Hello everyone. Today I tried t |
| 0.8137 | Yield dependency not cleaning up immediately after the request  ### Fi | performance latency after updating to Fastapi > 0.82.0  ### First Chec |

## Threshold sweep

| threshold | false positives (of 15 real negatives) | true positives (of 15 real paraphrases) |
|---|---|---|
| 0.75 | 15/15 | 13/15 |
| 0.80 | 15/15 | 13/15 |
| 0.85 | 1/15 | 11/15 |
| 0.88 | 0/15 | 9/15 |
| 0.90 | 0/15 | 7/15 |
| 0.92 | 0/15 | 5/15 |
| 0.94 | 0/15 | 4/15 |
| 0.96 | 0/15 | 0/15 |

## Chosen threshold: 0.88

Lowest threshold in the sweep with **zero** false positives on the real hardest-negative set (0/15), catching 9/15 real paraphrases. Same asymmetric-cost logic as Phase 10's tau: a false positive here serves a confidently wrong cached answer (this project's own North Star failure mode), a false negative just costs one extra real pipeline call -- so the threshold is chosen to eliminate the former on real measured data, not to maximize the latter.
