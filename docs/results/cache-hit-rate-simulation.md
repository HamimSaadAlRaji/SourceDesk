# Cache Hit-Rate Simulation — Zipfian Traffic (Phase 12 Unit 3)

Up to 9000 simulated requests per skew value, drawn over the 150 real dev-positive golden intents (`golden_set_v1.jsonl`), real `ExactMatchCache`/`SemanticCache` classes (threshold=0.88), real `bge-large` embeddings (165 real encode calls total, memoized by exact text). Zero LLM calls -- cache-miss cost bootstrapped from 4 real (latency_ms, completion_tokens) samples (`docs/results/qwen3-8-27b-reasoning-check-raw.json`) plus real dense-retrieval latency (6.54 ms, `docs/results/ann-recall-vs-latency.md`).

No real production traffic log exists for this system (never deployed) -- the Zipf skew is a disclosed modeling assumption, swept rather than asserted. Only 15/150 intents have a real known paraphrase (Unit 2's hand-written set), so the semantic-tier hit rate here is a conservative lower bound, not an estimate of the full population.

## Hit rate vs. traffic volume (requests per unique intent), by skew

Cumulative hit rate at each checkpoint within one real request stream per skew -- reported this way, not just at one final N, because a first real run showed traffic volume (N over only 150 real unique intents) drives the exact-hit rate far more than skew shape does once N is large relative to the intent pool; see 'what surprised me' in the design record.

| skew | N | requests/intent | hit rate (total) | exact | semantic | latency reduction | token reduction |
|---|---|---|---|---|---|---|---|
| 0.8 | 150 | 1.0 | 56.7% | 56.0% | 0.7% | 52.3% | 56.3% |
| 0.8 | 450 | 3.0 | 72.0% | 70.9% | 1.1% | 70.7% | 72.4% |
| 0.8 | 1500 | 10.0 | 89.7% | 88.3% | 1.4% | 89.0% | 89.8% |
| 0.8 | 4500 | 30.0 | 96.5% | 95.0% | 1.6% | 96.3% | 96.6% |
| 0.8 | 9000 | 60.0 | 98.3% | 96.7% | 1.5% | 98.2% | 98.3% |
| 1.2 | 150 | 1.0 | 69.3% | 69.3% | 0.0% | 71.3% | 69.6% |
| 1.2 | 450 | 3.0 | 80.0% | 79.8% | 0.2% | 80.1% | 79.8% |
| 1.2 | 1500 | 10.0 | 91.2% | 91.0% | 0.2% | 91.0% | 91.0% |
| 1.2 | 4500 | 30.0 | 96.6% | 96.1% | 0.5% | 96.4% | 96.4% |
| 1.2 | 9000 | 60.0 | 98.3% | 97.5% | 0.8% | 98.2% | 98.2% |
| 1.6 | 150 | 1.0 | 86.7% | 86.7% | 0.0% | 86.2% | 87.2% |
| 1.6 | 450 | 3.0 | 89.6% | 89.6% | 0.0% | 89.8% | 89.9% |
| 1.6 | 1500 | 10.0 | 94.6% | 94.5% | 0.1% | 94.6% | 94.7% |
| 1.6 | 4500 | 30.0 | 97.3% | 97.0% | 0.2% | 97.3% | 97.3% |
| 1.6 | 9000 | 60.0 | 98.3% | 98.1% | 0.3% | 98.5% | 98.4% |

## Detail at final checkpoint per skew

### skew = 0.8

- Tier counts (of 9000): exact=8706, semantic=138, miss=156
- Real mean exact-hit latency: 0.014 ms
- Real mean semantic-hit latency: 0.796 ms
- Total latency: 428.1s with caching vs. 23367.5s without (98.2% reduction)
- Total completion tokens: 28047 with caching vs. 1627563 without (98.3% reduction)

### skew = 1.2

- Tier counts (of 9000): exact=8774, semantic=70, miss=156
- Real mean exact-hit latency: 0.014 ms
- Real mean semantic-hit latency: 0.791 ms
- Total latency: 414.8s with caching vs. 23301.1s without (98.2% reduction)
- Total completion tokens: 29047 with caching vs. 1624527 without (98.2% reduction)

### skew = 1.6

- Tier counts (of 9000): exact=8827, semantic=24, miss=149
- Real mean exact-hit latency: 0.015 ms
- Real mean semantic-hit latency: 0.635 ms
- Total latency: 352.0s with caching vs. 23207.2s without (98.5% reduction)
- Total completion tokens: 26684 with caching vs. 1624335 without (98.4% reduction)

