# Generation bake-off (OpenAI) -- generated table

Prompt v2, temperature 0.0, max_tokens 400, 20 seeded dev positives (same as Phase 9 sweep). Analysis: `generation-bakeoff-openai.md`.

| generator | judge | mean valid_citation_rate | claim status counts |
|---|---|---|---|
| qwen/qwen3.8-27b | qwen | 0.7296 | exempt=10, missing_citation=22, supported_entailment=14, supported_lexical=59, unsupported=6 |
| gpt-4.1-mini | qwen | 0.4486 | exempt=19, missing_citation=47, supported_entailment=7, supported_lexical=49, unsupported=16 |
| gpt-4o-mini | qwen | 0.3458 | exempt=12, missing_citation=68, supported_entailment=5, supported_lexical=34, unsupported=7 |
| gpt-4.1-nano | qwen | 0.6665 | exempt=5, missing_citation=17, supported_entailment=7, supported_lexical=74, unsupported=26 |
| gpt-5-mini | qwen | 0.7158 | invalid_citation=6, missing_citation=6, supported_entailment=6, supported_lexical=35, unsupported=6 |
| gpt-5-nano | qwen | 0.4217 | missing_citation=36, supported_entailment=7, supported_lexical=29, unsupported=7 |
| qwen/qwen3.8-27b | nano | 0.6329 | exempt=10, missing_citation=22, supported_entailment=8, supported_lexical=59, unsupported=12 |
| gpt-4.1-mini | nano | 0.4361 | exempt=19, missing_citation=47, supported_entailment=6, supported_lexical=49, unsupported=17 |
| gpt-4o-mini | nano | 0.3264 | exempt=12, missing_citation=68, supported_entailment=4, supported_lexical=34, unsupported=8 |
| gpt-4.1-nano | nano | 0.6288 | exempt=5, missing_citation=17, supported_entailment=4, supported_lexical=74, unsupported=29 |
| gpt-5-mini | nano | 0.6242 | invalid_citation=6, missing_citation=6, supported_entailment=3, supported_lexical=35, unsupported=9 |
| gpt-5-nano | nano | 0.3717 | missing_citation=36, supported_entailment=3, supported_lexical=29, unsupported=11 |

| generator | mean TTFT ms | p50 total ms | max total ms | mean output tokens | answers at cap | total cost USD | cost/answer USD |
|---|---|---|---|---|---|---|---|
| gpt-4.1-mini | 1060 | 2690 | 3907 | 260 | 3 | 0.035297 | 0.001765 |
| gpt-4o-mini | 799 | 2937 | 3906 | 186 | 0 | 0.012359 | 0.000618 |
| gpt-4.1-nano | 746 | 1884 | 3466 | 242 | 1 | 0.008684 | 0.000434 |
| gpt-5-mini | 1137 | 2426 | 4486 | 184 | 2 | 0.023465 | 0.001173 |
| gpt-5-nano | 881 | 2116 | 3362 | 249 | 7 | 0.005217 | 0.000261 |
