# Worked audit-trail reconstruction

Each section below was rebuilt **from the stored Langfuse trace alone** (`fetch_observations` -> `build_audit_trail` ->
`resolve_evidence`), then checked field-by-field against what the live pipeline actually returned, and joined to the
trace-ID-tagged log lines. Source: `experiments/13j_audit_trail_demo.py`. Real queries, real Groq, real Qdrant; nothing edited by hand.

---

## Case: answered

# Audit trail — trace `2e0f275ee71fcd5f88674ae8f32b6fac`

## Request

- Query (`golden::issue::604`): 'Query parameters with hyphens\n\n**Description**\r\n\r\nHow can I get a query parameter that contains hyphens?'
- Cache: **miss**
- Route: **docs_howto**
- Outcome: **answered**

## Retrieved (before the token budget)

| rank | chunk | score |
|---|---|---|
| 1 | `issue::4842::heading_aware::8` | 0.8381 |
| 2 | `issue::2635::heading_aware::0` | 0.756 |
| 3 | `doc::tutorial/query-params-str-validations.md::heading_aware::10` | 0.7293 |
| 4 | `issue::2040::heading_aware::1` | 0.7268 |
| 5 | `doc::tutorial/query-param-models.md::heading_aware::2` | 0.7262 |
| 6 | `issue::2635::heading_aware::1` | 0.7246 |
| 7 | `issue::3656::heading_aware::4` | 0.7192 |
| 8 | `issue::2971::heading_aware::3` | 0.7162 |
| 9 | `issue::2971::heading_aware::2` | 0.7127 |
| 10 | `issue::3217::heading_aware::2` | 0.712 |

## Confidence

- Score: **0.7953**
- reranker_top_score: raw 0.9563 → normalized 0.9624
- score_gap: raw 0.7542 → normalized 1.0000
- semantic_agreement: raw 0.7368 → normalized 0.2665
- distribution_shape: raw 9.0263 → normalized 0.2915

## What the model was shown

- Prompt version: `v2`; params: {'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 1000, 'budget_tokens': 4096}
- Context: 10 chunks, 1047 tokens, 0 dropped by the budget

### [1] `issue::4842::heading_aware::8` (CITED, source text: OK)

_422 error when using a Pydantic model for query parameters with aliases > Additional Context_

> This only happens when one of the query parameter names contains a hyphen, the `item-name` parameter.

### [3] `doc::tutorial/query-params-str-validations.md::heading_aware::10` (not cited, source text: OK)

_Query Parameters and String Validations { #query-parameters-and-string-validations } > Add regular expressions { #add-regular-expressions }_

> You can define a <dfn title="A regular expression, regex or regexp is a sequence of characters that define a search pattern for strings.">regular expression</dfn> `pattern` that the parameter should match:
> 
> 
> 
> ```python
> from typing import Annotated
> 
> from fastapi import FastAPI, Query
> 
> app = FastAPI()
> 
> 
> @app.get("/items/")
> async def read_items(
>     q: Annotated[
>         str | None, Query(min_length=3, max_length=50, pattern="^fixedquery$")
>     ] = None,
> ):
>     results = {"items": [{"item_id": "Foo…

### [5] `doc::tutorial/query-param-models.md::heading_aware::2` (not cited, source text: OK)

_Query Parameter Models { #query-parameter-models } > Check the Docs { #check-the-docs }_

> You can see the query parameters in the docs UI at `/docs`:

### [7] `issue::3656::heading_aware::4` (not cited, source text: OK)

_Allow custom type for Query parameters > Wanted Solution_

> It would be nice to have a custom validator (data converter) for `Query` (and possibly `Path`, `Header`, `Cookie`, etc, which input is always `str`). And then support custom types.

### [9] `issue::2971::heading_aware::2` (not cited, source text: OK)

_"Query Parameters and String Validations" highlight lines mismatch > Example_

> As is: https://fastapi.tiangolo.com/ja/tutorial/query-params-str-validations/#_9 (First one)
> To be: https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#declare-more-metadata

### [10] `issue::3217::heading_aware::2` (not cited, source text: OK)

_Typo/Bad wording in docs "FastAPI Path Parameters and Numeric Validations" page > Description_

> I believe there is a typo in the docs [here](https://fastapi.tiangolo.com/tutorial/path-params-numeric-validations/)! In the info box under the recap section, it states the following:
> 
> > Query, Path, and others you will see later are subclasses of a common Param class (that you don't need to use).
> And all of them share the same all these same parameters of additional validation and metadata you have seen.
> 
> Seems a bit hard to read, no?
> 
> **Resolution (@tiangolo):**
> 
> Thanks for the help here …

### [8] `issue::2971::heading_aware::3` (not cited, source text: OK)

_"Query Parameters and String Validations" highlight lines mismatch > Description_

> As is: In fastapi/docs/ja/docs/tutorial/query-params-str-validations.md
> `hl_lines="9"`
> 
> To be: `hl_lines="10"`
> like fastapi/docs/en/docs/tutorial/query-params-str-validations.md

### [6] `issue::2635::heading_aware::1` (not cited, source text: OK)

_request query_param parse error when + in value > **right click can open the picture show**_

> request.query_param.get('a')  got 'b c' instead of 'b+c'
> 
> when query_param.get('key') it '+' was replaced by ' '
> 
> **Resolution (@tiangolo):**
> 
> Thanks for the help here @ycd and @Mause ! 👏  🙇 
> 
> Thanks for reporting back and closing the issue @cst4049  👍
> 
> > Sorry for the long delay! 🙈  I wanted to personally address each issue/PR and they piled up through time, but now I'm checking each one in order.

### [4] `issue::2040::heading_aware::1` (not cited, source text: OK)

_String validation in the query parameters to accept the values based on predefined values > Description_

> I want to use query parameters to pass a string value to the GET request, but it can't be any string and should be one of the predefined ones. Assume that we want to pass one of the `rating`, `published_at`, and `alphabetical` using the query parameter `sorted_by`, so if anything else rather than these three has been sent to the function, we don't want to accept it. 
> 
> I found [Query Parameters and String Validation](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/) section o…

### [2] `issue::2635::heading_aware::0` (not cited, source text: OK)

_request query_param parse error when + in value > First check_

> http://localhost:8000/test?a=b+c&q=search
> query_param like this
> {"a": "b c","q":"search"}
> this parse might be a bug
> 
> ![image](http://user-images.githubusercontent.com/16145670/104309801-aec4c900-550d-11eb-8194-4b4213167986.png)
> 
> 
> ![image](http://user-images.githubusercontent.com/16145670/104309725-9359be00-550d-11eb-966c-9d7fc12de460.png)

## Answer served

To retrieve a query parameter that contains hyphens, you must use the `alias` argument in the `Query` definition to map the hyphenated URL parameter name to a valid Python variable name [1].

The provided context does not contain specific code examples or further details on how to configure the `alias` for this scenario, so I cannot provide a complete code snippet based solely on the available information [1].

- Provider: groq
- Validated citations: ['issue::4842::heading_aware::8'] (2 claims checked)

## Timings (span durations, ms)

- cache_lookup: 2
- routing: 573
- retrieval: 62
- confidence: 322
- generation: 641
- validation: 295

## Integrity notes

- All expected stages present; per-stage spans agree with the recorded result.


### Reconstruction vs. the live pipeline result

- **Identical** on every audited field (cache status, route, outcome, answer text, citations, provider, prompt version, retrieved ids+scores, context map incl. content hashes, confidence).

### Log lines for this trace (2)

```json
{"ts": "2026-09-30T14:46:57.429298+00:00", "level": "INFO", "logger": "aegis.pipeline", "event": "pipeline.request_start", "trace_id": "2e0f275ee71fcd5f88674ae8f32b6fac", "query_id": "golden::issue::604"}
{"ts": "2026-09-30T14:46:59.331910+00:00", "level": "INFO", "logger": "aegis.pipeline", "event": "pipeline.request_end", "trace_id": "2e0f275ee71fcd5f88674ae8f32b6fac", "query_id": "golden::issue::604", "cache_status": "miss", "escalate": false, "route": "docs_howto", "provider": "groq", "stage_latencies_ms": {"cache_lookup": 1.5, "routing": 573.0, "retrieval": 62.4, "confidence": 322.4, "generation": 641.9, "validation": 295.3}}
```

---

## Case: hard-route escalation

# Audit trail — trace `d7b35a7fa4538b2ebccd1f3c0a7738c8`

## Request

- Query (`negative::billing_legal::4`): "Can you send me an invoice for last quarter's support contract?"
- Cache: **miss**
- Route: **billing_account_legal**
- Outcome: **escalated to a human — billing_account_legal always hard-escalates to a human, regardless of query phrasing**

## Answer served

This request has been escalated to a human agent.

## Timings (span durations, ms)

- cache_lookup: 2
- routing: 212
- escalation: 1

## Integrity notes

- All expected stages present; per-stage spans agree with the recorded result.


### Reconstruction vs. the live pipeline result

- **Identical** on every audited field (cache status, route, outcome, answer text, citations, provider, prompt version, retrieved ids+scores, context map incl. content hashes, confidence).

### Log lines for this trace (3)

```json
{"ts": "2026-09-30T14:46:59.335225+00:00", "level": "INFO", "logger": "aegis.pipeline", "event": "pipeline.request_start", "trace_id": "d7b35a7fa4538b2ebccd1f3c0a7738c8", "query_id": "negative::billing_legal::4"}
{"ts": "2026-09-30T14:46:59.549594+00:00", "level": "WARNING", "logger": "aegis.pipeline", "event": "pipeline.escalated", "trace_id": "d7b35a7fa4538b2ebccd1f3c0a7738c8", "query_id": "negative::billing_legal::4", "route": "billing_account_legal", "reason": "billing_account_legal always hard-escalates to a human, regardless of query phrasing", "after_retrieval": false}
{"ts": "2026-09-30T14:46:59.550243+00:00", "level": "INFO", "logger": "aegis.pipeline", "event": "pipeline.request_end", "trace_id": "d7b35a7fa4538b2ebccd1f3c0a7738c8", "query_id": "negative::billing_legal::4", "cache_status": "miss", "escalate": true, "route": "billing_account_legal", "provider": null, "stage_latencies_ms": {"cache_lookup": 2.2, "routing": 211.9, "escalation": 0.7}}
```

---

## Case: confidence escalation

# Audit trail — trace `dc8a9055322c0e123c347522942f5f51`

## Request

- Query (`golden::issue::2689`): 'execution optiomization, redundant validation.\n\nHello!\r\n\r\nI\'m noticed a little problem with `response_model` feature:\r\n\r\nCode sample:\r\n\r\n```\r\n@router.patch("/settings/swipes", response_model=schemas.UserSwipes, status_code=200)\r\ndef update_users_settings_preferences(\r\n        user_update: schemas.UserSwipes,\r\n        response: Response = Response,\r\n        token: str = Header(...),\r\n        db: Session = Depends(get_db)):\r\n    try:\r\n        ...\r\n\r\nclass UserSwipes(BaseModel):  # Consider UserProfile attributes as low boundary\r\n    smoking_low: Optional[int] = Field(..., ge=0, le=4, description="Minimal boundary smoking", example=2)\r\n    smoking_high: Optional[int] = Field(..., ge=0, le=4, description="Maximum boundary smoking", example=3)\r\n    alcohol_low: Optional[int] = Field(..., ge=0, le=4, description="Minimal boundary smoking", example=2)\r\n    alcohol_high: Optional[int] = Field(..., ge=0, le=4, description="Maximum boundary smoking", example=3)\r\n\r\n    @root_validator\r\n    def ensure_bigger(cls, values: dict[str]) -> dict[str]:\r\n        old_k_v = list(values.items())[0]\r\n        for k, v in values.items():\r\n            if \'low\' in k:\r\n                old_k_v = k, v\r\n            else:\r\n                if k.split(\'_\')[0] == old_k_v[0].split(\'_\')[0]:\r\n                    if old_k_v[-1] > v:\r\n                        raise Exception(\'Minimal value biggest that Maximum item!\')\r\n                else:\r\n                    raise Exception("\'high\' item must follow \'low\' item.")\r\n        return values\r\n```\r\n\r\nThis `@root_validator` executes twice because of `respomse_model` and `user_update: schemas.UserSwipes`.\r\nBut obviously you don\'t need to execute a  validation a second time.'
- Cache: **miss**
- Route: **known_issue**
- Outcome: **escalated to a human — confidence 0.2295 < tau 0.5000**

## Retrieved (before the token budget)

| rank | chunk | score |
|---|---|---|
| 1 | `doc::release-notes.md::heading_aware::961` | 0.7812 |
| 2 | `issue::2073::heading_aware::1` | 0.772 |
| 3 | `issue::4578::heading_aware::3` | 0.7708 |
| 4 | `issue::475::heading_aware::0` | 0.7689 |
| 5 | `issue::2182::heading_aware::0` | 0.7665 |
| 6 | `issue::988::heading_aware::0` | 0.7654 |
| 7 | `issue::482::heading_aware::0` | 0.7625 |
| 8 | `issue::1468::heading_aware::3` | 0.762 |
| 9 | `issue::421::heading_aware::0` | 0.7593 |
| 10 | `issue::305::heading_aware::0` | 0.759 |

## Confidence

- Score: **0.2295**
- reranker_top_score: raw 0.2450 → normalized 0.2465
- score_gap: raw 0.0600 → normalized 0.0864
- semantic_agreement: raw 0.7568 → normalized 0.3227
- distribution_shape: raw 3.5596 → normalized 0.0820

## Answer served

This request has been escalated to a human agent.

## Timings (span durations, ms)

- cache_lookup: 1
- routing: 190
- retrieval: 18
- confidence: 1108
- escalation: 1

## Integrity notes

- All expected stages present; per-stage spans agree with the recorded result.


### Reconstruction vs. the live pipeline result

- **Identical** on every audited field (cache status, route, outcome, answer text, citations, provider, prompt version, retrieved ids+scores, context map incl. content hashes, confidence).

### Log lines for this trace (3)

```json
{"ts": "2026-09-30T14:46:59.550602+00:00", "level": "INFO", "logger": "aegis.pipeline", "event": "pipeline.request_start", "trace_id": "dc8a9055322c0e123c347522942f5f51", "query_id": "golden::issue::2689"}
{"ts": "2026-09-30T14:47:00.867750+00:00", "level": "WARNING", "logger": "aegis.pipeline", "event": "pipeline.escalated", "trace_id": "dc8a9055322c0e123c347522942f5f51", "query_id": "golden::issue::2689", "route": "known_issue", "reason": "confidence 0.2295 < tau 0.5000", "after_retrieval": true}
{"ts": "2026-09-30T14:47:00.868181+00:00", "level": "INFO", "logger": "aegis.pipeline", "event": "pipeline.request_end", "trace_id": "dc8a9055322c0e123c347522942f5f51", "query_id": "golden::issue::2689", "cache_status": "miss", "escalate": true, "route": "known_issue", "provider": null, "stage_latencies_ms": {"cache_lookup": 0.8, "routing": 190.3, "retrieval": 17.8, "confidence": 1108.1, "escalation": 0.4}}
```
