"""Phase 15 step 2 (design record 15g4): router re-evaluation on the Phase 8
routing set (58 labeled dev queries, `08a_router_eval.py`'s labels, imported).

Arms: gpt-4.1-nano (served default since 15g), gpt-4.1-mini, and
qwen/qwen3.8-27b on Groq. Phase 8's 0.6897 was measured on
`llama-3.1-8b-instant` (since retired); the Groq model that replaced it was
never scored on this set, so it gets a free arm here as the like-for-like
pre-migration baseline.

Same system prompt, temperature, JSON mode and 50-token cap as served. Two
differences from served behaviour, both so the measurement can't be
flattered:
  - unparseable output is counted separately. Served, it falls back to
    billing_account_legal (fail safe), which would inflate billing recall.
  - API errors are retried on 429 and otherwise abort the run; served, they
    raise RouterUnavailableError -> escalation.

Raw output is written per arm; the Phase 8 files are left untouched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import Counter

from aegis.config import REPO_ROOT
from aegis.generation.llm.chat import LLMCallError, chat_completion
from aegis.router.classifier import SYSTEM_PROMPT, Route
from aegis.config import settings

_spec = importlib.util.spec_from_file_location("router_eval", REPO_ROOT / "experiments" / "08a_router_eval.py")
router_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(router_eval)

ARMS = {"gpt-4.1-nano": "openai", "gpt-4.1-mini": "openai", "qwen/qwen3.8-27b": "groq"}
GROQ_PAUSE_S = 2.5
RESULTS_RAW = REPO_ROOT / "docs" / "results" / "router-reeval-openai-raw.json"
RESULTS_MD = REPO_ROOT / "docs" / "results" / "router-reeval-openai.md"
ROUTES = [r.value for r in Route]


def call(provider: str, model: str, text: str) -> str:
    for attempt in range(5):
        if provider == "groq":
            time.sleep(GROQ_PAUSE_S)
        try:
            return chat_completion(provider, model,
                                   [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
                                   temperature=settings.router.temperature, max_tokens=settings.router.max_tokens,
                                   purpose="router_eval", json_mode=True)
        except LLMCallError as exc:
            if exc.status_code == 429 and attempt < 4:
                time.sleep(20 * (attempt + 1))
                continue
            sys.exit(f"{model}: API error {exc.__cause__!r} -- aborting, not scoring a partial arm")
    raise AssertionError("unreachable")


def parse(raw: str) -> str | None:
    try:
        return Route(json.loads(raw)["route"]).value
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def score(rows: list[dict]) -> dict:
    confusion = {t: Counter() for t in ROUTES}
    for r in rows:
        confusion[r["true"]][r["pred"] or "UNPARSEABLE"] += 1
    per_class = {}
    for c in ROUTES:
        tp = confusion[c][c]
        support = sum(confusion[c].values())
        predicted = sum(confusion[t][c] for t in ROUTES)
        per_class[c] = {"tp": tp, "support": support, "predicted": predicted,
                        "recall": tp / support, "precision": tp / predicted if predicted else float("nan")}
    correct = sum(confusion[c][c] for c in ROUTES)
    return {"accuracy": correct / len(rows), "correct": correct, "n": len(rows),
            "unparseable": sum(r["pred"] is None for r in rows),
            "confusion": {k: dict(v) for k, v in confusion.items()}, "per_class": per_class}


def main() -> None:
    labeled = router_eval.load_labeled_queries()
    data = json.loads(RESULTS_RAW.read_text()) if RESULTS_RAW.exists() else {"rows": {}, "scores": {}}
    for model, provider in ARMS.items():
        rows = data["rows"].setdefault(model, [])
        done = {r["query_id"] for r in rows}
        for q, true in labeled:
            if q.id in done:
                continue
            t0 = time.perf_counter()
            raw = call(provider, model, q.text)
            rows.append({"query_id": q.id, "true": true.value, "pred": parse(raw), "raw": raw[:200],
                         "latency_ms": (time.perf_counter() - t0) * 1000})
            RESULTS_RAW.write_text(json.dumps(data, indent=2))
        data["scores"][model] = score(rows)
        s = data["scores"][model]
        print(f"{model}: accuracy {s['accuracy']:.4f} ({s['correct']}/{s['n']}), unparseable {s['unparseable']}, "
              + ", ".join(f"{c} R={p['recall']:.3f} P={p['precision']:.3f}" for c, p in s["per_class"].items()))
    RESULTS_RAW.write_text(json.dumps(data, indent=2))
    write_md(data)


def write_md(data: dict) -> None:
    cols = ROUTES + ["UNPARSEABLE"]
    lines = ["# Router re-evaluation (Phase 15 step 2) -- generated", "",
             "58 labeled dev queries (Phase 8 labels, `08a_router_eval.load_labeled_queries`), served router prompt, "
             "temperature 0, JSON mode, max_tokens 50. Phase 8 baseline (llama-3.1-8b-instant, retired): 0.6897 (40/58), "
             "billing recall 8/8. Analysis: design record 15g4.", ""]
    for model, s in data["scores"].items():
        lines += [f"## {model}: accuracy {s['accuracy']:.4f} ({s['correct']}/{s['n']}), unparseable {s['unparseable']}", "",
                  "| true \\ predicted | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for t in ROUTES:
            lines.append(f"| **{t}** | " + " | ".join(str(s["confusion"][t].get(c, 0)) for c in cols) + " |")
        lines += ["", "| Route | Precision | Recall | TP | support |", "|---|---|---|---|---|"]
        for c, p in s["per_class"].items():
            lines.append(f"| {c} | {p['precision']:.4f} | {p['recall']:.4f} | {p['tp']} | {p['support']} |")
        lines.append("")
    RESULTS_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
