"""Phase 15 unit 15d follow-up: does padding an injection with a real pasted
issue evade protectai/deberta-v3-base-prompt-injection-v2, and does scanning
in overlapping windows fix it? (Design record 15d §5-6.) Zero LLM calls, CPU.

Carrier: the first dev positive between 600 and 1,500 DeBERTa tokens (a real
issue body). Injections: the 7 dev golden adversarial queries, placed before
or after the carrier. Windowed = 400-token windows, stride 200, flag if any.

    uv run python experiments/15d_deberta_padding_evasion.py
"""

from __future__ import annotations

import json
import time

from transformers import AutoTokenizer, pipeline

from aegis.config import REPO_ROOT, settings
from aegis.guardrails.input_boundary import injection_flags

OUT = REPO_ROOT / "docs" / "results" / "deberta-padding-evasion-raw.json"
MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
WINDOW, STRIDE = 400, 200


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    clf = pipeline("text-classification", model=MODEL, tokenizer=tok, truncation=True, max_length=512, device="cpu")

    def windows(text: str) -> list[str]:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= WINDOW:
            return [text]
        return [tok.decode(ids[i:i + WINDOW]) for i in range(0, max(1, len(ids) - WINDOW + STRIDE), STRIDE)]

    single = lambda t: clf(t)[0]["label"] == "INJECTION"
    windowed = lambda t: any(clf(w)[0]["label"] == "INJECTION" for w in windows(t))

    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        dev = [json.loads(line) for line in f]
    carrier_q = next(q for q in dev if not q["is_negative"] and 600 < len(tok(q["text"])["input_ids"]) < 1500)
    carrier = carrier_q["text"]
    inj = [q["text"] for q in dev if q.get("negative_category") == "adversarial_injection"]
    out: dict = {"model": MODEL, "carrier_query_id": carrier_q["id"], "carrier_tokens": len(tok(carrier)["input_ids"]),
                 "carrier_alone_flagged": single(carrier), "window": WINDOW, "stride": STRIDE, "padding": {}}
    for where in ("start", "end"):
        texts = [f"{t}\n\n{carrier}" if where == "start" else f"{carrier}\n\n{t}" for t in inj]
        out["padding"][where] = {"deberta_single": f"{sum(map(single, texts))}/7",
                                 "deberta_windowed": f"{sum(map(windowed, texts))}/7",
                                 "regex": f"{sum(bool(injection_flags(t)) for t in texts)}/7"}
    positives = [q for q in dev if not q["is_negative"]]
    t0 = time.perf_counter()
    fp = [q["id"] for q in positives if windowed(q["text"])]
    out["windowed_dev_positive_fp"] = f"{len(fp)}/{len(positives)}"
    out["windowed_fp_ids"] = fp
    out["windowed_ms_per_query_mean"] = round((time.perf_counter() - t0) / len(positives) * 1000)
    out["windows_for_dev_positives"] = sum(len(windows(q["text"])) for q in positives)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
