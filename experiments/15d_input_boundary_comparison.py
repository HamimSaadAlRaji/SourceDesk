"""Phase 15 unit 15d: hand-rolled input boundary vs the libraries it was chosen
over (design record 15d). Zero LLM calls; CPU only.

Injection detection -- hand-rolled flag regex (`aegis.guardrails.input_boundary`,
frozen before this ran) vs protectai/deberta-v3-base-prompt-injection-v2:
  false positives on the 150 dev positives and 23 non-injection dev negatives;
  detection on the 7 dev golden injections, 14 wrapped, 14 paraphrased
  (data/security/attacks_v1.jsonl).
PII -- hand-rolled `redact` vs Presidio AnalyzerEngine (default config,
en_core_web_lg): per-kind detection on data/security/pii_probes_v1.jsonl, and what
each flags on the 180 dev queries (where real PII is ~3 queries, design record 15 A7).

The libraries are NOT project dependencies. Run them in a throwaway overlay:

    uv run --with presidio-analyzer --with sentencepiece \
      --with "en_core_web_lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl" \
      python experiments/15d_input_boundary_comparison.py [--limit N]

Without them it measures only the hand-rolled side.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter

from aegis.config import REPO_ROOT, settings
from aegis.guardrails.input_boundary import injection_flags, redact

OUT = REPO_ROOT / "docs" / "results" / "input-boundary-comparison-raw.json"
SEC = REPO_ROOT / "data" / "security"
DEBERTA = "protectai/deberta-v3-base-prompt-injection-v2"


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _sets(limit: int | None) -> dict[str, list[str]]:
    dev = _jsonl(settings.paths.data_golden / "golden_set_v1.jsonl")
    by_id = {q["id"]: q for q in dev}
    attacks = _jsonl(SEC / "attacks_v1.jsonl")
    sets = {
        "dev_positives": [q["text"] for q in dev if not q["is_negative"]],
        "dev_other_negatives": [q["text"] for q in dev if q["is_negative"] and q.get("negative_category") != "adversarial_injection"],
        "golden_injections": [by_id[a["golden_id"]]["text"] for a in attacks if a["variant"] == "golden_reference"],
        "wrapped": [a["text"] for a in attacks if a["variant"] == "wrapped"],
        "paraphrase": [a["text"] for a in attacks if a["variant"] == "paraphrase"],
    }
    return {k: v[:limit] for k, v in sets.items()} if limit else sets


def _rate(flags: list[bool]) -> str:
    return f"{sum(flags)}/{len(flags)}"


def injection(sets, limit) -> dict:
    out: dict = {"regex": {}, "deberta": None}
    for name, texts in sets.items():
        out["regex"][name] = _rate([bool(injection_flags(t)) for t in texts])
    out["regex_missed_paraphrases"] = [t for t in sets["paraphrase"] if not injection_flags(t)]
    try:
        from transformers import AutoTokenizer, pipeline
    except ImportError:
        return out
    tok = AutoTokenizer.from_pretrained(DEBERTA)
    clf = pipeline("text-classification", model=DEBERTA, tokenizer=tok, truncation=True, max_length=512, device="cpu")
    clf("warm up")
    res, lat, over = {}, [], 0
    missed = []
    for name, texts in sets.items():
        flags = []
        for t in texts:
            over += len(tok(t)["input_ids"]) > 512
            t0 = time.perf_counter()
            r = clf(t)[0]
            lat.append((time.perf_counter() - t0) * 1000)
            flags.append(r["label"] == "INJECTION")
            if name == "paraphrase" and r["label"] != "INJECTION":
                missed.append(t)
        res[name] = _rate(flags)
    out["deberta"] = res | {
        "model": DEBERTA, "missed_paraphrases": missed, "inputs_truncated_at_512_tokens": over,
        "latency_ms_mean": round(statistics.mean(lat), 1), "latency_ms_p95": round(sorted(lat)[int(0.95 * len(lat)) - 1], 1),
        "n_calls": len(lat),
    }
    return out


def pii(limit) -> dict:
    probes = _jsonl(SEC / "pii_probes_v1.jsonl")
    dev = [q["text"] for q in _jsonl(settings.paths.data_golden / "golden_set_v1.jsonl")][:limit or None]
    regex_probe = [{"id": p["id"], "expected": p["expected"], "found": sorted(redact(p["text"])[1])} for p in probes]
    regex_dev = Counter(k for t in dev for k in redact(t)[1])
    out = {"regex": {"probes": regex_probe,
                     "dev_queries_with_any_redaction": sum(bool(redact(t)[1]) for t in dev),
                     "dev_redactions_by_kind": dict(regex_dev), "n_dev": len(dev)},
           "presidio": None}
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError:
        return out
    engine = AnalyzerEngine()
    engine.analyze("warm up", language="en")
    lat, dev_types, dev_types_05, dev_any, dev_any_05, examples = [], Counter(), Counter(), 0, 0, Counter()
    for t in dev:
        t0 = time.perf_counter()
        found = engine.analyze(t, language="en")
        lat.append((time.perf_counter() - t0) * 1000)
        dev_types.update(r.entity_type for r in found)
        hi = [r for r in found if r.score >= 0.5]
        dev_types_05.update(r.entity_type for r in hi)
        dev_any += bool(found)
        dev_any_05 += bool(hi)
        for r in hi:
            if r.entity_type in ("PERSON", "LOCATION", "NRP", "URL", "DATE_TIME"):
                examples[(r.entity_type, t[r.start:r.end][:40])] += 1
    probe = []
    for p in probes:
        found = engine.analyze(p["text"], language="en")
        probe.append({"id": p["id"], "expected": p["expected"],
                      "found": sorted({f"{r.entity_type}@{r.score:.2f}" for r in found})})
    out["presidio"] = {
        "probes": probe, "n_dev": len(dev),
        "dev_queries_with_any_entity": dev_any, "dev_queries_with_entity_score_ge_0_5": dev_any_05,
        "dev_entities_by_type": dict(dev_types), "dev_entities_by_type_score_ge_0_5": dict(dev_types_05),
        "most_common_name_like_hits_ge_0_5": [f"{k[0]}: {k[1]!r} x{v}" for k, v in examples.most_common(25)],
        "latency_ms_mean": round(statistics.mean(lat), 1), "latency_ms_p95": round(sorted(lat)[int(0.95 * len(lat)) - 1], 1),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="per-set cap, for a timing benchmark")
    args = ap.parse_args()
    t0 = time.perf_counter()
    sets = _sets(args.limit)
    out = {
        "attacks_v1_sha256": hashlib.sha256((SEC / "attacks_v1.jsonl").read_bytes()).hexdigest(),
        "pii_probes_v1_sha256": hashlib.sha256((SEC / "pii_probes_v1.jsonl").read_bytes()).hexdigest(),
        "limit": args.limit,
        "injection": injection(sets, args.limit),
        "pii": pii(args.limit),
    }
    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    if args.limit is None:
        OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    print(json.dumps({k: v for k, v in out.items() if k not in ("pii",)}, indent=1, ensure_ascii=False)[:3000])
    print("elapsed", out["elapsed_s"])


if __name__ == "__main__":
    main()
