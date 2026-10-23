"""Design record 15l: false-positive / detection measurement for the risky-advice
check (Open Question #20). Zero LLM calls.

  - frozen probes (data/security/risky_advice_probes_v1.jsonl): 12 risky, 12 legit
  - 15h H1: did it fire on the obeyed poisoned install claims?
  - every stored generated answer in this project (bake-off, prompt-fit, 15h): sentence-level fire rate
  - documentation chunks: a correct answer quoting the docs must not fire
"""

from __future__ import annotations

import hashlib
import json
import re

from aegis.config import REPO_ROOT, settings
from aegis.guardrails.citations import _iter_claim_sentences
from aegis.guardrails.risky_advice import documentation_vouched, risky_reasons

R = REPO_ROOT / "docs" / "results"
OUT = R / "risky-advice-fp-raw.json"
PROBES = REPO_ROOT / "data" / "security" / "risky_advice_probes_v1.jsonl"


def answers() -> dict[str, list[str]]:
    bake = json.loads((R / "generation-bakeoff-openai-raw.json").read_text())
    fit = json.loads((R / "prompt-fit-openai-raw.json").read_text())
    real = json.loads((R / "security-realstack-raw.json").read_text())
    out = {f"bakeoff:{m}": [g["text"] for g in gens.values()] for m, gens in bake["generations"].items()}
    for prompt, by_model in fit["generations"]["dev"].items():
        for m, gens in by_model.items():
            out[f"promptfit:{prompt}:{m}"] = [g["text"] for g in gens.values()]
    out["15h:h2:v3"] = [g["text"] for g in real["h2"]["generations"].values()]
    out["15h:h3"] = [r["text"] for r in real["h3"].values() if not r["escalate"]]
    out["15h:h1"] = [r["text"] for r in real["h1"].values()]
    return out


def main() -> None:
    vouched = documentation_vouched()
    result: dict = {"probes_sha256": hashlib.sha256(PROBES.read_bytes()).hexdigest()}

    probes = [json.loads(line) for line in PROBES.read_text().splitlines() if line.strip()]
    rows = [{"id": p["id"], "expected": p["expected_fire"], "reasons": risky_reasons(p["text"], vouched)} for p in probes]
    result["probes"] = {"risky_caught": sum(bool(r["reasons"]) for r in rows if r["expected"]),
                        "risky_total": sum(r["expected"] for r in rows),
                        "legit_fired": sum(bool(r["reasons"]) for r in rows if not r["expected"]),
                        "legit_total": sum(not r["expected"] for r in rows), "rows": rows}

    real = json.loads((R / "security-realstack-raw.json").read_text())
    obeyed_install = [c["sentence"] for r in real["h1"].values() if r["poison"] == "poison::malicious_package"
                      for c in r["claims"] if re.search("fastapi-securefix", c["sentence"], re.I)]
    result["h1_securefix_sentences"] = [{"sentence": s, "reasons": risky_reasons(s, vouched)} for s in obeyed_install]

    fired: list[dict] = []
    total = 0
    for name, texts in answers().items():
        if name == "15h:h1":
            continue  # contains the attack payloads; reported above
        for text in texts:
            for sentence in _iter_claim_sentences(text):
                total += 1
                if reasons := risky_reasons(sentence, vouched):
                    fired.append({"set": name, "sentence": sentence, "reasons": reasons})
    result["real_answers"] = {"sentences": total, "fired": len(fired), "rows": fired}

    doc_total = doc_fired = 0
    doc_rows = []
    with open(settings.paths.data_processed / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["doc_type"] != "documentation":
                continue
            for sentence in _iter_claim_sentences(c["text"]):
                doc_total += 1
                if reasons := risky_reasons(sentence, vouched):
                    doc_fired += 1
                    doc_rows.append({"chunk": c["id"], "sentence": sentence[:300], "reasons": reasons})
    result["documentation"] = {"sentences": doc_total, "fired": doc_fired, "rows": doc_rows}

    OUT.write_text(json.dumps(result, indent=2))
    p = result["probes"]
    print(f"probes: risky {p['risky_caught']}/{p['risky_total']}, legit fired {p['legit_fired']}/{p['legit_total']}")
    print(f"15h securefix sentences fired: {sum(bool(r['reasons']) for r in result['h1_securefix_sentences'])}/{len(obeyed_install)}")
    print(f"real answers: {len(fired)}/{total} sentences fired")
    print(f"documentation: {doc_fired}/{doc_total} sentences fired")


if __name__ == "__main__":
    main()
