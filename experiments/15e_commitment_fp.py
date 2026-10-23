"""Phase 15 unit 15e: false-positive rate of the output-side commitment check
on real answers this system generated in earlier phases (design record 15e).
Zero LLM calls: re-reads answers already saved in docs/results/.

Plus two harder probes, because the real answers contain no business terms
at all (a vacuous zero): every sentence of the 19,236-chunk corpus (answers
mostly restate it), and data/security/commitment_probes_v1.jsonl -- technical
FastAPI sentences about payments that must NOT fire.

Sources: Phase 9 sampling sweep, Phase 9 provider bake-off, Phase 9 prompt
v1-vs-v2 (Groq), Phase 11 judge batch (rubric fix). Answers are de-duplicated
by exact text, split into sentences exactly as the citation validator does,
and every flagged sentence is written out for inspection.

    uv run python experiments/15e_commitment_fp.py
"""

from __future__ import annotations

import json

from aegis.config import REPO_ROOT
from aegis.guardrails.citations import _iter_claim_sentences
from aegis.factory import load_chunks
from aegis.guardrails.commitments import _TERM, is_unauthorized_commitment

RESULTS = REPO_ROOT / "docs" / "results"
OUT = RESULTS / "commitment-fp-raw.json"


def _answers() -> dict[str, tuple[str, str]]:
    """answer text -> (source file, query_id)."""
    found: dict[str, tuple[str, str]] = {}

    def add(fname: str, rows, get):
        for r in rows:
            text, qid = get(r)
            if text:
                found.setdefault(text, (fname, qid))

    d = json.loads((RESULTS / "sampling-sweep-raw.json").read_text())
    add("sampling-sweep-raw.json", d["raw"], lambda r: (r.get("answer_text"), r["query_id"]))
    d = json.loads((RESULTS / "llm-bakeoff-raw.json").read_text())
    add("llm-bakeoff-raw.json", d["raw"], lambda r: (r.get("answer_text"), r["query_id"]))
    d = json.loads((RESULTS / "prompt-v1-vs-v2-groq-raw.json").read_text())
    add("prompt-v1-vs-v2-groq-raw.json", d, lambda r: (r.get("answer_text"), r["query_id"]))
    d = json.loads((RESULTS / "judge-agreement-batch-raw-rubric-fix.json").read_text())
    add("judge-agreement-batch-raw-rubric-fix.json", d,
        lambda r: ((r.get("generation") or {}).get("answer_text"), r["query_id"]))
    return found


def main() -> None:
    answers = _answers()
    n_sent = 0
    flagged = []
    for text, (src, qid) in answers.items():
        for sentence in _iter_claim_sentences(text):
            n_sent += 1
            if is_unauthorized_commitment(sentence):
                flagged.append({"source": src, "query_id": qid, "sentence": sentence})
    term_sents = sum(1 for t in answers for s in _iter_claim_sentences(t) if _TERM.search(s))
    corpus_n = corpus_term = 0
    corpus_flagged = []
    for ch in load_chunks():
        for s in _iter_claim_sentences(ch.text):
            corpus_n += 1
            corpus_term += bool(_TERM.search(s))
            if is_unauthorized_commitment(s):
                corpus_flagged.append({"chunk_id": ch.id, "sentence": s})
    with open(REPO_ROOT / "data" / "security" / "commitment_probes_v1.jsonl", encoding="utf-8") as f:
        probes = [json.loads(line) for line in f]
    probe_fp = [p["text"] for p in probes if is_unauthorized_commitment(p["text"]) != p["expected_fire"]]
    out = {"real_answers": {"n_answers": len(answers), "n_sentences": n_sent, "sentences_with_business_term": term_sents,
                            "n_flagged": len(flagged), "flagged": flagged},
           "corpus": {"n_sentences": corpus_n, "sentences_with_business_term": corpus_term,
                      "n_flagged": len(corpus_flagged), "flagged": corpus_flagged[:50]},
           "technical_payment_probes": {"n": len(probes), "wrong": len(probe_fp), "wrong_texts": probe_fp}}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
