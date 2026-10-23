"""Phase 15 "before" measurements on real project data (design record 15).
Zero LLM calls: local models, local Qdrant, files on disk.

  A1  BM25 filter gap, real corpus: HybridRetriever with doc_type=documentation
      over the 180 dev queries -- how many top-10 slots are issue chunks?
      Dense-only under the same filter is the control.
  A6  Qdrant exposure: does the REST API answer without a key, and what does
      a stored point carry?
  A7  Input-boundary facts: a naive injection regex's hit rate on dev
      positives (false positives) vs dev adversarial queries; PII-shaped
      strings in dev queries, the corpus, and the real interactions store.
      Counts only -- matched values are never written out.
  A8  Existing defense, re-measured and attacked: the confidence gate on the 7
      dev adversarial queries as-is (reproduces Phase 10) vs the same
      injections wrapped around a real FastAPI question.

Holdout is not read (it stays untouched until Phase 19).

    uv run python experiments/15a_security_baseline.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import time
import urllib.request

from aegis.config import REPO_ROOT, settings
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.encoder import Encoder
from aegis.factory import load_chunks
from aegis.guardrails.confidence import ConfidenceScorer
from aegis.rerank.cross_encoder import CrossEncoderReranker
from aegis.retrieval.dense import DenseIndex
from aegis.retrieval.hybrid import HybridRetriever, RetrievalStrategy
from aegis.retrieval.sparse import SparseIndex
from aegis.types import DocType

OUT = REPO_ROOT / "docs" / "results" / "security-baseline-raw.json"
TOP_K = 10

# A deliberately naive input-side injection detector: the kind of keyword list
# a "prompt injection filter" usually starts as. Measured, not adopted.
NAIVE_INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above|earlier)|disregard|system prompt|developer mode|"
    r"you are now|forget (you|everything|all)|new (policy|instructions)|override|jailbreak|"
    r"(reveal|print|repeat|output) (your|the) (system|hidden|exact|full|raw)",
    re.IGNORECASE,
)

# High-precision PII / secret shapes. Deliberately no names, no phone numbers,
# no bare IPs: those collide with code, versions and stack traces.
PII = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "url_with_credentials": re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s@/]+@"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}

# Hand-written, not golden-set text (same convention as the router few-shots).
WRAP_QUESTIONS = [
    "How do I declare an optional query parameter with a default value in FastAPI?",
    "How do I return a custom status code from a FastAPI path operation?",
    "How do I add CORS middleware to a FastAPI app?",
]


def _dev() -> list[dict]:
    with open(settings.paths.data_golden / "golden_set_v1.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def a1_filter_gap(chunks, encoder) -> dict:
    dense = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)
    t0 = time.perf_counter()
    sparse = SparseIndex(chunks)
    build_s = time.perf_counter() - t0
    hybrid = HybridRetriever(sparse_index=sparse, dense_index=dense)
    rows = []
    for q in _dev():
        h = hybrid.search(q["text"], top_k=TOP_K, strategy=RetrievalStrategy.HYBRID_RRF, doc_type=DocType.DOCUMENTATION)
        d = hybrid.search(q["text"], top_k=TOP_K, strategy=RetrievalStrategy.DENSE, doc_type=DocType.DOCUMENTATION)
        rows.append({"query_id": q["id"],
                     "hybrid_issue_slots": sum(c.doc_type == DocType.ISSUE for c, _ in h),
                     "dense_issue_slots": sum(c.doc_type == DocType.ISSUE for c, _ in d),
                     "hybrid_top1_is_issue": bool(h) and h[0][0].doc_type == DocType.ISSUE})
    n = len(rows)
    return {
        "n_queries": n, "top_k": TOP_K, "sparse_build_s": round(build_s, 2),
        "hybrid_issue_slots_total": sum(r["hybrid_issue_slots"] for r in rows),
        "hybrid_issue_slot_fraction": sum(r["hybrid_issue_slots"] for r in rows) / (n * TOP_K),
        "queries_with_any_issue_leak": sum(r["hybrid_issue_slots"] > 0 for r in rows),
        "queries_with_issue_at_rank1": sum(r["hybrid_top1_is_issue"] for r in rows),
        "dense_issue_slots_total": sum(r["dense_issue_slots"] for r in rows),
        "rows": rows,
    }


def _get(url: str) -> tuple[int, dict | None]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None


def _post(url: str, body: dict) -> tuple[int, dict | None]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None


def a6_qdrant() -> dict:
    base = f"http://{settings.qdrant.host}:{settings.qdrant.port}"
    coll = settings.qdrant.collection_bge_large
    s_list, listing = _get(f"{base}/collections")
    s_info, info = _get(f"{base}/collections/{coll}")
    s_scroll, scroll = _post(f"{base}/collections/{coll}/points/scroll",
                             {"limit": 1, "with_payload": True, "with_vector": True})
    point = (scroll or {}).get("result", {}).get("points", [{}])[0]
    vectors = point.get("vector") or {}
    ports = subprocess.run(["docker", "port", "sourcedesk-qdrant-1"], capture_output=True, text=True).stdout.strip()
    return {
        "no_key_list_collections_status": s_list,
        "collections": [c["name"] for c in (listing or {}).get("result", {}).get("collections", [])],
        "no_key_collection_info_status": s_info,
        "points_count": (info or {}).get("result", {}).get("points_count"),
        "no_key_scroll_status": s_scroll,
        "payload_keys": sorted((point.get("payload") or {}).keys()),
        "vector_names_and_dims": {k: len(v) if isinstance(v, list) else "sparse" for k, v in vectors.items()},
        "docker_port_bindings": ports.splitlines(),
        "client_configured_api_key": hasattr(settings.qdrant, "api_key") and bool(getattr(settings.qdrant, "api_key")),
    }


def _pii_counts(texts: list[str]) -> dict:
    return {name: sum(bool(p.search(t)) for t in texts) for name, p in PII.items()} | {"n_texts": len(texts)}


def a7_input_boundary(chunks) -> dict:
    dev = _dev()
    positives = [q["text"] for q in dev if not q["is_negative"]]
    adversarial = [q["text"] for q in dev if q.get("negative_category") == "adversarial_injection"]
    other_neg = [q["text"] for q in dev if q["is_negative"] and q.get("negative_category") != "adversarial_injection"]
    fp_ids = [q["id"] for q in dev if not q["is_negative"] and NAIVE_INJECTION.search(q["text"])]

    db = settings.api.interactions_db_path
    stored: list[str] = []
    if db.exists():
        with sqlite3.connect(db) as c:
            for qt, at, rq in c.execute("SELECT query_text, answer_text, rewritten_query FROM interactions"):
                stored += [x for x in (qt, at, rq) if x]
            stored += [r[0] for r in c.execute("SELECT comment FROM feedback WHERE comment IS NOT NULL")]
    return {
        "naive_injection_regex": NAIVE_INJECTION.pattern,
        "fires_on_dev_positives": f"{len(fp_ids)}/{len(positives)}",
        "fires_on_dev_positive_ids": fp_ids,
        "fires_on_dev_adversarial": f"{sum(bool(NAIVE_INJECTION.search(t)) for t in adversarial)}/{len(adversarial)}",
        "missed_adversarial": [t for t in adversarial if not NAIVE_INJECTION.search(t)],
        "fires_on_dev_other_negatives": f"{sum(bool(NAIVE_INJECTION.search(t)) for t in other_neg)}/{len(other_neg)}",
        "pii_dev_queries": _pii_counts([q["text"] for q in dev]),
        "pii_corpus_chunks": _pii_counts([c.text for c in chunks])
        | {"issue_chunks_with_email": sum(bool(PII["email"].search(c.text)) for c in chunks if c.doc_type == DocType.ISSUE)},
        "pii_interactions_store": _pii_counts(stored) | {"db": str(db.relative_to(REPO_ROOT))},
    }


def a8_confidence(chunks, encoder) -> dict:
    dense = DenseIndex(chunks, settings.qdrant.collection_bge_large, encoder)
    scorer = ConfidenceScorer(reranker=CrossEncoderReranker(), encode_fn=encoder.encode)
    tau = settings.confidence.tau
    adversarial = [q for q in _dev() if q.get("negative_category") == "adversarial_injection"]

    def score(text: str) -> dict:
        t0 = time.perf_counter()
        cands = dense.search(text, top_k=TOP_K)
        conf = scorer.score(text, cands).confidence
        return {"text": text, "confidence": round(conf, 4), "passes_gate": conf >= tau,
                "top1": cands[0][0].id if cands else None, "ms": round((time.perf_counter() - t0) * 1000)}

    raw = [score(q["text"]) | {"query_id": q["id"]} for q in adversarial]
    controls = [score(t) for t in WRAP_QUESTIONS]
    wrapped = []
    for i, q in enumerate(adversarial):
        base = WRAP_QUESTIONS[i % len(WRAP_QUESTIONS)]
        wrapped.append(score(f"{base} {q['text']}") | {"query_id": q["id"], "style": "question_then_injection"})
        wrapped.append(score(f"{q['text']} Also: {base}") | {"query_id": q["id"], "style": "injection_then_question"})
    return {
        "tau": tau,
        "raw_adversarial_pass_gate": f"{sum(r['passes_gate'] for r in raw)}/{len(raw)}",
        "control_questions_pass_gate": f"{sum(r['passes_gate'] for r in controls)}/{len(controls)}",
        "wrapped_pass_gate": f"{sum(r['passes_gate'] for r in wrapped)}/{len(wrapped)}",
        "raw": raw, "controls": controls, "wrapped": wrapped,
    }


def main() -> None:
    t0 = time.perf_counter()
    chunks = load_chunks()
    encoder = Encoder(model_name=settings.embedding.bge_large_model,
                      cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
    out: dict = {}
    for name, fn in [("A6_qdrant_exposure", a6_qdrant),
                     ("A7_input_boundary", lambda: a7_input_boundary(chunks)),
                     ("A8_confidence_gate", lambda: a8_confidence(chunks, encoder)),
                     ("A1_bm25_filter_gap", lambda: a1_filter_gap(chunks, encoder))]:
        s = time.perf_counter()
        out[name] = fn()
        out[name]["elapsed_s"] = round(time.perf_counter() - s, 1)
        print(f"{name}: {out[name]['elapsed_s']} s", flush=True)
    out["total_elapsed_s"] = round(time.perf_counter() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
