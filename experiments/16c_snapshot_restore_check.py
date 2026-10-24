"""Phase 16, unit 16b: does a Qdrant snapshot of the live collection reproduce the
live index's retrieval metrics in a fresh, key-less Qdrant (design record 16b section 5)?

Setup, by hand (the experiment script only scores):
  1. POST /collections/aegis_chunks_bge_large/snapshots, download it, delete the server copy.
  2. docker run -d --rm --name ci-snap-test -p 127.0.0.1:6335:6333 qdrant/qdrant:v1.18.3
  3. POST localhost:6335/collections/aegis_chunks_bge_large/snapshots/upload?priority=snapshot -F snapshot=@<file>
Zero LLM cost.
"""
import json, numpy as np
from qdrant_client import QdrantClient
from aegis.config import settings
from aegis.embedding.encoder import Encoder
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.vector_store import get_client
from aegis.retrieval.dense import DenseIndex
from aegis.eval.harness import RetrievalConfig, run_eval
from aegis.types import Chunk, GoldenQuery
chunks=[Chunk(**json.loads(l)) for l in open(settings.paths.data_processed/"chunks.jsonl")]
qs=[q for q in (GoldenQuery(**json.loads(l)) for l in open(settings.paths.data_golden/"golden_set_v1.jsonl")) if not q.is_negative]
enc=Encoder(settings.embedding.bge_large_model, cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
QV=enc.encode([q.text for q in qs]); by={q.text:v for q,v in zip(qs,QV)}
class P:
    def encode(self,t): return np.stack([by[x] for x in t])
cfg=RetrievalConfig(name="x",chunking_strategy="heading_aware",contextual_enrichment=False,retrieval_strategy="dense",encoder="bge_large")
def run(client):
    idx=DenseIndex(chunks,"aegis_chunks_bge_large",P(),client=client); got={}
    def s(q):
        h=idx.search(q.text,top_k=10,exclude_doc_id=q.source_issue); got[q.id]=[c.id for c,_ in h]; return [(c.id,sc) for c,sc in h]
    return run_eval(cfg,qs,s,max_workers=1).overall, got
m_live,g_live=run(get_client())
m_snap,g_snap=run(QdrantClient(host="localhost",port=6335))
m_snap2,g_snap2=run(QdrantClient(host="localhost",port=6335))
same=sum(g_live[k]==g_snap[k] for k in g_live); same2=sum(g_snap[k]==g_snap2[k] for k in g_snap)
print("live  ",{k:round(v,4) for k,v in m_live.items()})
print("snap  ",{k:round(v,4) for k,v in m_snap.items()})
print("queries with identical ordered top-10 live vs snapshot:",same,"/",len(g_live))
print("snapshot vs snapshot (repeat):",same2,"/",len(g_snap))
