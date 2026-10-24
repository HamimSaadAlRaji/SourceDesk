"""Phase 16, unit 16b: why does a rebuilt HNSW index disagree with the live one
(design record 16b section 5)? Tests three explanations against exact cosine:
duplicate-vector ties, the `exclude_doc_id` filter, and plain ANN inaccuracy.
Result: ties cover 4% of exact top-10 hits and the filter is not the cause
(Qdrant exact=True with the filter agrees 98.7% with numpy exact); the graph
itself returns ~67% of the true top-10. Zero LLM cost; writes and deletes one
throwaway collection (`_tie_check`).
"""
import json, numpy as np
from aegis.config import settings
from aegis.embedding.encoder import Encoder
from aegis.embedding.cache import EmbeddingCache
from aegis.embedding.vector_store import get_client, DENSE_VECTOR_NAME
from aegis.eval.ci_index import restore_index
from qdrant_client.models import Filter, FieldCondition, MatchValue
from aegis.types import Chunk, GoldenQuery

chunks=[Chunk(**json.loads(l)) for l in open(settings.paths.data_processed/"chunks.jsonl")]
V=np.load(settings.paths.data_processed/"vectors_bge_large.npy").astype(np.float32)
qs=[q for q in (GoldenQuery(**json.loads(l)) for l in open(settings.paths.data_golden/"golden_set_v1.jsonl")) if not q.is_negative]
enc=Encoder(settings.embedding.bge_large_model, cache=EmbeddingCache(settings.paths.embedding_cache), device="cpu")
QV=enc.encode([q.text for q in qs]).astype(np.float32)
ids=np.array([c.id for c in chunks]); docs=np.array([c.doc_id for c in chunks])

# duplicate groups by rounded vector
keys=[v.round(5).tobytes() for v in V]
from collections import Counter
cnt=Counter(keys); dup_of=np.array([cnt[k] for k in keys])
print("chunks in a duplicate group (size>1):", int((dup_of>1).sum()), "of", len(V))

client=get_client(); NAME="_tie_check"
restore_index(client, NAME, chunks, V)
def q_hits(qv, src, filt=True, exact=False):
    from qdrant_client.models import SearchParams
    f=Filter(must_not=[FieldCondition(key="doc_id",match=MatchValue(value=src))]) if filt else None
    r=client.query_points(NAME,query=qv.tolist(),using=DENSE_VECTOR_NAME,limit=10,query_filter=f,
        search_params=SearchParams(exact=exact), with_payload=["chunk_id"]).points
    return [(p.payload["chunk_id"],p.score) for p in r]
res={"idov":[], "score_rec":[], "tied_in_top10":[], "exact_qdrant_vs_np":[], "nofilter_idov":[]}
pos={c:i for i,c in enumerate(ids)}
for q,qv in zip(qs,QV):
    s=V@qv; s[docs==q.source_issue]=-np.inf
    top=np.argsort(-s)[:10]; kth=s[top[-1]]
    ex_ids={ids[i] for i in top}
    h=q_hits(qv,q.source_issue)
    res["idov"].append(len({c for c,_ in h}&ex_ids)/10)
    res["score_rec"].append(np.mean([ (s[pos[c]]>=kth-1e-5) for c,_ in h]))
    res["tied_in_top10"].append(np.mean([dup_of[i]>1 for i in top]))
    he=q_hits(qv,q.source_issue,exact=True)
    res["exact_qdrant_vs_np"].append(len({c for c,_ in he}&ex_ids)/10)
    s2=V@qv; t2=np.argsort(-s2)[:10]; e2={ids[i] for i in t2}
    res["nofilter_idov"].append(len({c for c,_ in q_hits(qv,"",filt=False)}&e2)/10)
for k,v in res.items(): print(f"{k:20s} mean={np.mean(v):.4f}")
client.delete_collection(NAME)
