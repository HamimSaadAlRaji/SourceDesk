# RAG Course — Topics Learned

## Module 1: Introduction to RAG
- What RAG is & why LLMs need it
- The RAG loop & its advantages
- How LLMs work (tokens, autoregression)
- Why not put everything in the prompt
- Information retrieval basics

## Module 2: Information Retrieval Foundations
- Three retrieval techniques (keyword, semantic, metadata filtering)
- Metadata filtering
- Keyword search: bag of words → TF-IDF → BM25
- Semantic search & embeddings
- Distance metrics (Euclidean distance, cosine similarity, dot product)
- Contrastive training
- Hybrid search & Reciprocal Rank Fusion (RRF)
- Retrieval evaluation metrics (Precision, Recall, MAP@k, MRR)

## Module 3: Information Retrieval in Production
- KNN → ANN → HNSW
- Vector databases
- Chunking strategies (fixed, overlapping, recursive, structure-specific, semantic, LLM-based, context-aware)
- Query parsing (query rewriting, NER, HyDE)
- Bi-encoders, cross-encoders, and ColBERT
- Reranking

## Module 4: LLMs & Text Generation
- Transformer architecture (attention, feed-forward, encoder/decoder)
- Sampling strategies (greedy decoding, temperature, top-k, top-p, repetition penalty, logit bias)
- Choosing an LLM & reading benchmarks
- Prompt engineering (chat message format, system prompt, prompt templates)
- Advanced prompting & reasoning (in-context learning, Chain-of-Thought, reasoning models, context pruning)
- Handling hallucinations
- Evaluating the LLM's performance (RAGAS: Response Relevancy, Faithfulness)
- Agentic RAG (sequential, conditional, iterative, parallel workflows)
- RAG vs. fine-tuning

## Module 5: RAG Systems in Production
- What makes production challenging
- Evaluation strategies (system vs. component, code-based vs. LLM-judge vs. human feedback)
- Logging, monitoring & observability
- Customized evaluation
- Quantization (8-bit, 1-bit, Matryoshka embeddings)
- Cost vs. response quality
- Latency vs. response quality
- Security (knowledge-base leakage, tenant separation, LLM data leakage, database hacking, vector reconstruction)
- Multimodal RAG
