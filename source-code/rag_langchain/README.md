# Chapter 4 — RAG patterns with LangChain

Four progressively-better retrieval patterns over the same tiny corpus (`../data/`), all running on your laptop with local embeddings, a local vector store, and a local LLM.

## Setup

```console
$ uv sync
$ ollama pull qwen3.5:4b
```

The first script also downloads two small models from the Hugging Face Hub the first time it runs (an embedding model and a cross-encoder reranker), roughly 350 MB total. They are cached under `~/.cache/huggingface/` after that.

## Corpus

The four scripts share `../data/` — four short text files on chemistry, economics, health, and sports. Small enough that you can eyeball whether the retriever picked the right paragraph, big enough that the different retrieval strategies actually behave differently.

## Scripts

| Script | Pattern |
|---|---|
| `01_naive_rag.py` | Dense-embedding vector search into a prompt into a model. The baseline. |
| `02_reranked_rag.py` | Retrieve top 10 with the vector store, then rerank to top 3 with a cross-encoder. |
| `03_hybrid_rag.py` | BM25 keyword search + dense embeddings combined with `EnsembleRetriever`. |
| `04_multi_query_rag.py` | Let an LLM rewrite the query several ways and union the retrievals. |

Run any one with:

```console
$ uv run 01_naive_rag.py
```

Each script asks the same three questions so you can compare answers across patterns.
