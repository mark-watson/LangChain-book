# Chapter 14 — RAG with reranking

Two scripts showing the same retriever with and without a cross-encoder reranker post-processor.

## Setup

```console
$ uv sync
$ ollama pull qwen3.5:4b
```

The reranker model (`BAAI/bge-reranker-base`, ~275 MB) downloads on first run.

## Scripts

| Script | What it shows |
|---|---|
| `01_no_reranker.py` | Retrieve top 5 with cosine similarity, hand straight to the query engine. |
| `02_with_reranker.py` | Retrieve top 10, rerank to top 3 with a cross-encoder, then hand to the query engine. |

Both scripts ask the same question against the same corpus so the effect of the reranker is visible in the retrieved node scores.
