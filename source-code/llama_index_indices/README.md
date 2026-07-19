# Chapter 13 — Choosing an index type

Three scripts covering the LlamaIndex index types beyond `VectorStoreIndex`, plus a hybrid retriever that combines dense (embedding) and sparse (BM25) retrieval.

## Setup

```console
$ uv sync
$ ollama pull qwen3.5:4b
```

## Scripts

| Script | What it shows |
|---|---|
| `01_summary_index.py` | `SummaryIndex` — every query touches every Node. Right choice when the question is about the whole corpus, not a specific passage. |
| `02_keyword_table.py` | `SimpleKeywordTableIndex` — keyword-match retrieval, no embeddings. Fast, no ML dependency, useful when the corpus is dominated by specific terms. |
| `03_hybrid_fusion.py` | `QueryFusionRetriever` — combines BM25 and vector retrieval with reciprocal rank fusion. The practical hybrid pattern. |

All three read from `../data/`.

Run any one with:

```console
$ uv run 01_summary_index.py
```
