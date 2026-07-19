# Chapter 17 — Multi-index query pipelines

Two scripts that route a query across multiple corpora: `RouterQueryEngine` (pick one) and `SubQuestionQueryEngine` (decompose into per-corpus subquestions).

## Setup

```console
$ uv sync
$ ollama pull qwen3.5:4b
```

Both scripts split `../data/*.txt` into per-topic mini-corpora (chemistry, economics, health, sports) and build one index per corpus.

## Scripts

| Script | What it shows |
|---|---|
| `01_router_query_engine.py` | LLM picks the one most relevant index for each query and forwards to it. |
| `02_subquestion_query_engine.py` | LLM decomposes a compound query into per-index subquestions, runs each, synthesizes a combined answer. |
