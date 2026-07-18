# Chapter 14 — LlamaIndex 0.14 in one hour

Four scripts that together cover every LlamaIndex primitive we use in the rest of Part II: documents, indices, query engines, retrievers, persistence, and provider swapping.

## Setup

```console
$ uv sync
$ ollama pull qwen3.5:4b
```

The first script also downloads a small embedding model from Hugging Face (~130 MB), cached under `~/.cache/huggingface/`.

## Corpus

All four scripts read from `../data/` — the same four short text files (chemistry, economics, health, sports) used by Chapter 4.

## Scripts

| Script | What it shows |
|---|---|
| `01_hello_llamaindex.py` | Load documents, build an in-memory `VectorStoreIndex`, query it — all local (Ollama LLM + HF embeddings via `Settings`). |
| `02_hosted_swap.py` | The same program pointed at OpenAI. Only the `Settings.llm` line changes. Requires `OPENAI_API_KEY`. |
| `03_persist_and_reload.py` | Build an index, persist to `./storage/`, then reload it from disk and query it. |
| `04_retriever_only.py` | Use `.as_retriever()` instead of `.as_query_engine()` — get raw nodes back without the LLM synthesis step. |

Run any one with:

```console
$ uv run 01_hello_llamaindex.py
```
