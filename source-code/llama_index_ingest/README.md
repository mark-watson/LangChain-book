# Chapter 12 — Local documents and local embeddings

Four scripts covering the ingestion side of LlamaIndex end-to-end: loading files from disk, building Documents by hand, comparing two local embedding models on the same corpus, and running documents through an ingestion pipeline that chunks them before indexing.

## Setup

```console
$ uv sync
```

Script 3 downloads two small embedding models from Hugging Face on first run (about 220 MB total).

## Scripts

| Script | What it shows |
|---|---|
| `01_directory_loading.py` | `SimpleDirectoryReader` with a `file_metadata` callback, recursive scanning, and exclude patterns. Inspects the metadata every Document ends up with. |
| `02_custom_documents.py` | Building `Document` objects by hand — the pattern when your data comes from an API, a database, or anywhere `SimpleDirectoryReader` doesn't cover. |
| `03_embedding_comparison.py` | Retrieves the same query against the same corpus using two different embedding models (BGE-small and MiniLM) so you can see how the choice affects ranking. |
| `04_ingestion_pipeline.py` | `IngestionPipeline` with a `SentenceSplitter` — how to chunk long documents into smaller Nodes before they hit the index. |

All four scripts read from `../data/` (the same four-file corpus used in Chapters 2 and 11).

Run any one with:

```console
$ uv run 01_directory_loading.py
```
