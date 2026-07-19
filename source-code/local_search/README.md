# Chapter 10 — Perplexity-style local search agent

A LangGraph pipeline that answers open-ended questions by searching the web, filtering results for relevance, fetching page text, summarizing each page against the query, and synthesizing a final answer. Runs entirely on your laptop — Ollama for the LLM, DuckDuckGo for search, `trafilatura` for HTML-to-text extraction.

## Setup

```console
$ uv sync
$ ollama pull qwen3.5:4b
```

## The pipeline

```text
START -> search -> filter -> fetch -> summarize -> synthesize -> END
```

Each node adds one field to the shared state. The final state carries the full trail (query, raw results, filtered results, fetched texts, per-page summaries, final synthesis).

## Scripts

| Script | What it shows |
|---|---|
| `_pipeline.py` | The graph — one node per pipeline step. |
| `01_search.py` | Invoke the pipeline on a question and print the final synthesis. |
| `02_stream_search.py` | Stream the pipeline so you can watch each stage's output. |

Run either driver:

```console
$ uv run 01_search.py
$ uv run 02_stream_search.py
```

Because both scripts hit DuckDuckGo and fetch a handful of live web pages, expect the whole run to take 10-60 seconds depending on your network and how large the pages are.
