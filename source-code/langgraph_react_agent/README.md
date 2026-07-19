# Chapter 4 — Building a ReAct agent

Three ways to build the same ReAct agent — prebuilt, from scratch, and streaming — so you can see exactly what the framework is doing on your behalf.

## Setup

```console
$ uv sync
$ ollama pull qwen3.5:4b
```

## The agent's tools

Both scripts give the agent two tools:

- **`multiply(a, b)`** — a trivial arithmetic tool, so we can ask questions that require calculation.
- **`web_search(query)`** — a DuckDuckGo search returning three top text snippets. Requires network access but no API key.

## Scripts

| Script | What it shows |
|---|---|
| `01_prebuilt_agent.py` | `langgraph.prebuilt.create_react_agent` — a working ReAct agent in about 30 lines total. |
| `02_react_from_scratch.py` | The same agent built explicitly with `StateGraph`, `ToolNode`, and a conditional edge. |
| `03_streaming_agent.py` | `.stream()` on the from-scratch graph so you can watch each node execute. |

Run any one with:

```console
$ uv run 01_prebuilt_agent.py
```
