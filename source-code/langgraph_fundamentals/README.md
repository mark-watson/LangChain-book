# Chapter 3 — LangGraph 1.0 fundamentals

Four small graphs that together cover the mechanics you need for the rest of Part I. No LLM in the first three — the goal is to make the graph engine legible before we point it at a model.

## Setup

```console
$ uv sync
```

Only the fourth script uses an LLM. If you want to run it:

```console
$ ollama pull qwen3.5:4b
```

## Scripts

| Script | What it shows |
|---|---|
| `01_hello_graph.py` | Minimum viable graph: one TypedDict state, one node, one edge. |
| `02_reducers.py` | `Annotated[..., operator.add]` — how state accumulates across nodes. |
| `03_conditional_routing.py` | `add_conditional_edges` — the graph picks its own next node based on state. |
| `04_llm_in_a_node.py` | A node that calls `ChatOllama`, using the `add_messages` reducer. |

Run any one with:

```console
$ uv run 01_hello_graph.py
```
