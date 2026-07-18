# Chapter 8 — Durable, restart-safe agents

Four scripts that show how a `checkpointer` turns a stateless LangGraph app into one that remembers conversations, survives process restarts, and lets you inspect its own history.

## Setup

```console
$ uv sync
$ ollama pull qwen3.5:4b
```

## Scripts

| Script | What it shows |
|---|---|
| `01_memory_saver.py` | `MemorySaver` — three-turn conversation in one process; the agent remembers turn 1 when it answers turn 3. |
| `02_sqlite_first_run.py` | `SqliteSaver` — start a conversation and persist it to `checkpoints.db`. |
| `03_sqlite_second_run.py` | Run this **after** script 02 finishes and Python has exited. The agent resumes the same `thread_id` from disk and continues the conversation. |
| `04_state_history.py` | `.get_state_history()` — walk every checkpoint the graph has recorded for a thread. |

All four scripts share `_graph.py`, which builds the (uncompiled) state graph. Each script compiles the graph with its own checkpointer.

## Running the durability demo

The two-script demo is the whole point of the chapter. Run them separately:

```console
$ uv run 02_sqlite_first_run.py
# ... exits ...
$ uv run 03_sqlite_second_run.py
```

Between the two runs, the Python process exits completely. The second script finds the conversation waiting for it in `checkpoints.db`.

To reset, delete `checkpoints.db` and start over.
