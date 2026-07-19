# Chapter 7 — Multi-agent supervisor pattern

Two specialist ReAct agents (research + math) coordinated by a supervisor graph. The whole thing is pure OSS LangGraph — no `create_supervisor` prebuilt, no `langgraph-swarm`, no hosted service.

## Setup

```console
$ uv sync
$ ollama pull qwen3.5:4b
```

## Layout

- `_specialists.py` — the two specialist agents (each a compiled `create_react_agent` graph).
- `_supervisor.py` — the supervisor graph that decides which specialist runs next.
- `01_run_supervisor.py` — invoke the supervisor graph on three test questions.
- `02_stream_supervisor.py` — same graph, streamed step-by-step so you can watch the routing.

## Test questions

Chosen to exercise all three code paths:

1. `"What is 137 times 24?"` — math only, one specialist call.
2. `"What is the population of Canada?"` — research only, one specialist call.
3. `"What is the population of Canada times 2?"` — research **then** math. Supervisor chains two specialists in one query.

Run either driver:

```console
$ uv run 01_run_supervisor.py
$ uv run 02_stream_supervisor.py
```
