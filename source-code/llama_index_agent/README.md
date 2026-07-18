# Chapter 19 — Building an agent as a Workflow

Two paths to the same ReAct agent: the prebuilt `FunctionAgent` factory, and the same agent built explicitly with the Workflows API.

## Setup

```console
$ uv sync
$ ollama pull qwen3.5:4b
```

## Scripts

| Script | What it shows |
|---|---|
| `01_function_agent.py` | `FunctionAgent(tools, llm)` — a working ReAct agent in about 30 lines. LlamaIndex's answer to LangGraph's `create_react_agent`. |
| `02_agent_workflow.py` | The same agent built manually as a `Workflow` with three steps (call model, run tools, decide next). Useful when the prebuilt shape does not fit. |
