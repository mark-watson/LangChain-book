# Chapter 15 — The Workflows API

Four workflows, from a hello-world to a branching multi-step LLM pipeline.

## Setup

```console
$ uv sync
$ ollama pull qwen3.5:4b
```

## Scripts

| Script | What it shows |
|---|---|
| `01_hello_workflow.py` | Minimum viable workflow: one step, one event, run to completion. |
| `02_two_steps.py` | Two steps chained by a custom event. |
| `03_branching.py` | A step that emits different events based on state, driving branching downstream. |
| `04_llm_workflow.py` | A three-step workflow that uses an LLM in each step: classify → answer or decline. |
