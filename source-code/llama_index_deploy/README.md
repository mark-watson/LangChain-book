# Chapter 19 — Serving a Workflow with FastAPI

Two scripts that stand up a local workflow-as-a-service on your laptop, no cloud, no LlamaCloud.

`llama-deploy` (0.9.x) is incompatible with `llama-index-core` 0.14 as of this writing, so the workflow is served directly with FastAPI instead — simpler, no external dependencies, no Redis.

## Setup

```console
$ uv sync
$ ollama pull qwen3.5:4b
```

That's it — no other services required.

## Scripts

Open two separate terminals:

| Terminal | Command |
|---|---|
| 1 | `uv run 01_serve_workflow.py` — starts the FastAPI server on port 8000 (`POST /ask`, `GET /health`). |
| 2 | `uv run 02_client.py` — sends a health check and a question, prints the answer. |

To shut down, Ctrl-C the first script.
