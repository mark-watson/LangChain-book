# Chapter 22 — Serving a Workflow with llama-deploy

Three scripts that stand up a local workflow-as-a-service on your laptop, no cloud, no LlamaCloud.

## Setup

```console
$ uv sync
$ ollama pull qwen3.5:4b
```

You also need a local **Redis** running. If you don't have one:

```console
$ brew install redis && brew services start redis
# or
$ docker run -d --name llama-deploy-redis -p 6379:6379 redis:7
```

## Scripts

Open three separate terminals and run in order:

| Terminal | Command |
|---|---|
| 1 | `uv run 01_control_plane.py` — starts the control plane on port 8000. |
| 2 | `uv run 02_workflow_service.py` — registers the workflow with the control plane. |
| 3 | `uv run 03_client.py` — sends a request and prints the answer. |

To shut down, Ctrl-C the first two scripts.
