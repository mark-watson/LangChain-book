# Serving a Workflow with FastAPI

You have built a working Workflow. You want it to serve traffic (accept requests, run the workflow, return the answer) without a lot of ceremony. This chapter walks through the smallest useful setup: one HTTP server process, one client. The workflow itself is a two-step Q&A workflow (deliberately trivial so the deployment plumbing is the interesting part).

A quick note on `llama-deploy`. If you have read about LlamaIndex deployment before, you may expect this chapter to be built on it: a control plane, one or more workflow services, a Redis-backed message queue between them. That was the plan, and for most of this book's life it was the code sitting in this directory. It broke: the current `llama-deploy` release (0.9.x) is incompatible with `llama-index-core` 0.14, the version this book uses everywhere else. Rather than pin an old `llama-index-core` just to keep a deployment framework working, this chapter serves the workflow directly with FastAPI. It is simpler, it has no external dependencies (no Redis, nothing to install with Homebrew or Docker), and it is the same OSS-first, run-it-on-your-laptop approach as every other chapter in the book. If `llama-deploy` catches back up, the swap is confined to this one file; the workflow itself does not change at all.

Everything lives in `source-code/llama_index_deploy/`. Setup:

```console
$ cd source-code/llama_index_deploy
$ uv sync
$ ollama pull qwen3.5:4b
```

No Redis, no other services: that is the whole setup.

## The moving parts

Two things run:

- **The server**: a FastAPI app wrapping the Workflow, served by `uvicorn`. It exposes `POST /ask` (run the workflow and return the answer) and `GET /health` (liveness check). One process, one port.
- **The client**: anything that can send an HTTP request. This chapter uses a small `httpx` script, but a `curl` command or a client in any other language works identically, since the server speaks plain JSON over HTTP.

You run the server in one terminal and the client in another.

## The workflow being served

`_workflow.py` is deliberately small:

```python
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from llama_index.llms.ollama import Ollama


class QAWorkflow(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = Ollama(model="qwen3.5:4b", temperature=0, request_timeout=180.0)

    @step
    async def answer(self, ev: StartEvent) -> StopEvent:
        question = ev.get("question", "")
        reply = await self.llm.acomplete(f"Answer briefly: {question}")
        return StopEvent(result=reply.text.strip())
```

One step, one LLM call. In a real deployment this would be your actual workflow: a RAG pipeline, a ReAct agent, a multi-step research workflow. Nothing about the workflow class itself changes when you serve it; that is the point of the Workflow abstraction from Chapter "The Workflows API".

## Terminal 1: the server

`01_serve_workflow.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from _workflow import QAWorkflow

app = FastAPI(title="QA Workflow API")
workflow = QAWorkflow(timeout=180.0)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    """Run the QA workflow and return the answer."""
    result = await workflow.run(question=request.question)
    return AnswerResponse(answer=str(result))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

Walking through it. The `QAWorkflow` is instantiated exactly once, at import time, and reused across every request, the same pattern you would use for a model or a database connection pool. `QuestionRequest` and `AnswerResponse` are Pydantic models; FastAPI uses them to validate the incoming JSON body and to generate an OpenAPI schema for free (visit `http://127.0.0.1:8000/docs` while the server is running). `ask` is an `async def` route, so it can `await workflow.run(...)` without blocking the event loop while the LLM call is in flight, so other requests can be served concurrently. `health` is the kind of endpoint a load balancer or container orchestrator polls before sending traffic to this process.

Run it and leave it running:

```console
$ uv run 01_serve_workflow.py
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

## Terminal 2: the client

`02_client.py`:

```python
import asyncio

import httpx


async def main():
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Health check
        resp = await client.get("http://127.0.0.1:8000/health")
        print(f"Health: {resp.json()}")

        # Ask a question
        resp = await client.post(
            "http://127.0.0.1:8000/ask",
            json={"question": "What is the capital of Arizona?"},
        )
        data = resp.json()
        print(f"AGENT: {data['answer']}")


asyncio.run(main())
```

Nothing LlamaIndex-specific here: this is a generic async HTTP client hitting two REST endpoints. `httpx.AsyncClient` is used instead of the synchronous `requests` because the rest of the book's async code (Workflows, agents) already pulls in an async HTTP stack; for a one-off script, `requests` or `curl` work exactly as well.

Expected output:

```console
$ uv run 02_client.py
Health: {'status': 'ok'}
AGENT: Phoenix is the capital of Arizona.
```

The same request from the command line, no Python client needed:

```console
$ curl -s -X POST http://127.0.0.1:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the capital of Arizona?"}'
{"answer":"Phoenix is the capital of Arizona."}
```

## Scaling from here

Everything above runs as one process on one laptop. The path to production is incremental, not a rewrite:

1. **One process**: where you started. Fine for development and for low-traffic internal tools.
2. **Multiple worker processes, one machine**: `uvicorn`'s `--workers` flag runs several copies of the app behind one port and spreads connections across them; that is normally the first lever to pull for more throughput. It needs the app passed as an import string (`uvicorn server:app --workers 4`) rather than as a live object, which means two small changes from what is shown above: move the FastAPI app into a plain importable module (the `01_` prefix used for teaching order in this book's scripts is not a valid Python module name to import) and read `host`/`port` from the environment instead of hard-coding them. Neither the workflow nor the routes change.
3. **More workflows, same server**: add another route (`/summarize`, `/extract`, whatever the next workflow does), each backed by its own `Workflow` instance, to the same FastAPI app. Ordinary FastAPI, not a new concept to learn.
4. **Move to a small VPS**: the server needs Python and enough RAM to hold whatever model it talks to (or a network path to a hosted model); nothing else. Appendix D walks through putting exactly this kind of process behind a reverse proxy on a $5/month VPS.
5. **Clients in any language**: the server speaks plain HTTP and JSON, so anything that can issue a `POST` (`curl`, a browser `fetch`, a mobile app, a service written in Go) can call it. There is no client library to install anywhere except in this one example script, and even that is just a convenience.

If a workflow's steps genuinely need a shared task queue (long-running jobs, retries, work distributed across many machines), that is the point where reaching for Redis plus a worker library (`arq`, Celery) starts to pay for itself. It is worth adding when a specific problem shows up, not before; nothing in this chapter needs it.

## What we covered

- A LlamaIndex `Workflow` is served over HTTP with plain FastAPI: instantiate it once, `await workflow.run(...)` inside an `async def` route.
- Two endpoints are enough for a real service: one that does the work, one (`/health`) that reports liveness.
- The client side is unremarkable: any HTTP client, in any language, talking JSON.
- The scaling path is `uvicorn --workers`, more routes, and a VPS behind a reverse proxy: infrastructure you add when you need it, not infrastructure the framework requires up front.

That closes Part II. The four appendices that follow cover cross-cutting topics that apply to both LangChain and LlamaIndex projects: choosing a model, doing evaluation without LangSmith, doing observability without LangSmith, and putting a small LLM app on a $5/month VPS.
