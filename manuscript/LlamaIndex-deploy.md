# Serving a Workflow with `llama-deploy`

You have built a working Workflow. You want it to serve traffic — accept requests, run the workflow, return the answer — without writing FastAPI boilerplate for every project. `llama-deploy` is the open-source package that does this. It runs on your laptop, in a Docker container, or on a VPS; it needs Redis; it does not touch LlamaCloud.

This chapter walks through the smallest useful `llama-deploy` setup: one control plane, one workflow service, one client. The workflow itself is a two-step Q&A workflow (deliberately trivial so the deployment plumbing is the interesting part).

Everything lives in `source-code/llama_index_deploy/`. Setup:

```console
$ cd source-code/llama_index_deploy
$ uv sync
$ ollama pull qwen3.5:4b
```

You also need Redis running locally:

```console
$ brew install redis && brew services start redis
# or
$ docker run -d --name llama-deploy-redis -p 6379:6379 redis:7
```

## The moving parts

`llama-deploy` has three components:

- **Control plane** — accepts client requests, dispatches them to workflow services, collects results. Speaks HTTP on port 8000 by default.
- **Workflow services** — one per workflow you want to serve. Each registers itself with the control plane on startup and pulls jobs from the shared message queue (Redis).
- **Clients** — anything that talks to the control plane. A Python script, another service, a curl command against the HTTP endpoint.

The three scripts in this chapter play those three roles. You run each in its own terminal.

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

One step, one LLM call. In a real deployment this would be your actual workflow — a RAG pipeline, a ReAct agent, a multi-step research workflow.

## Terminal 1: control plane

`01_control_plane.py`:

```python
import asyncio

from llama_deploy import ControlPlaneConfig, deploy_core


async def main():
    config = ControlPlaneConfig(host="127.0.0.1", port=8000)
    await deploy_core(config)


asyncio.run(main())
```

`deploy_core(config)` starts the control-plane HTTP server on the configured port and blocks. Ctrl-C to stop.

## Terminal 2: workflow service

`02_workflow_service.py`:

```python
import asyncio

from llama_deploy import ControlPlaneConfig, WorkflowServiceConfig, deploy_workflow

from _workflow import QAWorkflow


async def main():
    control_plane = ControlPlaneConfig(host="127.0.0.1", port=8000)
    service = WorkflowServiceConfig(
        service_name="qa_workflow",
        host="127.0.0.1",
        port=8001,
    )
    await deploy_workflow(
        workflow=QAWorkflow(timeout=180.0),
        workflow_config=service,
        control_plane_config=control_plane,
    )


asyncio.run(main())
```

`deploy_workflow` instantiates the workflow, registers the service with the control plane, and starts pulling jobs from the message queue. `service_name` is how clients address this workflow — think of it as the workflow's route.

## Terminal 3: client

`03_client.py`:

```python
import asyncio

from llama_deploy import Client


async def main():
    client = Client(control_plane_url="http://127.0.0.1:8000")
    session = await client.core.sessions.create()

    result = await session.run(
        service_name="qa_workflow",
        question="What is the capital of Arizona?",
    )
    print(f"AGENT: {result}")


asyncio.run(main())
```

`sessions.create()` opens a session — the unit of client interaction. `session.run(service_name=..., **workflow_inputs)` sends the request to the named workflow service and waits for the result.

Expected output:

```console
$ uv run 03_client.py
AGENT: Phoenix is the capital of Arizona.
```

## Scaling from here

Everything above runs on one laptop. The path to production is incremental, not a rewrite:

1. **One workflow service, one process** — where you started.
2. **One workflow service, many processes** — start N copies of `02_workflow_service.py` behind the same control plane. Redis handles the fan-out; each request goes to whichever process pulls it first. Horizontal scaling for free.
3. **Multiple workflow services** — deploy several `deploy_workflow(...)` calls, each with a different `service_name`. Clients pick which one to invoke by name.
4. **Move to a small VPS** — `llama-deploy` needs Python and Redis; both fit comfortably on a $5/month VPS. Appendix D walks through this in more detail.
5. **HTTP clients from other languages** — the control plane speaks HTTP, so anything that can `POST` JSON can invoke your workflow. No requirement to use `llama-deploy`'s Python client.

Nothing on this path involves LlamaCloud, an API key you pay for, or a managed service. The whole thing is open source Python plus Redis.

## What we covered

- `llama-deploy` turns any `Workflow` into a service without hand-rolled FastAPI.
- Three components: control plane, workflow services, clients. Redis in between for the message queue.
- Local development is three scripts in three terminals. Production is the same three components on different hosts.
- The path from "runs on my laptop" to "serves real traffic on a VPS" is incremental, not a rewrite.

That closes Part II. The four appendices that follow cover cross-cutting topics that apply to both LangChain and LlamaIndex projects: choosing a model, doing evaluation without LangSmith, doing observability without LangSmith, and putting a small LLM app on a $5/month VPS.
