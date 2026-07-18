"""Serve a LlamaIndex Workflow as an HTTP API using FastAPI.

This replaces the old llama-deploy control-plane pattern. The new
llama-deploy (0.9.x) is incompatible with llama-index-core 0.14, so
we serve the workflow directly with FastAPI — which is simpler, has
no external dependencies, and is the OSS-first approach the book
advocates.

Run this in its own terminal; leave it running:

    uv run python 01_serve_workflow.py

The server listens on http://127.0.0.1:8000
POST a JSON body {"question": "..."} to /ask
"""

import asyncio

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
