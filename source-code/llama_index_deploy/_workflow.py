"""The workflow being served — a minimal Q&A workflow.

Kept small on purpose so the deployment plumbing is the interesting part.
"""

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
