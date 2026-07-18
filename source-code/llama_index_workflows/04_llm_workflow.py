"""A three-step LLM workflow: classify → answer or decline.

Step 1 asks the LLM whether the incoming question is on-topic (about
software). If yes, step 2 answers it. If no, step 3 politely declines.

This is the shape of nearly every real content-moderation, routing, or
gatekeeping workflow: a cheap classification step guards an expensive
answering step.
"""

import asyncio

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.llms.ollama import Ollama


class OnTopicEvent(Event):
    question: str


class OffTopicEvent(Event):
    question: str


class TopicRoutingWorkflow(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = Ollama(model="qwen3.5:4b", temperature=0, request_timeout=120.0)

    @step
    async def classify(self, ev: StartEvent) -> OnTopicEvent | OffTopicEvent:
        question = ev.get("question", "")
        answer = (
            await self.llm.acomplete(
                "Reply with a single character, Y or N. "
                "Is the following question about software or programming?\n\n"
                f"{question}"
            )
        ).text.strip().upper()
        if answer.startswith("Y"):
            return OnTopicEvent(question=question)
        return OffTopicEvent(question=question)

    @step
    async def answer(self, ev: OnTopicEvent) -> StopEvent:
        reply = await self.llm.acomplete(
            f"Answer this software question briefly:\n\n{ev.question}"
        )
        return StopEvent(result=reply.text.strip())

    @step
    async def decline(self, ev: OffTopicEvent) -> StopEvent:
        return StopEvent(
            result="Sorry, I only answer questions about software and programming."
        )


async def main():
    wf = TopicRoutingWorkflow(timeout=120.0)
    for q in [
        "What is a Python decorator?",
        "What is the capital of France?",
    ]:
        print(f"USER: {q}")
        print(f"AGENT: {await wf.run(question=q)}\n")


asyncio.run(main())
