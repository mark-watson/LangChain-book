"""Branching by event type.

A step's return type can be a Union of possible events. Whichever event
the step returns determines which downstream step runs next. This is
LlamaIndex's equivalent of LangGraph's conditional edges.
"""

import asyncio

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step


class PositiveEvent(Event):
    value: int


class NegativeEvent(Event):
    value: int


class BranchingWorkflow(Workflow):
    @step
    async def classify(self, ev: StartEvent) -> PositiveEvent | NegativeEvent:
        n = ev.get("number", 0)
        return PositiveEvent(value=n) if n >= 0 else NegativeEvent(value=n)

    @step
    async def handle_positive(self, ev: PositiveEvent) -> StopEvent:
        return StopEvent(result=f"{ev.value} is positive.")

    @step
    async def handle_negative(self, ev: NegativeEvent) -> StopEvent:
        return StopEvent(result=f"{ev.value} is negative.")


async def main():
    wf = BranchingWorkflow()
    for n in [5, -3, 0]:
        print(await wf.run(number=n))


asyncio.run(main())
