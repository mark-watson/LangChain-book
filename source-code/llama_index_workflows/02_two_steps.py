"""Two steps chained by a custom event.

To chain steps, define your own Event subclass. The step that produces it
returns an instance; the step that consumes it takes it as its argument
type. The framework matches producer and consumer purely by type.
"""

import asyncio

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step


class UpperEvent(Event):
    text: str


class TwoStepWorkflow(Workflow):
    @step
    async def uppercase(self, ev: StartEvent) -> UpperEvent:
        return UpperEvent(text=ev.get("text", "").upper())

    @step
    async def reverse(self, ev: UpperEvent) -> StopEvent:
        return StopEvent(result=ev.text[::-1])


async def main():
    wf = TwoStepWorkflow()
    result = await wf.run(text="hello world")
    print(result)


asyncio.run(main())
