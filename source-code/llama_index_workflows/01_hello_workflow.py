"""The smallest useful workflow: one step from StartEvent to StopEvent.

Every workflow is a class that inherits from Workflow. Each @step method
takes an event as input and returns an event. The framework wires steps
together based on their type annotations — the step that accepts a
StartEvent runs first, the step that returns a StopEvent ends the run.
"""

import asyncio

from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step


class HelloWorkflow(Workflow):
    @step
    async def greet(self, ev: StartEvent) -> StopEvent:
        name = ev.get("name", "world")
        return StopEvent(result=f"Hello, {name}!")


async def main():
    wf = HelloWorkflow()
    result = await wf.run(name="Mark")
    print(result)


asyncio.run(main())
