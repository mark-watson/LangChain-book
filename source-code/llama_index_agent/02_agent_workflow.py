"""The same ReAct agent built explicitly with the Workflows API.

The graph shape is identical to Chapter 7's LangGraph version:

  START -> call_model -> [tool_calls? tools : done]
                       tools -> call_model  (loop)

Two steps, one branching return type, `Context` to hold the running
message list across the loop.
"""

import asyncio
import json

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.ollama import Ollama

from _tools import multiply, web_search

TOOLS = [
    FunctionTool.from_defaults(fn=multiply),
    FunctionTool.from_defaults(fn=web_search),
]
TOOLS_BY_NAME = {t.metadata.name: t for t in TOOLS}


class ToolCallEvent(Event):
    """Emitted when the model wants to call one or more tools."""

    tool_calls: list


class DoneEvent(Event):
    """Emitted when the model returns a final answer."""

    text: str


class ReActWorkflow(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = Ollama(
            model="qwen3.5:4b", temperature=0, request_timeout=180.0
        )

    @step
    async def call_model(
        self, ctx: Context, ev: StartEvent | ToolCallEvent
    ) -> ToolCallEvent | DoneEvent:
        # First entry: initialize the transcript from the user's message.
        if isinstance(ev, StartEvent):
            messages = [
                ChatMessage(role=MessageRole.USER, content=ev.get("user_msg", ""))
            ]
        else:
            messages = await ctx.get("messages")

        response = await self.llm.achat_with_tools(TOOLS, chat_history=messages)
        messages.append(response.message)
        await ctx.set("messages", messages)

        tool_calls = self.llm.get_tool_calls_from_response(response)
        if tool_calls:
            return ToolCallEvent(tool_calls=tool_calls)
        return DoneEvent(text=str(response))

    @step
    async def run_tools(self, ctx: Context, ev: ToolCallEvent) -> ToolCallEvent | DoneEvent:
        messages = await ctx.get("messages")
        for tc in ev.tool_calls:
            tool = TOOLS_BY_NAME[tc.tool_name]
            result = tool.call(**tc.tool_kwargs)
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL,
                    content=json.dumps(str(result.raw_output)),
                    additional_kwargs={"tool_call_id": tc.tool_id},
                )
            )
        await ctx.set("messages", messages)
        # Loop back to the model by returning a ToolCallEvent-shaped signal.
        # call_model will consume it and produce either more tool calls or a Done.
        return ToolCallEvent(tool_calls=[])

    @step
    async def finish(self, ev: DoneEvent) -> StopEvent:
        return StopEvent(result=ev.text)


async def main():
    wf = ReActWorkflow(timeout=180.0)
    for q in [
        "What is 137 times 24?",
        "What is the population of Canada, doubled?",
    ]:
        print(f"USER: {q}")
        result = await wf.run(user_msg=q)
        print(f"AGENT: {result}\n")


asyncio.run(main())
