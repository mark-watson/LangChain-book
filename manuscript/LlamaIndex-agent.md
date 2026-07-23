# Building an agent as a Workflow

Chapter "The Workflows API" introduced Workflows in the abstract. This chapter uses them to build the same thing Chapter "Building a ReAct agent with LangGraph + Ollama" built in LangGraph: a ReAct agent that alternates between "call the model" and "run the tools the model asked for" until the model returns a final answer.

As with Chapter "Building a ReAct agent with LangGraph + Ollama", we build it twice. The first version uses `FunctionAgent`, the LlamaIndex prebuilt equivalent of `create_react_agent`. The second version constructs the same behavior explicitly as a `Workflow`. Seeing the two side by side clarifies what the prebuilt is doing on your behalf.

Everything lives in `source-code/llama_index_agent/`. Setup:

```console
$ cd source-code/llama_index_agent
$ uv sync
$ ollama pull qwen3.5:4b
```

## The tools

Same two-tool shape as Chapter "Building a ReAct agent with LangGraph + Ollama":

```python
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return their product."""
    return a * b


def web_search(query: str) -> str:
    """Search DuckDuckGo. Returns the top three text results."""
    from ddgs import DDGS
    try:
        results = list(DDGS().text(query, max_results=3))
    except Exception as exc:
        return f"Search failed: {exc}"
    if not results:
        return "No results."
    return "\n\n".join(
        f"- {r.get('title', '')}\n  {r.get('body', '')}" for r in results
    )
```

Plain Python functions. LlamaIndex wraps them with `FunctionTool.from_defaults(fn=...)` at the call site, analogous to LangChain's `@tool` decorator, just applied later.

## Version 1: `FunctionAgent`

`01_function_agent.py`:

```python
import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

from _tools import multiply, web_search

tools = [
    FunctionTool.from_defaults(fn=multiply),
    FunctionTool.from_defaults(fn=web_search),
]

llm = Ollama(model="qwen3.5:4b", temperature=0, request_timeout=180.0)

agent = FunctionAgent(
    tools=tools,
    llm=llm,
    system_prompt=(
        "You are a helpful assistant. Use the provided tools when they can "
        "answer part of the question. Return a concise final answer."
    ),
)


async def main():
    for q in [
        "What is 137 times 24?",
        "What is the population of Canada, doubled?",
    ]:
        print(f"USER: {q}")
        response = await agent.run(user_msg=q)
        print(f"AGENT: {response}\n")


asyncio.run(main())
```

`FunctionAgent(tools, llm, system_prompt=...)` returns a compiled Workflow. Under the hood it is exactly the shape you would build in version 2; LlamaIndex ships it as a factory because 90% of agents want this shape.

Representative output:

```console
USER: What is 137 times 24?
AGENT: 137 times 24 is 3288.

USER: What is the population of Canada, doubled?
AGENT: Canada's population is approximately 40,528,396, so doubled that is approximately 81,056,792.
```

The second question forces two tool calls in sequence: one to `web_search`, one to `multiply`. `FunctionAgent` handles the loop internally.

## Version 2: same agent, built explicitly

`02_agent_workflow.py` builds the same behavior with `Workflow`:

```python
class ToolCallEvent(Event):
    tool_calls: list


class DoneEvent(Event):
    text: str


class ReActWorkflow(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = Ollama(model="qwen3.5:4b", temperature=0, request_timeout=180.0)

    @step
    async def call_model(
        self, ctx: Context, ev: StartEvent | ToolCallEvent
    ) -> ToolCallEvent | DoneEvent:
        if isinstance(ev, StartEvent):
            messages = [ChatMessage(role=MessageRole.USER, content=ev.get("user_msg", ""))]
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
        return ToolCallEvent(tool_calls=[])

    @step
    async def finish(self, ev: DoneEvent) -> StopEvent:
        return StopEvent(result=ev.text)
```

Three things worth spelling out.

**`Context` is shared state.** The transcript (`messages`) lives in `ctx` because both `call_model` and `run_tools` need to read and update it across the loop. `ctx.get(key)` and `ctx.set(key, value)` are the two operations you use.

**`call_model` accepts a union of two events.** It runs on the initial `StartEvent` and on every subsequent `ToolCallEvent` from the `run_tools` step. Inside, an `isinstance` check distinguishes the first invocation (which initializes the transcript) from the loop iterations (which pull it from `ctx`).

**Looping is expressed by event flow, not explicit edges.** `run_tools` returns a `ToolCallEvent`, which `call_model` accepts, which either returns another `ToolCallEvent` (looping) or a `DoneEvent` (terminating). No explicit `add_edge("tools", "model")`; the loop is implicit in the event types.

The workflow produces the same output as the `FunctionAgent` version. Which is exactly the point: the prebuilt is just this workflow with the plumbing hidden.

## When to reach for which

Same guidance as Chapter "Building a ReAct agent with LangGraph + Ollama":

- **Use `FunctionAgent`** if you have a flat list of tools and want the standard ReAct behavior.
- **Drop down to `Workflow`** if you need extra steps (a planner before the model, a validator after the tools), custom state fields beyond the transcript, or unusual routing.

## What we covered

- LlamaIndex's ReAct agent primitive is `FunctionAgent`, analogous to LangGraph's `create_react_agent`. Both build the same "call model, run tools, loop" graph.
- The manual `Workflow` version is not much more code and unlocks all of the flexibility of Chapter "The Workflows API".
- `Context` is Workflows' shared-state mechanism; use it whenever multiple steps need to see or update the same data.

Chapter "Multi-index query pipelines" builds the LlamaIndex equivalent of the supervisor pattern from Chapter "Multi-agent supervisor pattern", but for retrieval instead of tool use.
