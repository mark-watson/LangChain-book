# Building a ReAct agent with LangGraph + Ollama

Chapter 6 built graphs that did not do much. This chapter uses the same graph primitives to build the standard workhorse of applied LLM development: a **ReAct agent** — a program that alternates between "let the model think" and "run a tool the model asked for" until the model decides it has a final answer.

The name comes from the [2022 ReAct paper](https://react-lm.github.io/) — *Reason and Act* — but at this point the pattern is more folklore than research. Ninety percent of the "AI agents" you read about are some variant of ReAct with two or three custom tools bolted on.

We are going to build the same agent twice. The first version uses `langgraph.prebuilt.create_react_agent`, which is what I reach for in most real projects. The second version constructs the same graph explicitly using the primitives from Chapter 6. Seeing the two side by side is the fastest way I know to understand what the prebuilt factory is doing on your behalf and when it is worth dropping down to the manual version.

## The shape of a ReAct loop

Before any code, the pattern in five bullets:

1. The user sends a message.
2. The model reads the transcript and produces either **plain text** (it is done) or **one or more tool calls** (it wants to use a tool).
3. If the model produced tool calls, each one is executed and the results are appended to the transcript as `ToolMessage`s.
4. The updated transcript goes back to the model. Go to step 2.
5. Eventually the model returns plain text with no tool calls. That is the final answer.

Chapter 6's conditional edges are exactly the mechanism for "step 2 asks a question, step 3 or 5 depending on the answer." A ReAct agent is a two-node graph:

```text
START -> model
model -> tools   (if the model's last message has tool_calls)
model -> END     (otherwise)
tools -> model   (always loop back)
```

That is the whole thing.

## The tools the agent will use

Both example scripts share `source-code/langgraph_react_agent/_tools.py`:

```python
from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return their product."""
    return a * b


@tool
def web_search(query: str) -> str:
    """Search DuckDuckGo for a query and return the top three text results."""
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


TOOLS = [multiply, web_search]
```

Two tools, chosen for contrast. `multiply` is deterministic and instant. `web_search` is nondeterministic and network-bound. Together they let us pose questions of the form "look something up, then compute something with it" that require the loop to run for real.

Both are plain Python functions decorated with `@tool`. The decorator uses the docstring as the tool description the model sees and the type annotations as the argument schema. Docstring quality directly affects tool-selection accuracy — worth taking seriously.

## Version 1: `create_react_agent`

`source-code/langgraph_react_agent/01_prebuilt_agent.py`:

```python
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from _tools import TOOLS

model = ChatOllama(model="qwen3.5:4b", temperature=0)
agent = create_react_agent(model, TOOLS)

result = agent.invoke(
    {"messages": [HumanMessage(content="What is 137 times 24?")]}
)

for m in result["messages"]:
    print(f"--- {type(m).__name__} ---")
    if getattr(m, "tool_calls", None):
        for call in m.tool_calls:
            print(f"  tool_call: {call['name']}({call['args']})")
    if m.content:
        print(m.content)
```

`create_react_agent(model, tools)` returns a compiled `StateGraph` — the same kind of object you get from `graph.compile()` in Chapter 6. It handles `bind_tools` on the model, wraps the tools in a `ToolNode`, and wires the two-node graph shown above.

A representative run:

```console
$ uv run 01_prebuilt_agent.py
--- HumanMessage ---
What is 137 times 24?
--- AIMessage ---
  tool_call: multiply({'a': 137, 'b': 24})
--- ToolMessage ---
3288
--- AIMessage ---
137 times 24 is 3288.
```

Four messages: the user's question, the model's tool call, the tool's result, the model's final answer. That is one full turn through the ReAct loop.

If the standard ReAct shape is all you need — one model, a list of tools, a normal chat transcript — `create_react_agent` is what to reach for. You do not need to see the graph plumbing.

## Version 2: the same agent, built explicitly

Now here is what `create_react_agent` actually did. `source-code/langgraph_react_agent/02_react_from_scratch.py`:

```python
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from _tools import TOOLS


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


model = ChatOllama(model="qwen3.5:4b", temperature=0, thinking=False).bind_tools(TOOLS)
tool_node = ToolNode(TOOLS)


def call_model(state: State) -> dict:
    reply = model.invoke(state["messages"])
    return {"messages": [reply]}


def route_after_model(state: State) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


graph = StateGraph(State)
graph.add_node("model", call_model)
graph.add_node("tools", tool_node)

graph.add_edge(START, "model")
graph.add_conditional_edges(
    "model", route_after_model, {"tools": "tools", END: END}
)
graph.add_edge("tools", "model")

agent = graph.compile()
```

Walking through it top to bottom.

**The state.** A single `messages` field reduced by `add_messages`, exactly like the last example of Chapter 6. Every message the graph produces — the model's replies, the tool results — gets appended here.

**The model.** `ChatOllama(...).bind_tools(TOOLS)` gives the model the list of callable tools. When invoked, the model may respond with `.tool_calls` populated instead of `.content`.

**`ToolNode`.** From `langgraph.prebuilt`. Given a list of tools, `ToolNode` reads the last message on the transcript, executes each tool call in that message against the matching tool, and returns a list of `ToolMessage` objects containing the results. You could write this yourself in about a dozen lines — reading `state["messages"][-1].tool_calls`, dispatching each one, wrapping results in `ToolMessage(content=..., tool_call_id=...)` — but there is no reason to.

**The two node functions.** `call_model` invokes the model on the transcript and returns its reply. `route_after_model` is not a node, it is a router — a plain function from state to a string. If the model's last reply has tool calls, we route to `"tools"`; otherwise we route to `END`.

**The wiring.** Three edges. Start goes to the model. The model is followed by a conditional edge that either goes to the tool node or ends. After tools run, we always loop back to the model.

Running it gives the same output as version 1. That is the point: the prebuilt factory is not magic, it is this file compressed.

## When to reach for which version

- **Use `create_react_agent`** if you have a model and a flat list of tools and you want the standard behavior. It is one line of graph construction and it stays in sync with best practices as the LangGraph team refines the pattern.
- **Drop down to the manual `StateGraph`** if you need any of the following: extra state fields beyond `messages` (retrieval context, user profile, running scratchpad), extra nodes (a planner before the model, a validator after the tools, a memory writer at the end), custom routing (route based on which tool was called, not just whether one was called), or extra edges (parallel tool execution, human-in-the-loop interrupts).

In my own projects I probably start with `create_react_agent` about half the time and drop to the manual graph the other half. The nice thing about LangGraph 1.0 is that the migration is mechanical when you need it — the primitives are the same.

## Watching each step with `.stream()`

The compiled agent is a `Runnable`, so it supports `.stream()`. Each yielded item is a dict of the form `{node_name: partial_state_update_the_node_produced}`. That is the fastest way I know to understand what an agent is actually doing.

`source-code/langgraph_react_agent/03_streaming_agent.py` reconstructs the same manual graph and streams a two-tool question:

```python
question = (
    "Search for the current population of Canada, then multiply it by 2."
)

for step in agent.stream({"messages": [HumanMessage(content=question)]}):
    for node_name, node_output in step.items():
        print(f"=== node: {node_name} ===")
        for m in node_output["messages"]:
            if getattr(m, "tool_calls", None):
                for call in m.tool_calls:
                    print(f"  tool_call: {call['name']}({call['args']})")
            if m.content:
                snippet = m.content if len(m.content) < 400 else m.content[:400] + "..."
                print(f"  {type(m).__name__}: {snippet}")
        print()
```

A representative session (your model's tool choices and the DuckDuckGo results will vary):

```console
$ uv run 03_streaming_agent.py
USER: Search for the current population of Canada, then multiply it by 2.

=== node: model ===
  tool_call: web_search({'query': 'current population of Canada'})

=== node: tools ===
  ToolMessage: - Population of Canada - Wikipedia
    The population of Canada is 40,528,396 as of ...

=== node: model ===
  tool_call: multiply({'a': 40528396, 'b': 2})

=== node: tools ===
  ToolMessage: 81056792

=== node: model ===
  AIMessage: Doubling Canada's current population of about 40,528,396 gives approximately 81,056,792.
```

Five node executions. Two `model` invocations that produced tool calls, one that produced the final answer. Two `tools` invocations that fed data back into the transcript. `.stream()` makes this legible in a way that `.invoke()` — which only returns the final state — cannot.

I use `.stream()` for essentially every agent I write, and I usually convert it to `.invoke()` only after I am satisfied with the behavior. In practice, an agent that streams sensibly almost always invokes sensibly, and streaming while iterating catches bugs early.

## Two failure modes worth knowing

**The model does not call the tool at all.** If `bind_tools` is pointed at a model that does not support tool calling (most 1-3 B parameter models, and some larger models that were not fine-tuned for it), the model will respond in prose describing what it *would* do instead of returning `.tool_calls`. The router will then send the response straight to `END` and the agent will terminate with a plausible-sounding but wrong answer. If you see the model narrating its plan instead of executing it, you probably have the wrong model. As of mid-2026 the models I use for tool-calling work in this book are `qwen3.5:4b`, `llama3.2:3b`, `gemma3:12b-it-qat`, and `mistral-small`.

**The agent loops forever.** With a bad system prompt or a weak model, the ReAct loop can call the same tool over and over. LangGraph does not enforce a step limit by default. In production I always pass `recursion_limit` when invoking:

```python
agent.invoke({"messages": [...]}, config={"recursion_limit": 25})
```

Twenty-five is my usual number — enough for a real multi-step task, low enough to bail out before the LLM bill or the wall clock runs away.

## What we covered

- The ReAct loop is a two-node graph with one conditional edge — precisely the pattern you would write with the primitives from Chapter 6.
- `create_react_agent` is a factory that builds and compiles that graph for you. Use it unless you need custom state, extra nodes, or unusual routing.
- Building the graph explicitly is not much more code and unlocks all of Chapter 6's flexibility.
- `.stream()` on the compiled agent is essential for debugging and iteration.

Chapter 8 keeps the same agent but adds a checkpointer, which is what turns it from a script that runs once into a service that can pause, resume, and survive a process restart without losing conversation state.
