# LangGraph 1.0 Fundamentals

## What LangGraph is, and is not

LangGraph is an MIT-licensed Python library that lets you build stateful LLM applications as directed graphs of small step functions. It runs entirely on your laptop with no hosted service in the loop. This chapter and the four that follow use only the open source `langgraph` package plus in-process persistence: no LangSmith account, no LangGraph Platform / LangGraph Cloud, no LangSmith Deployment, no LangGraph Studio. Those are separate paid or freemium products from the LangChain company that this book deliberately does not cover; the library on its own is complete for everything we do.

## Why a graph at all

By now you have seen that a `Runnable` chain (`prompt | model | parser`) is enough for a huge class of LLM apps. What it is not enough for is anything with a loop, a decision point, or state that persists across steps. A ReAct agent needs to alternate between "call the model" and "run a tool" as many times as the model asks. A human-in-the-loop workflow needs to pause partway through and wait for approval. A long-running assistant needs to remember the conversation across process restarts. LCEL chains cannot express any of that cleanly.

LangGraph reintroduces the state machine as the primary abstraction. You describe your application as:

- A **state schema**: a TypedDict of the fields your app needs to track.
- A set of **nodes**: plain Python functions from state to a partial state update.
- A set of **edges**: which node runs after which, optionally chosen by a routing function that reads the current state.

You compile the graph once and then `.invoke()` or `.stream()` it like any other `Runnable`. The graph engine handles running nodes in order, applying state updates, and threading the same state dict through the whole execution.

Nothing in that description mentions LLMs. That is deliberate: the graph engine does not care whether a node happens to call a model, hit a database, do arithmetic, or all three. Teaching the graph mechanics first, without a model in sight, makes the machinery easy to reason about. We add an LLM in the last example of the chapter.

## Setup

The example code lives in `source-code/langgraph_fundamentals/`:

```console
$ cd source-code/langgraph_fundamentals
$ uv sync
```

Only the fourth script calls an LLM; if you want to run it, pull the same tool-capable model we have been using:

```console
$ ollama pull qwen3.5:4b
```

## Example 1: hello graph

The smallest program that exercises every piece of the LangGraph API. `01_hello_graph.py`:

```python
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    question: str
    answer: str


def answer_node(state: State) -> dict:
    """A node is a function from state to a partial state update."""
    return {"answer": f"You asked: {state['question']!r}. I have no idea."}


graph = StateGraph(State)
graph.add_node("answer", answer_node)
graph.add_edge(START, "answer")
graph.add_edge("answer", END)

app = graph.compile()

final_state = app.invoke({"question": "What is the meaning of life?"})
print(final_state)
```

Five things happen. We declare a `State` TypedDict with two string fields. We write one node function that takes the full state and returns a dict containing only the fields it wants to update. We build a `StateGraph` around the schema, register the node, and wire it between two sentinels named `START` and `END`. We `.compile()` the graph, which returns a `Runnable` we can invoke. And we invoke it with an initial state dict.

Output:

```console
$ uv run 01_hello_graph.py
{'question': 'What is the meaning of life?', 'answer': "You asked: 'What is the meaning of life?'. I have no idea."}
```

The final state is a dict with both fields populated: the one we passed in and the one the node wrote.

Two conventions worth stating explicitly. A node returns a *partial* state update, not the full state; the engine merges the returned dict into the current state. Fields the node did not mention are left alone. And a node is a *pure function* from state in to state update out. The engine is what mutates the state; the node just describes what should change.

## Example 2: state fields with reducers

The default merge rule is replacement: when a node returns `{"key": value}`, the new value overwrites the old one. Sometimes that is not what you want. If a node produces a log line, you want the new line appended to the existing log, not written on top of it. If several nodes produce chat messages, you want the transcript to grow, not to be reset each turn.

You express this by annotating the state field with a **reducer** function. `02_reducers.py`:

```python
from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    log: Annotated[list[str], add]


def step_one(state: State) -> dict:
    return {"log": ["step one ran"]}


def step_two(state: State) -> dict:
    return {"log": ["step two ran"]}


def step_three(state: State) -> dict:
    return {"log": ["step three ran"]}


graph = StateGraph(State)
graph.add_node("one", step_one)
graph.add_node("two", step_two)
graph.add_node("three", step_three)
graph.add_edge(START, "one")
graph.add_edge("one", "two")
graph.add_edge("two", "three")
graph.add_edge("three", END)

app = graph.compile()

final_state = app.invoke({"log": []})
for line in final_state["log"]:
    print(line)
```

`Annotated[list[str], add]` tells the graph engine "the `log` field is a list of strings, and when a node returns a new value for it, combine the old and new with `operator.add`." Because `operator.add` on lists is concatenation, each node's log line ends up appended to what came before.

Output:

```console
$ uv run 02_reducers.py
step one ran
step two ran
step three ran
```

Any two-argument function that takes the current value and the update and returns the new value can be a reducer. In the last example of the chapter we use `langgraph.graph.message.add_messages`, which knows how to intelligently merge message lists (including pairing tool-call messages with their tool-result messages). Ninety percent of the graphs I write in practice use exactly two reducers: `add_messages` for the transcript, and `operator.add` for any auxiliary log or trace.

## Example 3: conditional routing

Sequential edges are enough for a pipeline. To model an actual state machine you need edges the graph engine can *choose* at runtime, based on the current state. That is `add_conditional_edges`. `03_conditional_routing.py`:

```python
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    number: int
    verdict: str


def classify(state: State) -> dict:
    n = state["number"]
    if n < 0:
        verdict = "negative"
    elif n == 0:
        verdict = "zero"
    else:
        verdict = "positive"
    return {"verdict": verdict}


def announce_negative(state: State) -> dict:
    return {"verdict": f"{state['number']} is negative."}


def announce_zero(state: State) -> dict:
    return {"verdict": "You gave me zero."}


def announce_positive(state: State) -> dict:
    return {"verdict": f"{state['number']} is positive."}


def route_after_classify(state: State) -> str:
    return state["verdict"]


graph = StateGraph(State)
graph.add_node("classify", classify)
graph.add_node("neg", announce_negative)
graph.add_node("zero", announce_zero)
graph.add_node("pos", announce_positive)

graph.add_edge(START, "classify")
graph.add_conditional_edges(
    "classify",
    route_after_classify,
    {"negative": "neg", "zero": "zero", "positive": "pos"},
)
graph.add_edge("neg", END)
graph.add_edge("zero", END)
graph.add_edge("pos", END)

app = graph.compile()

for n in [-4, 0, 7]:
    result = app.invoke({"number": n, "verdict": ""})
    print(result["verdict"])
```

The interesting call is `graph.add_conditional_edges(source, router, mapping)`. After the `source` node runs, the engine calls `router(state)` and follows the edge to whichever destination the router's return value maps to. In the output:

```console
$ uv run 03_conditional_routing.py
-4 is negative.
You gave me zero.
7 is positive.
```

Three invocations, three different paths through the graph. The ReAct agent in Chapter "Building a ReAct Agent with LangGraph + Ollama" is built on exactly this pattern: after the "call model" node runs, a router looks at the model's reply, and either routes to the "run tool" node (if there is a tool call) or to `END` (if the model returned a final answer).

Conditional edges can also loop. The destination can be a node earlier in the graph, and the engine will happily re-execute it. That is how you build the "call model, run tool, call model, run tool, ..., call model, done" cycle of a ReAct loop without writing any explicit loop code.

## Example 4: an LLM in a node

Now we put an actual model inside a node in the example `04_llm_in_a_node.py`:

```python
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


model = ChatOllama(model="qwen3.5:4b", temperature=0)


def call_model(state: State) -> dict:
    reply = model.invoke(state["messages"])
    return {"messages": [reply]}


graph = StateGraph(State)
graph.add_node("model", call_model)
graph.add_edge(START, "model")
graph.add_edge("model", END)

app = graph.compile()

initial_messages = [
    SystemMessage(content="You answer in one short sentence."),
    HumanMessage(content="What is the capital of Arizona?"),
]

final_state = app.invoke({"messages": initial_messages})

for m in final_state["messages"]:
    print(f"{type(m).__name__}: {m.content}")
```

Two things are new. The state has a single field, `messages`, reduced by `add_messages`, the standard shape for a chat-style graph. And the node function calls `model.invoke(state["messages"])` and returns the model's reply wrapped in a single-element list. The reducer appends it to the transcript.

Output:

```console
$ uv run 04_llm_in_a_node.py
SystemMessage: You answer in one short sentence.
HumanMessage: What is the capital of Arizona?
AIMessage: The capital of Arizona is Phoenix.
```

The final state contains the full transcript in order: the two messages we passed in plus the model's reply. If we ran the graph again with the returned state as the new initial state, we would have a two-turn conversation. If we added a second node that called the model on the transcript again, we would have a self-dialoguing loop. That is essentially the shape of a ReAct agent, minus the tool-calling step in the middle, which is what Chapter "Building a ReAct Agent with LangGraph + Ollama" assembles.

## What we covered

Four primitives are the entire vocabulary of LangGraph:

1. A **state schema**: a `TypedDict` with fields the graph tracks.
2. **Nodes**: pure functions from state to partial state update.
3. **Reducers**: `Annotated[type, reducer_fn]` to control how new values merge with old ones.
4. **Edges**: plain sequential ones with `add_edge`, or state-dependent ones with `add_conditional_edges`.

Everything else in Part I is a combination of those four. Chapter "Building a ReAct Agent with LangGraph + Ollama" builds a ReAct agent by combining a "call model" node, a "run tool" node, and a conditional edge that routes between them. Chapter "Durable, Restart-Safe Agents" adds a `SqliteSaver` checkpoint so the same graph can pause and resume across process restarts. Chapter "Human-in-the-Loop Patterns" uses interrupts and checkpoint editing to hand control to a human mid-run. Chapter "Multi-Agent Supervisor Pattern" composes multiple graphs into a supervisor pattern. All of it is the same four primitives, in slightly different arrangements.
