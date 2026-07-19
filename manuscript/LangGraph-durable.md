# Durable, restart-safe agents

The agents from Chapter 4 have one important limitation: they forget everything the moment the Python process exits. If you build a chatbot on top of `create_react_agent`, every user turn starts from scratch — no memory of the previous message, let alone yesterday's conversation. That is a script, not a service.

LangGraph fixes this with one concept: the **checkpointer**. You pass a checkpointer to `.compile()`, you thread a `thread_id` through your invoke config, and now every step of your graph — every state update from every node — gets serialized to whatever storage the checkpointer manages. Multi-turn conversations remember previous turns. Interrupted work resumes where it stopped. Crashed processes come back up mid-transaction with no lost state.

This chapter builds four small scripts that demonstrate this progressively: in-process memory, on-disk SQLite persistence, cross-process restart, and inspecting the recorded checkpoint history. All four share one uncompiled state graph — the checkpointer is the only thing that varies.

## Three flavors of checkpointer

LangGraph ships several checkpointer implementations. Only the first is included in the core package; the others are separate PyPI packages.

- **`MemorySaver`** — from `langgraph.checkpoint.memory`, included in the core `langgraph` install. Stores checkpoints in a Python dict. Fast, zero configuration, forgotten on process exit. Use in tests, notebooks, and request-scoped web endpoints where conversation state only needs to live as long as the request.
- **`SqliteSaver`** — from `langgraph.checkpoint.sqlite`, in the separate `langgraph-checkpoint-sqlite` package. Persists checkpoints to a single SQLite file. Zero infrastructure — no server, no daemon, no config file. Perfect for personal projects, small tools, and single-node services where a SQLite file is enough storage.
- **`PostgresSaver`** — from `langgraph.checkpoint.postgres`, in the separate `langgraph-checkpoint-postgres` package. For production services with multiple workers or high concurrency. You bring your own Postgres.

All three implement the same `BaseCheckpointSaver` interface, so swapping between them is a one-line change. Nothing in the graph itself changes.

## The shared graph

All four scripts in `source-code/langgraph_durable/` use the same uncompiled state graph. `_graph.py`:

```python
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful assistant. You keep answers short. "
        "You freely refer back to earlier turns in the conversation."
    )
)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


_model = ChatOllama(model="qwen3.5:4b", temperature=0)


def call_model(state: State) -> dict:
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT] + list(messages)
    return {"messages": [_model.invoke(messages)]}


def build_graph() -> StateGraph:
    graph = StateGraph(State)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph
```

Deliberately the smallest useful graph: `MessagesState`-style transcript, one node that calls the model on it. No tools. The tool-calling ReAct loop from Chapter 4 works exactly the same way once you add a checkpointer, but starting simple lets us focus on what the checkpointer does.

`build_graph()` returns an uncompiled `StateGraph`. Each script compiles it with a different checkpointer.

## Example 1: `MemorySaver` and `thread_id`

`01_memory_saver.py`:

```python
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from _graph import build_graph

checkpointer = MemorySaver()
agent = build_graph().compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "demo"}}

turns = [
    "My name is Mark. What is 12 * 12?",
    "What did I just ask you to compute?",
    "What is my name?",
]

for user_turn in turns:
    print(f"USER: {user_turn}")
    result = agent.invoke({"messages": [HumanMessage(content=user_turn)]}, config=config)
    reply = result["messages"][-1]
    print(f"AGENT: {reply.content.strip()}\n")
```

Two things are new relative to Chapter 3's `04_llm_in_a_node.py`.

**The `checkpointer=` argument to `.compile()`.** Without a checkpointer, each `.invoke()` call starts from a fresh empty state. With one, `.invoke()` looks up the state for the configured thread, appends the new input, runs the graph, and saves the resulting state before returning.

**The `thread_id` in the invoke config.** `thread_id` is the conversation identifier. Two invocations with the same `thread_id` share history; two invocations with different `thread_id`s do not. In a chat application this is typically the user's session ID or conversation ID. In a background job it might be a task ID. In a personal script like this one it can be any string.

Expected output:

```console
$ uv run 01_memory_saver.py
USER: My name is Mark. What is 12 * 12?
AGENT: 12 * 12 is 144, Mark.

USER: What did I just ask you to compute?
AGENT: You asked me to compute 12 * 12.

USER: What is my name?
AGENT: Your name is Mark.
```

Three invocations, three growing transcripts, one thread. On the third invocation the model is given all six prior messages plus the new question, so of course it can answer. That is the whole trick.

## Example 2: `SqliteSaver`, spread across two processes

Now we replace the checkpointer and watch the same mechanism survive a process restart. `02_sqlite_first_run.py`:

```python
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from _graph import build_graph

CHECKPOINT_DB = "checkpoints.db"
THREAD_ID = "sqlite-demo"

with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
    agent = build_graph().compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": THREAD_ID}}

    user_turn = "My name is Mark and I live in Sedona, Arizona. Please remember that."
    print(f"USER: {user_turn}")

    result = agent.invoke({"messages": [HumanMessage(content=user_turn)]}, config=config)

    print(f"AGENT: {result['messages'][-1].content.strip()}")
    print(f"\nSaved to {CHECKPOINT_DB!r} on thread {THREAD_ID!r}.")
    print("Now run 03_sqlite_second_run.py in a fresh process.")
```

`SqliteSaver.from_conn_string(path)` is a context manager that opens the SQLite file, creates the checkpoint tables on first use, and closes cleanly on exit. Everything else — thread config, `.compile()`, `.invoke()` — is identical to the `MemorySaver` version.

Run it:

```console
$ uv run 02_sqlite_first_run.py
USER: My name is Mark and I live in Sedona, Arizona. Please remember that.
AGENT: Got it, Mark — noted that you live in Sedona, Arizona.

Saved to 'checkpoints.db' on thread 'sqlite-demo'.
Now run 03_sqlite_second_run.py in a fresh process.
```

The script exits. Python is gone. All we have is `checkpoints.db` on disk.

`03_sqlite_second_run.py` is nearly identical to script 2, except it asks a question that requires remembering:

```python
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from _graph import build_graph

CHECKPOINT_DB = "checkpoints.db"
THREAD_ID = "sqlite-demo"

with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
    agent = build_graph().compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": THREAD_ID}}

    user_turn = "What is my name and where do I live?"
    print(f"USER: {user_turn}")

    result = agent.invoke({"messages": [HumanMessage(content=user_turn)]}, config=config)

    print(f"AGENT: {result['messages'][-1].content.strip()}")
    print(f"\nFull transcript for thread {THREAD_ID!r}:")
    for m in result["messages"]:
        print(f"  {type(m).__name__}: {m.content[:120]}")
```

Run it after script 2 has fully exited:

```console
$ uv run 03_sqlite_second_run.py
USER: What is my name and where do I live?
AGENT: Your name is Mark, and you live in Sedona, Arizona.

Full transcript for thread 'sqlite-demo':
  HumanMessage: My name is Mark and I live in Sedona, Arizona. Please remember that.
  AIMessage: Got it, Mark — noted that you live in Sedona, Arizona.
  HumanMessage: What is my name and where do I live?
  AIMessage: Your name is Mark, and you live in Sedona, Arizona.
```

The full transcript from both processes is intact. No code has been written to serialize messages, load them, or thread them into the prompt. The checkpointer did all of it.

Delete `checkpoints.db` to reset and start fresh.

## Example 3: inspecting checkpoint history

Every state update from every node in every invocation is a checkpoint. The graph exposes two methods to inspect them:

- `agent.get_state(config)` — returns the current state as a `StateSnapshot`.
- `agent.get_state_history(config)` — yields every checkpoint recorded for the thread, newest first.

`04_state_history.py`:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

from _graph import build_graph

CHECKPOINT_DB = "checkpoints.db"
THREAD_ID = "sqlite-demo"

with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
    agent = build_graph().compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": THREAD_ID}}

    print(f"=== Current state for thread {THREAD_ID!r} ===")
    current = agent.get_state(config)
    for m in current.values["messages"]:
        print(f"  {type(m).__name__}: {m.content[:100]}")

    print(f"\n=== Checkpoint history (newest first) ===")
    for i, snapshot in enumerate(agent.get_state_history(config)):
        n_messages = len(snapshot.values.get("messages", []))
        meta = snapshot.metadata or {}
        source = meta.get("source", "?")
        step = meta.get("step", "?")
        print(f"  [{i}] step={step} source={source} messages_len={n_messages}")
```

After running scripts 2 and 3, this shows the current transcript and then a summary of every checkpoint the graph has recorded for the thread. Each `StateSnapshot` carries a `.config` field you can pass back to `.invoke()` to time-travel — restart the graph from that historical state and run it forward with a different input. That is the mechanism the next chapter uses for human-in-the-loop editing: pause the graph, inspect its state, edit it, then resume.

## Two design points worth internalizing

**`thread_id` is your problem, not the framework's.** The graph does not invent one for you. Any invocation that omits `thread_id` from the config gets a fresh empty state, which is almost never what you want in a durable app. In practice you generate a stable `thread_id` per conversation at your application layer — session cookie, user ID plus timestamp, chat room ID — and thread it through every invoke and stream call.

**Checkpoints are automatic and per-step.** You do not call `.save_state()` or `.load_state()` anywhere. Every time a node produces a state update, the checkpointer writes it. Every time you invoke the graph, the checkpointer loads the latest state for the thread. This is what makes the migration from "prototype in a notebook with `MemorySaver`" to "production service with `PostgresSaver`" a two-line change instead of a rewrite.

## What we covered

- A `checkpointer` passed to `.compile()` gives a graph durable per-thread state.
- `MemorySaver` for in-process, `SqliteSaver` for single-file on-disk, `PostgresSaver` for production. Same interface, same graph, different storage.
- A `thread_id` in the invoke config identifies the conversation. Same thread ID means shared history; different means isolated.
- The state survives full Python process restarts with no extra code.
- `.get_state()` and `.get_state_history()` expose the recorded state for inspection and time-travel.

Chapter 6 uses this same mechanism to build human-in-the-loop patterns: pause the graph mid-run with `interrupt()`, hand control back to your calling code (or a human), edit the checkpoint, then resume from where you stopped. Every one of those capabilities is a direct consequence of the checkpointer machinery in this chapter.
