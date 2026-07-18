"""The minimum viable interrupt/resume.

Three-node linear graph. The middle node calls `interrupt()`, which halts
execution and hands control back to the caller. The caller then resumes
by invoking the graph with a `Command(resume=value)`; the interrupt call
returns `value` and the node continues.

The checkpointer is not optional — an interrupt without a checkpointer to
save state to would just be a crash. `MemorySaver` is fine for a demo like
this because the human and the graph live in the same process.
"""

from operator import add
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    log: Annotated[list[str], add]


def step_one(state: State) -> dict:
    return {"log": ["step one done"]}


def step_two(state: State) -> dict:
    answer = interrupt({"question": "What should I record next?"})
    return {"log": [f"human said: {answer}"]}


def step_three(state: State) -> dict:
    return {"log": ["step three done"]}


graph = StateGraph(State)
graph.add_node("one", step_one)
graph.add_node("two", step_two)
graph.add_node("three", step_three)
graph.add_edge(START, "one")
graph.add_edge("one", "two")
graph.add_edge("two", "three")
graph.add_edge("three", END)

app = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "1"}}

# First invocation runs until the interrupt in step_two.
first = app.invoke({"log": []}, config=config)

print("=== State after first invoke (paused at interrupt) ===")
print(f"log so far: {first.get('log')}")
print(f"interrupt payload: {first.get('__interrupt__')}")

# Resume with a value. That value becomes the return of interrupt().
final = app.invoke(Command(resume="hello from the human"), config=config)

print("\n=== Final state after resume ===")
for line in final["log"]:
    print(f"  {line}")
