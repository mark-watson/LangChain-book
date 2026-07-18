"""State fields with reducers.

By default, when a node returns {"key": value}, that value REPLACES the
current value of `key` in the state. If instead you want the new value to be
merged with the old one — appended to a list, added to a counter — you
annotate the field with a reducer function.

The classic reducer is `operator.add`, which concatenates lists.
"""

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
