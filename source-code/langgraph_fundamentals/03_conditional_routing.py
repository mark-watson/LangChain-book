"""Conditional edges: the graph picks its own next node.

`add_conditional_edges` takes a source node, a routing function that reads
state and returns a string, and a mapping from those strings to destination
node names. The graph engine calls the routing function after the source
node returns and follows whichever edge the routing function selected.

This is the piece that makes LangGraph a real state machine rather than a
straight-line pipeline.
"""

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
    """The routing function returns a key of the mapping below."""
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
