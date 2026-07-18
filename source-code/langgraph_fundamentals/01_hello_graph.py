"""The smallest LangGraph program: one state type, one node, one edge.

Every graph is (state schema, nodes, edges). Compile it, invoke it, get
the final state back.
"""

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
