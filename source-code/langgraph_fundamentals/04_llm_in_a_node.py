"""A node that calls a real LLM.

Two new pieces:

- `MessagesState`-style state, with a `messages` field reduced by
  `add_messages`. This is the standard shape for graphs that hold a chat
  transcript. `add_messages` appends new messages to the list and does the
  right thing with tool-call / tool-result message pairs.

- A node function that calls `ChatOllama.invoke(state["messages"])` and
  returns the model's reply for `add_messages` to append.

Everything else is the same graph engine from the previous scripts.
"""

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

# Show every message in the final transcript, in order.
for m in final_state["messages"]:
    print(f"{type(m).__name__}: {m.content}")
