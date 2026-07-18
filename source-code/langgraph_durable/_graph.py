"""Shared uncompiled graph used by all four scripts in this chapter.

The graph is a single-node chat graph: state is the message transcript,
the one node calls the model on the transcript and appends the reply.
This is deliberately the smallest thing that can carry conversation
state, so nothing distracts from what the checkpointer is doing.

Each script compiles this graph with a different checkpointer.
"""

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
    # Prepend the system prompt if it isn't already at the front.
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT] + list(messages)
    return {"messages": [_model.invoke(messages)]}


def build_graph() -> StateGraph:
    graph = StateGraph(State)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph
