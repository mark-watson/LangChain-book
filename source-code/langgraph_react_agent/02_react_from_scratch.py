"""The same ReAct agent, built explicitly.

Two nodes and one conditional edge is the whole ReAct pattern:

  START -> model
  model -> tools   (if the model's last message has tool_calls)
  model -> END     (otherwise)
  tools -> model   (always loop back after tools run)

`ToolNode` from `langgraph.prebuilt` reads the last message's `tool_calls`,
runs each one against the matching tool, and returns a list of
`ToolMessage` objects. We could hand-roll that but there is no reason to.
"""

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


if __name__ == "__main__":
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
