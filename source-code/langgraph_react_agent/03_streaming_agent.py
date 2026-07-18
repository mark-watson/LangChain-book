"""Watch each node execute with .stream().

The compiled graph is a Runnable, so it supports .stream() just like a
model does. Each yielded item is a dict of the form
{node_name: partial_state_update_that_node_produced}. This is the fastest
way I know to understand what an agent is doing and why — much more
informative than a final answer.

The graph construction below is identical to 02_react_from_scratch.py —
copied here so this script stands alone.
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
    return {"messages": [model.invoke(state["messages"])]}


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


question = (
    "Search for the current population of Canada, then multiply it by 2."
)

print(f"USER: {question}\n")

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
