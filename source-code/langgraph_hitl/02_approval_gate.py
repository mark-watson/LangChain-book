"""Approval gate for a dangerous tool.

The agent has two tools: `multiply` (safe, runs automatically) and
`send_email` (dangerous, requires human approval). The tool node inspects
each pending tool call; if the tool is on the DANGEROUS list, it calls
`interrupt()` with the tool name and args, then either runs the tool
(if the human resumed with "approve") or records a rejection ToolMessage
(otherwise) so the model can react to the refusal.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return their product."""
    return a * b


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. This is a mock — it just returns a confirmation string."""
    return f"[mock] email sent to {to} with subject {subject!r}"


TOOLS = [multiply, send_email]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
DANGEROUS = {"send_email"}


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


model = ChatOllama(model="qwen3.5:4b", temperature=0, thinking=False).bind_tools(TOOLS)


def call_model(state: State) -> dict:
    return {"messages": [model.invoke(state["messages"])]}


def approving_tools(state: State) -> dict:
    last = state["messages"][-1]
    tool_messages = []
    for call in last.tool_calls:
        if call["name"] in DANGEROUS:
            decision = interrupt(
                {
                    "type": "approval_request",
                    "tool": call["name"],
                    "args": call["args"],
                }
            )
            if decision != "approve":
                tool_messages.append(
                    ToolMessage(
                        content=f"[user rejected {call['name']}]",
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                )
                continue
        result = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call["id"],
                name=call["name"],
            )
        )
    return {"messages": tool_messages}


def route_after_model(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


graph = StateGraph(State)
graph.add_node("model", call_model)
graph.add_node("tools", approving_tools)
graph.add_edge(START, "model")
graph.add_conditional_edges(
    "model", route_after_model, {"tools": "tools", END: END}
)
graph.add_edge("tools", "model")

agent = graph.compile(checkpointer=MemorySaver())


def run_with_human_response(question: str, approvals: list[str]) -> None:
    """Run the agent, feeding a list of approval responses to any interrupts."""
    config = {"configurable": {"thread_id": "approval-demo"}}
    result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)

    for decision in approvals:
        interrupts = result.get("__interrupt__")
        if not interrupts:
            break
        info = interrupts[0].value
        print(f"  [interrupt] approval requested for {info['tool']}({info['args']})")
        print(f"  [human]     decision = {decision!r}")
        result = agent.invoke(Command(resume=decision), config=config)

    print("  final:", result["messages"][-1].content.strip())


print("--- Case 1: safe tool, no approval needed ---")
run_with_human_response("What is 137 * 24?", approvals=[])

print("\n--- Case 2: dangerous tool, human approves ---")
run_with_human_response(
    "Send an email to alice@example.com with the subject 'Hi' and body 'Hello'.",
    approvals=["approve"],
)

print("\n--- Case 3: dangerous tool, human rejects ---")
run_with_human_response(
    "Send an email to bob@example.com with the subject 'Meeting' and body 'Tomorrow 10am'.",
    approvals=["reject"],
)
