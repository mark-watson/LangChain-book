"""The supervisor graph.

Shape:

    START -> supervisor
    supervisor -> research   (if the supervisor says so)
    supervisor -> math       (if the supervisor says so)
    supervisor -> END        (if the supervisor says "FINISH")
    research   -> supervisor (after research responds, ask the supervisor
                              what to do next)
    math       -> supervisor (same)

The supervisor is a chat model whose output is constrained to a Pydantic
schema with one Literal field: the name of the next specialist, or the
string "FINISH". A tiny router function reads that field off the state
and returns the destination for the conditional edge.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from _specialists import math_agent, research_agent

Route = Literal["research", "math", "FINISH"]


class RouterDecision(BaseModel):
    next: Route = Field(
        description=(
            "Which specialist should handle the next step, "
            "or 'FINISH' if the last message already answers the user."
        )
    )


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str


SUPERVISOR_PROMPT = SystemMessage(
    content=(
        "You are the supervisor of a multi-agent system. "
        "Read the conversation so far and decide which specialist should act next. "
        "The specialists are:\n"
        "- 'research': can search the web for factual information.\n"
        "- 'math': can compute sums and products of integers.\n"
        "If the last message in the conversation already fully answers the user's "
        "original request, respond with 'FINISH'."
    )
)

_supervisor_llm = (
    ChatOllama(model="qwen3.5:4b", temperature=0, thinking=False)
    .with_structured_output(RouterDecision, method="json_schema")
)


def supervisor_node(state: State) -> dict:
    decision = _supervisor_llm.invoke([SUPERVISOR_PROMPT] + list(state["messages"]))
    return {"next": decision.next}


def research_node(state: State) -> dict:
    result = research_agent.invoke({"messages": state["messages"]})
    # Only append the specialist's final message to the shared transcript,
    # not the specialist's internal tool-calling turns.
    return {"messages": [result["messages"][-1]]}


def math_node(state: State) -> dict:
    result = math_agent.invoke({"messages": state["messages"]})
    return {"messages": [result["messages"][-1]]}


def route_from_supervisor(state: State) -> str:
    if state["next"] == "FINISH":
        return END
    return state["next"]


def build_supervisor():
    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", research_node)
    graph.add_node("math", math_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"research": "research", "math": "math", END: END},
    )
    graph.add_edge("research", "supervisor")
    graph.add_edge("math", "supervisor")

    return graph.compile()
