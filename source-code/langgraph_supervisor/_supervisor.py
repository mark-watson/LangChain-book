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

from _specialists import math_agent, research_agent, setup_debug

import re
import time


# suggested approach by Roberto Alessi:
def _strip_thinking(text: str) -> str:
    """Remove thinking blocks that qwen3 emits despite thinking=False."""
    # If there's a closing </think> tag, take everything after it

    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    # Extract the first JSON object from whatever remains
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text.strip()


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
        "original request, respond with 'FINISH'.\n\n"
        'Respond with exactly one JSON object with a single key "next", '
        "and nothing else. No explanation, no extra keys, no prose.\n"
        'Valid responses: {"next": "research"}, {"next": "math"}, '
        '{"next": "FINISH"}'
    )
)


def _make_supervisor_llm(debug: bool = False):
    return ChatOllama(
        model="qwen3.5:4b",
        temperature=0,
        thinking=False,
        verbose=debug,
        format=RouterDecision.model_json_schema(),
    )


# suggested approach by Roberto Alessi:
def supervisor_node(state: State) -> dict:
    _llm = _make_supervisor_llm(debug=DEBUG)

    if DEBUG:
        msgs = list(state["messages"])
        print(f"DEBUG supervisor_node: entered with {len(msgs)} message(s)")
        if msgs:
            last = msgs[-1]
            print(
                f"DEBUG supervisor_node: last message "
                f"({type(last).__name__}): {str(last.content)[:200]!r}"
            )

    t0 = time.monotonic()
    response = _llm.invoke([SUPERVISOR_PROMPT] + list(state["messages"]))
    elapsed = time.monotonic() - t0

    if DEBUG:
        print(f"DEBUG supervisor_node: LLM call took {elapsed:.2f}s")
        print(f"DEBUG supervisor_node: raw response.content = {response.content!r}")

    clean = _strip_thinking(response.content)

    if DEBUG:
        print(f"DEBUG supervisor_node: after _strip_thinking = {clean!r}")

    if not clean:
        # Model produced no JSON — the question is likely already answered.
        if DEBUG:
            print("DEBUG supervisor_node: empty after strip -> FINISH")
        return {"next": "FINISH"}
    try:
        decision = RouterDecision.model_validate_json(clean)
    except Exception as exc:
        # Malformed JSON — default to FINISH to avoid infinite loops.
        if DEBUG:
            print(f"DEBUG supervisor_node: JSON parse failed ({exc}) -> FINISH")
        return {"next": "FINISH"}

    if DEBUG:
        print(f"DEBUG supervisor_node: decision next = {decision.next!r}")
    return {"next": decision.next}


def research_node(state: State) -> dict:
    try:
        if DEBUG:
            print("DEBUG research_node: invoking research_agent")
        t0 = time.monotonic()
        result = research_agent.invoke({"messages": state["messages"]})
        if DEBUG:
            print(
                f"DEBUG research_node: agent returned after "
                f"{time.monotonic() - t0:.2f}s, "
                f"{len(result['messages'])} message(s)"
            )
            for m in result["messages"]:
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for tc in m.tool_calls:
                        print(
                            f"  [DEBUG] research tool call: {tc['name']}({tc['args']})"
                        )
                elif hasattr(m, "name") and m.name and m.content:
                    print(f"  [DEBUG] research tool result: {str(m.content)[:200]}")
            last = result["messages"][-1]
            print(
                f"DEBUG research_node: returning last message "
                f"({type(last).__name__}): {str(last.content)[:300]!r}"
            )
        return {"messages": [result["messages"][-1]]}
    except Exception as exc:
        if DEBUG:
            print(f"  [DEBUG] research agent failed: {exc}")
        raise


def math_node(state: State) -> dict:
    try:
        if DEBUG:
            print("DEBUG math_node: invoking math_agent")
        t0 = time.monotonic()
        result = math_agent.invoke({"messages": state["messages"]})
        if DEBUG:
            print(
                f"DEBUG math_node: agent returned after "
                f"{time.monotonic() - t0:.2f}s, "
                f"{len(result['messages'])} message(s)"
            )
            for m in result["messages"]:
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for tc in m.tool_calls:
                        print(f"  [DEBUG] math tool call: {tc['name']}({tc['args']})")
                elif hasattr(m, "name") and m.name and m.content:
                    print(f"  [DEBUG] math tool result: {str(m.content)[:200]}")
            last = result["messages"][-1]
            print(
                f"DEBUG math_node: returning last message "
                f"({type(last).__name__}): {str(last.content)[:300]!r}"
            )
        return {"messages": [result["messages"][-1]]}
    except Exception as exc:
        if DEBUG:
            print(f"  [DEBUG] math agent failed: {exc}")
        raise


def route_from_supervisor(state: State) -> str:
    if DEBUG:
        print(f"DEBUG route_from_supervisor: state['next'] = {state['next']!r}")
    if state["next"] == "FINISH":
        return END
    return state["next"]


DEBUG = False


def build_supervisor(debug: bool = False):
    global DEBUG
    DEBUG = debug
    if debug:
        setup_debug(debug=True)
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
