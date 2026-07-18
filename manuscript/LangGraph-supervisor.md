# Multi-agent supervisor pattern

The agents from Chapters 7 through 9 are all single agents with one flat list of tools. That is enough for a large class of applications. Once you have more than five or six tools, or tools that need clearly separate expertise (searching the web is nothing like writing SQL is nothing like reviewing legal documents), a single agent starts to feel unfocused. It hesitates, picks the wrong tool, or misinterprets the results of one tool by treating them like the results of another.

The **supervisor pattern** is the standard response. Instead of one agent with all the tools, you build:

- Several **specialists**, each a small compiled agent (typically `create_react_agent`) with its own focused tool set and system prompt.
- One **supervisor**, a graph node that reads the current conversation and decides which specialist should handle the next step, or that the conversation is complete.

The supervisor is not itself a specialist — it does not have tools of its own. It only routes. After each specialist responds, control returns to the supervisor, which decides whether to route to another specialist or to finish. That last part is what makes the pattern powerful: the supervisor can chain specialists together across a single user query.

LangChain Inc. ships a prebuilt `create_supervisor` helper (in the separate `langgraph-supervisor` package) that generates most of this for you, similar to how `create_react_agent` generates a single-agent ReAct graph. This chapter builds the pattern from scratch using only `langgraph` core, both because it is short — the whole thing is maybe forty lines — and because seeing the mechanics is the fastest way to understand when the prebuilt is or isn't the right shape for your problem.

## The example

Two specialists:

- **research** — a ReAct agent with one tool, `web_search` (DuckDuckGo, no API key).
- **math** — a ReAct agent with two tools, `add` and `multiply`.

Three test questions, chosen to exercise every code path:

1. `"What is 137 times 24?"` — math only.
2. `"What is the population of Canada?"` — research only.
3. `"What is the population of Canada times 2?"` — research **then** math. The supervisor chains two specialists in one query. This is the interesting case.

Setup:

```console
$ cd source-code/langgraph_supervisor
$ uv sync
$ ollama pull qwen3:8b
```

## The specialists

`_specialists.py`:

```python
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def web_search(query: str) -> str:
    """Search DuckDuckGo. Returns the top three text results."""
    from duckduckgo_search import DDGS
    try:
        results = list(DDGS().text(query, max_results=3))
    except Exception as exc:
        return f"Search failed: {exc}"
    if not results:
        return "No results."
    return "\n\n".join(
        f"- {r.get('title', '')}\n  {r.get('body', '')}" for r in results
    )


_model = ChatOllama(model="qwen3:8b", temperature=0)

research_agent = create_react_agent(_model, [web_search])
math_agent = create_react_agent(_model, [add, multiply])
```

Nothing in this file is new. Each specialist is exactly the single-agent ReAct graph from Chapter 7, built with `create_react_agent(model, tools)`. Both use the same `ChatOllama` — you could just as easily give each specialist a different model, which is a common reason to reach for the pattern in the first place (a small, fast model for the research agent that mostly summarizes text; a stronger model for the math agent that has to reason about numbers).

## The supervisor and the graph

`_supervisor.py` is where the actual pattern lives:

```python
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
    ChatOllama(model="qwen3:8b", temperature=0).with_structured_output(RouterDecision)
)


def supervisor_node(state: State) -> dict:
    decision = _supervisor_llm.invoke([SUPERVISOR_PROMPT] + list(state["messages"]))
    return {"next": decision.next}


def research_node(state: State) -> dict:
    result = research_agent.invoke({"messages": state["messages"]})
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
```

Walking the interesting parts.

**`RouterDecision`.** A one-field Pydantic model with a `Literal["research", "math", "FINISH"]` field. This is the schema we hand to `.with_structured_output()`. Because the field is a `Literal`, the model is constrained to return exactly one of those three values — no free-form text, no misspellings for the router function to handle.

**`supervisor_node`.** Invokes the supervisor LLM on the full transcript (prepended with the supervisor system prompt) and returns the model's routing decision as `{"next": ...}`. Nothing gets appended to `messages`, only the `next` field is updated. The supervisor stays silent from the user's point of view.

**`research_node` and `math_node`.** Each one wraps a specialist agent. It invokes the specialist with the shared transcript, then appends only the specialist's *final* message to the shared transcript. The specialist's internal tool-calling turns (its own ToolMessages and intermediate AIMessages) stay inside the specialist and don't pollute the supervisor's view. This is a deliberate design choice — the supervisor only needs the specialist's answer to decide what to do next, not its reasoning.

**`route_from_supervisor`.** Reads `state["next"]` and returns either the specialist name or the `END` sentinel. This is the function `add_conditional_edges` calls after the supervisor node returns.

**The wiring.** Start goes to the supervisor. The supervisor's conditional edges go to research, math, or end. After research runs, we always go back to the supervisor. Same for math. That is the loop.

## Running it

`01_run_supervisor.py`:

```python
from langchain_core.messages import HumanMessage
from _supervisor import build_supervisor

app = build_supervisor()

QUESTIONS = [
    "What is 137 times 24?",
    "What is the population of Canada?",
    "What is the population of Canada times 2?",
]

for q in QUESTIONS:
    print(f"USER: {q}")
    result = app.invoke(
        {"messages": [HumanMessage(content=q)], "next": ""},
        config={"recursion_limit": 25},
    )
    final = result["messages"][-1]
    print(f"FINAL: {final.content.strip()[:300]}\n")
```

Representative output (specialist LLMs will vary in exact wording):

```console
$ uv run 01_run_supervisor.py
USER: What is 137 times 24?
FINAL: 137 times 24 is 3288.

USER: What is the population of Canada?
FINAL: The current population of Canada is approximately 40,528,396.

USER: What is the population of Canada times 2?
FINAL: Doubling Canada's population of about 40,528,396 gives approximately 81,056,792.
```

The third question required both specialists. The supervisor routed to research first, saw the research agent return the population, decided the answer was not complete, routed to math, saw math return the doubled number, decided the answer was now complete, and finished. No code in the graph knew this specific query needed research first and math second — that was a runtime routing decision made by the supervisor model on each turn.

## Watching the routing

`.stream()` makes the routing visible. `02_stream_supervisor.py` streams the third question:

```python
for step in app.stream(
    {"messages": [HumanMessage(content=question)], "next": ""},
    config={"recursion_limit": 25},
):
    for node_name, node_output in step.items():
        print(f"=== node: {node_name} ===")
        if "next" in node_output:
            print(f"  next -> {node_output['next']!r}")
        for m in node_output.get("messages", []):
            snippet = m.content if len(m.content) < 250 else m.content[:250] + "..."
            print(f"  {type(m).__name__}: {snippet}")
        print()
```

A representative session:

```console
$ uv run 02_stream_supervisor.py
USER: What is the population of Canada times 2?

=== node: supervisor ===
  next -> 'research'

=== node: research ===
  AIMessage: The current population of Canada is approximately 40,528,396 ...

=== node: supervisor ===
  next -> 'math'

=== node: math ===
  AIMessage: 40,528,396 * 2 = 81,056,792.

=== node: supervisor ===
  next -> 'FINISH'
```

Five steps — three supervisor calls, one research call, one math call. Every supervisor turn shows its routing decision explicitly. When the routing goes wrong (and with a smaller model it sometimes does), this is where you see it.

## When multi-agent is worth it

The multi-agent pattern trades increased routing overhead (one extra LLM call per turn, plus the increased state complexity) for cleaner separation of concerns. Rough guidance from my own projects:

- **Skip it** if your total tool count is under about six and the tools are all in the same domain. A single ReAct agent handles that fine.
- **Reach for it** when tools cluster into obviously different domains (search + email + files + database + code execution), when different specialists need different LLMs (a fast small one for one job, a big one for another), or when specialists need different system prompts to behave correctly.
- **Skip it** if your users' queries always exercise one specialist. The supervisor's routing call is pure overhead in that case; just build the one specialist directly.
- **Reach for it** as soon as a single ReAct agent starts consistently picking the wrong tool or misinterpreting one tool's output through the lens of another.

You can also add the checkpointer and interrupt machinery from Chapters 8 and 9 to a supervisor graph — it is just a `StateGraph`, and the same `.compile(checkpointer=...)` and `interrupt()` mechanisms work identically. A supervisor with checkpointed state and an approval interrupt on every specialist call is a genuinely useful primitive for building semi-autonomous assistants that a human still oversees.

## What we covered

- Multi-agent: several small specialist graphs coordinated by a supervisor graph.
- Specialists are ordinary compiled `create_react_agent` graphs; nothing new required.
- The supervisor is a single node that uses `.with_structured_output(RouterDecision)` to pick the next specialist or `FINISH`.
- Specialists always loop back to the supervisor, which decides what happens next. That is the mechanism that lets a single user query chain multiple specialists.
- Checkpointers and interrupts (Chapters 8 and 9) work exactly the same way on a supervisor graph as on a single agent.

Chapter 11 leaves the pure-mechanics territory behind and applies everything so far to natural-language querying of a real SQLite database.
