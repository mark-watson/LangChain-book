"""Editing a checkpoint before the next node runs.

Two-node graph: `propose` writes a draft description of a topic, `refine`
rewrites the draft in a more formal tone. `interrupt_after=["propose"]`
at compile time tells the graph to pause AFTER the propose node runs.
While paused, we inspect the recorded state, edit it with
`agent.update_state(config, {...})`, then resume with `invoke(None)`.

This is a different resume mechanism from `Command(resume=...)`:
- `interrupt()` inside a node + `Command(resume=value)` on the caller side
  is a two-way conversation between the graph and its caller.
- `interrupt_before` / `interrupt_after` + `update_state()` + `invoke(None)`
  is a compile-time pause point that lets the caller freely edit the
  recorded state before the next node runs.
"""

from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    topic: str
    proposal: str
    refined: str


model = ChatOllama(model="qwen3.5:4b", temperature=0)


def propose(state: State) -> dict:
    reply = model.invoke(
        [HumanMessage(content=f"Write a one-sentence description of {state['topic']}.")]
    )
    return {"proposal": reply.content.strip()}


def refine(state: State) -> dict:
    reply = model.invoke(
        [
            HumanMessage(
                content=(
                    "Rewrite the following in a formal, encyclopedic tone. "
                    "Return only the rewritten sentence.\n\n"
                    f"{state['proposal']}"
                )
            )
        ]
    )
    return {"refined": reply.content.strip()}


graph = StateGraph(State)
graph.add_node("propose", propose)
graph.add_node("refine", refine)
graph.add_edge(START, "propose")
graph.add_edge("propose", "refine")
graph.add_edge("refine", END)

agent = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_after=["propose"],  # pause AFTER propose, BEFORE refine
)

config = {"configurable": {"thread_id": "draft-demo"}}

# Run until the interrupt_after pause point.
agent.invoke({"topic": "Sedona, Arizona", "proposal": "", "refined": ""}, config=config)

paused = agent.get_state(config)
print(f"=== Draft produced by 'propose' ===")
print(f"  {paused.values['proposal']}\n")

# Simulate a human edit. In a real app this comes from a UI or another agent.
edited = "Sedona is a small town in Arizona famous for its red-rock landscape."
print(f"=== Human overwrites the draft ===")
print(f"  {edited}\n")
agent.update_state(config, {"proposal": edited})

# Resume without injecting new input.
final = agent.invoke(None, config=config)

print(f"=== 'refine' output (using the edited draft) ===")
print(f"  {final['refined']}")
