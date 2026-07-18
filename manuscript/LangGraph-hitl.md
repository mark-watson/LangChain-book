# Human-in-the-loop patterns

The agents from Chapters 7 and 8 run to completion without ever pausing to check in with a human. That is exactly what you want for most background tasks and most read-only assistants. It is exactly what you *do not* want any time an agent is about to take a consequential action — send an email, make a purchase, publish a post, execute a shell command — or produce output that a human needs to see and possibly correct before anything downstream consumes it.

This chapter covers three LangGraph mechanisms for putting a human in the loop:

- **`interrupt()`** — called from inside a node, halts the graph and returns control to the caller with a payload. The caller decides what to do, then resumes with `Command(resume=value)`. The value shows up as the return of `interrupt()`, and the node continues.
- **`interrupt_before=[...]` and `interrupt_after=[...]`** — compile-time arguments that tell the graph to pause before or after specific nodes without any special code in the node bodies.
- **`agent.update_state(config, values)`** — modify the recorded state of a paused graph before resuming. Combined with `interrupt_after`, this is the pattern for "let the human see and edit what the graph produced."

Everything in this chapter assumes a checkpointer. `interrupt()` without a checkpointer would just be a crash — the graph would have nowhere to save the paused state to. All three examples use `MemorySaver` because the human lives in the same Python process, but every mechanism works identically with `SqliteSaver` or `PostgresSaver`, and in a real web-app deployment that is what you would use.

## Example 1: the minimum viable interrupt

`source-code/langgraph_hitl/01_interrupt_basic.py`:

```python
from operator import add
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    log: Annotated[list[str], add]


def step_one(state: State) -> dict:
    return {"log": ["step one done"]}


def step_two(state: State) -> dict:
    answer = interrupt({"question": "What should I record next?"})
    return {"log": [f"human said: {answer}"]}


def step_three(state: State) -> dict:
    return {"log": ["step three done"]}


graph = StateGraph(State)
graph.add_node("one", step_one)
graph.add_node("two", step_two)
graph.add_node("three", step_three)
graph.add_edge(START, "one")
graph.add_edge("one", "two")
graph.add_edge("two", "three")
graph.add_edge("three", END)

app = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "1"}}

first = app.invoke({"log": []}, config=config)

print("=== State after first invoke (paused at interrupt) ===")
print(f"log so far: {first.get('log')}")
print(f"interrupt payload: {first.get('__interrupt__')}")

final = app.invoke(Command(resume="hello from the human"), config=config)

print("\n=== Final state after resume ===")
for line in final["log"]:
    print(f"  {line}")
```

Three-node linear graph. Nothing special until the middle node calls `interrupt({"question": "..."})`. The graph engine catches that call, records the payload as the interrupt info on the current state, and returns from `.invoke()` with the state as of *before* the interrupt fired.

Expected output:

```console
$ uv run 01_interrupt_basic.py
=== State after first invoke (paused at interrupt) ===
log so far: ['step one done']
interrupt payload: [Interrupt(value={'question': 'What should I record next?'}, ...)]

=== Final state after resume ===
  step one done
  human said: hello from the human
  step three done
```

The first `.invoke()` returns after step_one but before step_two finishes; the log has one entry, and `__interrupt__` in the returned state holds the payload the node passed to `interrupt()`. The second `.invoke()` uses `Command(resume="hello from the human")` — the resume value goes back to the paused `interrupt()` call, which returns it, and step_two proceeds. Step_three runs afterwards as normal.

Two things worth internalizing:

- **The interrupt happens inside the node body**, but the caller decides the payload's meaning. The graph does not know or care that "reject" is different from "approve" — it just hands whatever value it receives back to the paused node.
- **Resume is not restart.** When you resume, the node continues from the `interrupt()` call; it does not re-run from the top. Anything the node did before the interrupt is intact, and no state changes have been committed yet (nodes commit their return value on completion, not during execution).

## Example 2: a tool-approval gate

Now the realistic use case. An agent has two tools; one is safe (`multiply`), the other is dangerous (`send_email`, mocked here so the example is self-contained). The tool node inspects each pending tool call: if the tool is on the dangerous list, it calls `interrupt()` with the tool name and args, waits for a decision from the caller, and either runs the tool or records a rejection.

`source-code/langgraph_hitl/02_approval_gate.py`, showing the two new pieces on top of the standard ReAct shape:

```python
DANGEROUS = {"send_email"}


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
```

Compared to the vanilla `ToolNode` from Chapter 7, this custom node adds one branch: if the tool is dangerous, call `interrupt()`, and act on the returned decision. If the decision is `"approve"` the tool runs. Otherwise the node records a `ToolMessage` with content `"[user rejected send_email]"` so the model sees on the next turn that its request was refused and can respond accordingly.

The driver code runs three cases through the same agent — safe tool, dangerous tool approved, dangerous tool rejected — feeding a scripted list of "approve" / "reject" responses to whatever interrupts fire. Representative output:

```console
$ uv run 02_approval_gate.py
--- Case 1: safe tool, no approval needed ---
  final: 137 times 24 is 3288.

--- Case 2: dangerous tool, human approves ---
  [interrupt] approval requested for send_email({'to': 'alice@example.com', 'subject': 'Hi', 'body': 'Hello'})
  [human]     decision = 'approve'
  final: The email has been sent to alice@example.com with the subject "Hi".

--- Case 3: dangerous tool, human rejects ---
  [interrupt] approval requested for send_email({'to': 'bob@example.com', 'subject': 'Meeting', 'body': 'Tomorrow 10am'})
  [human]     decision = 'reject'
  final: The user rejected sending the email, so no message was sent.
```

Notice how the model handles the rejection gracefully in case 3. It sees the `[user rejected send_email]` ToolMessage on the next turn, reasons that no email was sent, and produces a sensible final answer. That is why we return a `ToolMessage` rather than raising — the model needs to know its action was refused so it can adapt.

This approval pattern generalizes. Any tool with side effects — filesystem writes, database mutations, external API calls, financial transactions — belongs in the `DANGEROUS` set until you can prove the model handles it responsibly. Start conservative and remove tools from the list as you build evidence for each one.

## Example 3: pause after a node, edit the checkpoint, resume

The third mechanism is compile-time interrupts combined with `update_state()`. Instead of the node itself deciding to pause, you tell the compiler "always pause after this node runs," inspect the recorded state, edit it, then resume.

`source-code/langgraph_hitl/03_edit_draft.py`:

```python
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    topic: str
    proposal: str
    refined: str


model = ChatOllama(model="qwen3:8b", temperature=0)


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
    interrupt_after=["propose"],
)

config = {"configurable": {"thread_id": "draft-demo"}}

agent.invoke({"topic": "Sedona, Arizona", "proposal": "", "refined": ""}, config=config)

paused = agent.get_state(config)
print(f"=== Draft produced by 'propose' ===")
print(f"  {paused.values['proposal']}\n")

edited = "Sedona is a small town in Arizona famous for its red-rock landscape."
print(f"=== Human overwrites the draft ===")
print(f"  {edited}\n")
agent.update_state(config, {"proposal": edited})

final = agent.invoke(None, config=config)

print(f"=== 'refine' output (using the edited draft) ===")
print(f"  {final['refined']}")
```

Three pieces are new relative to earlier chapters.

**`interrupt_after=["propose"]`** at compile time. The graph pauses after `propose` completes, before `refine` starts. No `interrupt()` call inside either node.

**`agent.update_state(config, {"proposal": edited})`** overwrites part of the recorded state. `update_state` behaves exactly like a node return would: it merges its dict into the current state through the field's reducer (or replaces the field if no reducer is set). You can update any field, including `messages` if you want to edit the transcript directly.

**`agent.invoke(None, config=config)`** resumes without injecting new input. `None` is the signal for "just continue from where you paused." If you passed a dict here it would be applied as an additional state update before continuing, which is another way to inject changes.

Representative output (LLM wording will vary):

```console
$ uv run 03_edit_draft.py
=== Draft produced by 'propose' ===
  Sedona, Arizona is a scenic city in northern Arizona known for its striking red sandstone formations.

=== Human overwrites the draft ===
  Sedona is a small town in Arizona famous for its red-rock landscape.

=== 'refine' output (using the edited draft) ===
  Sedona is a small municipality located in the state of Arizona, distinguished by its notable red-rock geological features.
```

The `refine` node runs on the edited draft, not the original one. From the graph's point of view, nothing unusual happened — the propose node produced state, then the refine node ran on the state it found. The fact that a human sat in between and modified the state is invisible to the nodes themselves. That is exactly the encapsulation you want for HITL: the graph's business logic does not know or care whether a human is watching.

## Two mechanisms, when to reach for each

Both `interrupt()` and `interrupt_after`+`update_state` can pause a graph and involve a human. They are not interchangeable.

**Use `interrupt()`** when the pause is a decision request and the node needs the human's answer to keep going. Approval gates, clarifying questions, "which of these options should I pick" prompts. The node's execution logic depends on what the human says.

**Use `interrupt_after` + `update_state`** when the pause is a review point and the human is editing intermediate output. Draft review, retrieval curation, transcript correction. The node has already done its job; the human is modifying the output before the next node consumes it.

You can also combine both in the same graph — an `interrupt()` for an approval decision, then an `interrupt_after` on a downstream node so the human can edit the tool's result before the model reads it. Any node graph can have any mix.

## What we covered

- HITL requires a checkpointer (already covered in Chapter 8) — the pause has to save the state somewhere.
- `interrupt(payload)` inside a node halts the graph. Caller sees the payload in the returned state's `__interrupt__` key.
- `Command(resume=value)` on the next `.invoke()` resumes the paused node with `value` as the return of `interrupt()`.
- `interrupt_before=[node]` and `interrupt_after=[node]` at `.compile()` time create pause points without any special code in the nodes.
- `agent.update_state(config, values)` edits the recorded state of a paused graph before resuming.
- `.invoke(None, config=config)` resumes from a compile-time pause without injecting new input.

Chapter 10 combines everything so far into a multi-agent supervisor pattern: multiple specialized agents, a coordinating supervisor graph, and (optionally) human approvals on the transitions between them.
