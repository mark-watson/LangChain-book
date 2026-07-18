# The Workflows API

LlamaIndex used to compose multi-step LLM applications by chaining query engines together with Python glue code. That was fine when applications were simple. Once anything needed branching, retries, or shared state across steps, the chain-of-query-engines pattern started to fall over.

The 2026 answer is **Workflows** — LlamaIndex's own event-driven step-composition API. Structurally it plays the same role in LlamaIndex that LangGraph plays in the LangChain ecosystem: an explicit graph of small typed steps that orchestrates whatever the higher-level components (query engines, agents, retrievers) do individually.

The two APIs solve the same problem with slightly different vocabularies. LangGraph is state-machine flavored — you define a state schema, nodes update it, edges route based on it. Workflows is event-flavored — steps consume typed events and produce other typed events; routing happens implicitly through the type system. Both can express the same set of programs. Which one feels more natural depends on your background and the shape of the specific problem.

Everything in this chapter lives in `source-code/llama_index_workflows/`.

## The four primitives

- **`Workflow`** — a class you subclass. Its `@step` methods are the units of work.
- **`Event`** — the messages that flow between steps. Every step takes exactly one event and returns one event.
- **`StartEvent`** and **`StopEvent`** — special events. The step that accepts a `StartEvent` runs first when you call `.run()`; a step that returns a `StopEvent` ends the workflow.
- **`Context`** — an optional shared state object available to every step (analogous to LangGraph's state). Not used in the first three examples in this chapter; we introduce it in Chapter 19 when we need it.

That is the whole surface. Workflows are asynchronous by default, which is why every step method is `async` and every top-level call is wrapped in `asyncio.run()`.

## Setup

```console
$ cd source-code/llama_index_workflows
$ uv sync
$ ollama pull qwen3:8b
```

## Hello workflow

The smallest possible workflow: one step from `StartEvent` to `StopEvent`.

`01_hello_workflow.py`:

```python
import asyncio

from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step


class HelloWorkflow(Workflow):
    @step
    async def greet(self, ev: StartEvent) -> StopEvent:
        name = ev.get("name", "world")
        return StopEvent(result=f"Hello, {name}!")


async def main():
    wf = HelloWorkflow()
    result = await wf.run(name="Mark")
    print(result)


asyncio.run(main())
```

Three things worth noticing.

**`wf.run(name="Mark")` is how you start a workflow.** Whatever keyword arguments you pass end up on the `StartEvent`. The step that declares `StartEvent` as its input type is the entry point.

**Step wiring happens through Python type annotations.** No `add_edge` calls, no `START` sentinel. The framework inspects each `@step` method's signature, sees which event type it consumes and which it produces, and wires the graph automatically.

**A step that returns a `StopEvent` ends the workflow.** The `result=` value becomes the return of `wf.run()`. If no step returns a `StopEvent`, the workflow runs until it hits its `timeout` (default 10 seconds).

## Chaining steps with a custom event

To wire two steps together, define your own `Event` subclass. `02_two_steps.py`:

```python
import asyncio

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step


class UpperEvent(Event):
    text: str


class TwoStepWorkflow(Workflow):
    @step
    async def uppercase(self, ev: StartEvent) -> UpperEvent:
        return UpperEvent(text=ev.get("text", "").upper())

    @step
    async def reverse(self, ev: UpperEvent) -> StopEvent:
        return StopEvent(result=ev.text[::-1])


async def main():
    wf = TwoStepWorkflow()
    result = await wf.run(text="hello world")
    print(result)


asyncio.run(main())
```

The `uppercase` step returns an `UpperEvent`; the `reverse` step takes an `UpperEvent` as input. The framework matches them automatically. This is the whole edge-declaration mechanism — no `add_edge("uppercase", "reverse")` anywhere. Rearranging or renaming steps does not require touching the wiring, because there is no explicit wiring.

## Branching by event type

Conditional routing works the same way. A step whose return type is `A | B` returns either an `A` or a `B` at runtime, and the framework routes to whichever downstream step consumes that type.

`03_branching.py`:

```python
class BranchingWorkflow(Workflow):
    @step
    async def classify(self, ev: StartEvent) -> PositiveEvent | NegativeEvent:
        n = ev.get("number", 0)
        return PositiveEvent(value=n) if n >= 0 else NegativeEvent(value=n)

    @step
    async def handle_positive(self, ev: PositiveEvent) -> StopEvent:
        return StopEvent(result=f"{ev.value} is positive.")

    @step
    async def handle_negative(self, ev: NegativeEvent) -> StopEvent:
        return StopEvent(result=f"{ev.value} is negative.")
```

This is LlamaIndex's equivalent of LangGraph's `add_conditional_edges`. Same expressive power, different style.

## A three-step LLM workflow

Putting it together with a real model. `04_llm_workflow.py` is a classify-then-answer-or-decline pattern — one of the most common shapes for content-moderation or routing workflows.

```python
class TopicRoutingWorkflow(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = Ollama(model="qwen3:8b", temperature=0, request_timeout=120.0)

    @step
    async def classify(self, ev: StartEvent) -> OnTopicEvent | OffTopicEvent:
        question = ev.get("question", "")
        answer = (
            await self.llm.acomplete(
                "Reply with a single character, Y or N. "
                "Is the following question about software or programming?\n\n"
                f"{question}"
            )
        ).text.strip().upper()
        if answer.startswith("Y"):
            return OnTopicEvent(question=question)
        return OffTopicEvent(question=question)

    @step
    async def answer(self, ev: OnTopicEvent) -> StopEvent:
        reply = await self.llm.acomplete(
            f"Answer this software question briefly:\n\n{ev.question}"
        )
        return StopEvent(result=reply.text.strip())

    @step
    async def decline(self, ev: OffTopicEvent) -> StopEvent:
        return StopEvent(
            result="Sorry, I only answer questions about software and programming."
        )
```

The workflow instantiates its own LLM in `__init__` — that is one common pattern; another is to pass the LLM in as a constructor argument if you want to swap it. Every step is `async` and every LLM call uses the async variant (`.acomplete`, `.achat`) — that is the idiomatic style and it lets the workflow parallelize steps automatically if the graph allows.

`.run(question=q)` accepts a `timeout` in the workflow constructor (`TopicRoutingWorkflow(timeout=120.0)` above); LLM calls can be slow on cold Ollama, and the default 10-second timeout will trip on a first call.

## Workflows vs LangGraph

I have not shipped enough production systems on Workflows to make strong claims yet. What I can say from prototyping both:

- **LangGraph's `StateGraph` is easier to explain in one sitting.** You have a state dict; nodes update it; edges route based on it. That maps directly onto the state-machine mental model that most programmers already have.
- **LlamaIndex Workflows are easier to iterate on.** Adding a step means adding a class and a `@step` method — no wiring changes. Reorganizing the flow is often a matter of renaming event types. This friction difference matters when you are still figuring out what the workflow should do.
- **Both are fully open source.** No hosted-service dependency for either. Both work with Ollama and with hosted models.

If your app is already LlamaIndex-heavy (query engines, indices, LlamaHub readers) — use Workflows. If it is already LangChain-heavy — use LangGraph. If you are starting from scratch, try both on a small prototype and pick the one whose ergonomics fit better.

## What we covered

- Workflows are event-driven step composition: subclass `Workflow`, decorate methods with `@step`, use type annotations to wire.
- The framework matches producer and consumer steps by event type — no explicit edge declarations.
- Branching is a step whose return type is a union of possible events.
- Every step is `async`; every top-level run is wrapped in `asyncio.run`.
- Workflows and LangGraph solve the same problem with different styles. Pick based on the rest of your stack.

Chapter 19 uses Workflows to build a proper ReAct agent — LlamaIndex's answer to Chapter 7's LangGraph ReAct agent.
