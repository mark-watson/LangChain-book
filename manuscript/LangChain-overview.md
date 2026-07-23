# LangChain 1.0 in one hour

[LangChain](https://github.com/langchain-ai/langchain) 1.0 shipped in October 2025 as the first version the maintainers committed to no breaking changes on until 2.0. The library has been through a lot of API churn since I wrote the first edition of this book in early 2023, so if you have used LangChain before and stopped a year or two ago, be prepared for most of what you remember to have moved. The good news is that the 1.0 surface is small, well-organized, and, for the purposes of this book, entirely usable without any of the commercial services LangChain Inc. sells on top of it.

This chapter is the one-hour tour. By the end of it you will have run six standalone Python programs that together cover every LangChain primitive we use in the rest of Part I: chat models, `.invoke` / `.stream` / `.batch`, prompt templates, the LCEL `|` operator, output parsers (including structured Pydantic output), and tool binding. The example code lives in `source-code/langchain_getting_started/`.

## The chapter directory layout

Every chapter in this book follows the same source-code convention. In `source-code/langchain_getting_started/` you will find:

```text
langchain_getting_started/
  pyproject.toml       # pinned library versions
  README.md            # brief run instructions
  01_hello_model.py
  02_hosted_model.py
  03_stream_and_batch.py
  04_prompt_template.py
  05_output_parser.py
  06_tool_binding.py
```

To install and run:

```console
$ cd source-code/langchain_getting_started
$ uv sync
$ uv run 01_hello_model.py
```

The `uv sync` step creates a `.venv` local to the chapter directory and installs the exact library versions the chapter was written against. If you have never used `uv`, it is a drop-in replacement for `pip + venv` that is roughly ten times faster and eliminates the "which Python am I on" bug class. If you would rather stick with plain `pip`, the `pyproject.toml` is a normal PEP 621 file and `pip install -e .` in a manual virtualenv works fine.

## Your first chat model call

The first example is the smallest program that exercises the LangChain 1.0 chat model interface end to end. `source-code/langchain_getting_started/01_hello_model.py`:

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

model = ChatOllama(model="qwen3.5:4b", temperature=0)

messages = [
    SystemMessage(content="You answer concisely, in one sentence."),
    HumanMessage(content="What is the capital of Arizona?"),
]

response = model.invoke(messages)

print(type(response).__name__)
print(response.content)
```

There are three things worth pointing out in fourteen lines of code.

-The import paths. In LangChain version 1.0, LangChain is broken up into a handful of small packages. `langchain_core` holds the primitives (messages, prompt templates, runnables, output parsers). Each LLM provider has its own package: `langchain_ollama`, `langchain_openai`, `langchain_google_genai`, and so on. The umbrella `langchain` package pulls in higher-level pieces built on top of the core. In practice you almost always import messages and prompts from `langchain_core` and models from the provider package that matches the LLM you are using.
- Second, the message objects. `SystemMessage` sets up the assistant's persona and instructions; `HumanMessage` is the user's turn. There is also `AIMessage`, which is what `.invoke()` returns and what you would include in a `messages` list to represent prior assistant turns in a multi-turn conversation.
- Third, `.invoke()`. Every LangChain component that can be called (models, prompt templates, chains, agents, output parsers) is a `Runnable`, and every `Runnable` has the same `.invoke()`, `.stream()`, and `.batch()` methods. That uniformity is the point of the 1.0 API refactor.

Run this example:

```console
$ uv run 01_hello_model.py
AIMessage
The capital of Arizona is Phoenix.
```

The `AIMessage` on the first line of output is the class name of what came back from `.invoke()`. The `.content` attribute is the text. In a moment we will see other useful attributes on `AIMessage`, like `.tool_calls` and `.response_metadata`.

## Swapping in a hosted model

The whole point of LangChain's abstraction is that swapping the LLM provider changes exactly one line of code. `source-code/langchain_getting_started/02_hosted_model.py` is the same program pointed at OpenAI:

```python
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("Set OPENAI_API_KEY in your environment to run this example.")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

messages = [
    SystemMessage(content="You answer concisely, in one sentence."),
    HumanMessage(content="What is the capital of Arizona?"),
]

response = model.invoke(messages)
print(response.content)
```

The message list, the `.invoke()` call, and the response type are all identical. Only the constructor changes. This holds for the Gemini and Fireworks.ai chat classes too. If you write the rest of your application against the abstract `BaseChatModel` protocol (accept a model as a parameter, do not import a specific provider class in your business logic), you can move between local and hosted models without touching anything else.

I mostly develop against a local `ChatOllama` because iteration is instant and I do not pay for tokens I waste. When I want to compare against a stronger model I flip the constructor, and that is usually the entire diff.

## Two more Runnable methods `.stream()` and `.batch()`

Here we look at two  more `Runnable` methods you will use constantly. `source-code/langchain_getting_started/03_stream_and_batch.py`:

```python
from langchain_ollama import ChatOllama

model = ChatOllama(model="qwen3.5:4b", temperature=0)

print("=== .stream() ===")
for chunk in model.stream("Write a two-sentence description of Sedona, Arizona."):
    print(chunk.content, end="", flush=True)
print()

print("\n=== .batch() ===")
prompts = [
    "Name one bird native to Arizona.",
    "Name one bird native to Alaska.",
    "Name one bird native to Florida.",
]
responses = model.batch(prompts)
for prompt, response in zip(prompts, responses):
    print(f"{prompt!r} -> {response.content.strip()}")
```

`.stream()` yields `AIMessageChunk` objects one at a time as the model produces tokens. Use it any time a human is watching the output appear: a CLI, a chat UI, a terminal script. `.batch()` runs multiple inputs concurrently and returns the results in the same order. Use it for offline data processing where per-item latency does not matter but total throughput does.

Both methods work identically whether the underlying model is local or hosted, and both work on any `Runnable`, not just models: a whole chain composed with `|` streams and batches the same way.

## Prompt templates and the LCEL pipe

You could keep building message lists by hand for the rest of your career, but nobody does. LangChain's expression language, LCEL, lets you compose a prompt template and a model into a single `Runnable` with the `|` operator. `source-code/langchain_getting_started/04_prompt_template.py`:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

model = ChatOllama(model="qwen3.5:4b", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You answer in exactly one sentence."),
        ("human", "How do I {thing_to_do}?"),
    ]
)

chain = prompt | model

for task in ["get to the store", "hang a picture on the wall"]:
    result = chain.invoke({"thing_to_do": task})
    print(f"Q: How do I {task}?")
    print(f"A: {result.content.strip()}\n")
```

`ChatPromptTemplate.from_messages(...)` builds a template with named `{placeholder}` variables. The `|` operator wires that template's output into the model's input, producing a new `Runnable` that takes a dict of variable values and returns an `AIMessage`.

Read `prompt | model` as "call the prompt with whatever I pass in, then feed its result into the model." The whole thing behaves like a function you can `.invoke()`, `.stream()`, or `.batch()`. This is not clever operator overloading for its own sake; it is how the framework encourages you to compose steps without writing glue functions.

## Output parsers and structured output

The field `AIMessage.content` is a string. Most of the time the code downstream from your model call wants either plain text (not the `AIMessage` wrapper) or, better, a validated data structure. Both are one more pipe step. `source-code/langchain_getting_started/05_output_parser.py`:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

model = ChatOllama(model="qwen3.5:4b", temperature=0)

print("=== StrOutputParser ===")
prompt = ChatPromptTemplate.from_messages(
    [("human", "In one word, what color is the sky at noon?")]
)
chain = prompt | model | StrOutputParser()
print(chain.invoke({}).strip())


class Country(BaseModel):
    """Facts about a country."""

    name: str = Field(description="The country's common English name.")
    capital: str = Field(description="The country's capital city.")
    population_millions: float = Field(description="Population in millions, approximate.")


print("\n=== .with_structured_output() ===")
structured_model = model.with_structured_output(Country)

for country in ["Canada", "Germany", "Japan"]:
    result = structured_model.invoke(f"Give me facts about {country}.")
    print(result)
```

`StrOutputParser` is the everyday class to use. It unwraps the `AIMessage` and hands you the raw string. Nine times out of ten this is what you want when a chain's output is going to be printed, logged, or concatenated with other strings.

`.with_structured_output(SomeModel)` is the one that changes how you build things. Give it a Pydantic model and you get back a chain that returns a validated instance of that model. The description strings on the fields become part of the prompt behind the scenes, and the framework handles the JSON schema plumbing. In previous editions of this book we did this by hand with prompts that said "output JSON in this format." That whole class of prompt engineering is now a one-line method call.

Expected output looks like:

```console
$ uv run 05_output_parser.py
=== StrOutputParser ===
blue

=== .with_structured_output() ===
name='Canada' capital='Ottawa' population_millions=39.0
name='Germany' capital='Berlin' population_millions=83.0
name='Japan' capital='Tokyo' population_millions=125.0
```

The exact numbers depend on the model, but the shape is guaranteed.

## Tool binding

The last of the six examples we will look at use the primitive that agents are built on `.bind_tools([...])` that gives the chat model a list of Python functions and their descriptions; when the model decides one of them is the right thing to call, its response comes back with a populated `tool_calls` list instead of prose in `.content`. `source-code/langchain_getting_started/06_tool_binding.py`:

```python
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return their sum."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return their product."""
    return a * b


tools_by_name = {t.name: t for t in [add, multiply]}

model = ChatOllama(model="qwen3.5:4b", temperature=0)
model_with_tools = model.bind_tools([add, multiply])

response = model_with_tools.invoke("What is 137 times 24, plus 3?")

if not response.tool_calls:
    print("Model did not call a tool. Raw text response:")
    print(response.content)
else:
    for call in response.tool_calls:
        fn = tools_by_name[call["name"]]
        result = fn.invoke(call["args"])
        print(f"Model called {call['name']}({call['args']}) -> {result}")
```

The `@tool` decorator wraps a plain Python function into an object the model can understand. The function's docstring is the description the model sees; the type annotations become the JSON schema for the arguments. Both matter: poor docstrings produce poor tool selection.

Two important caveats before you go build an agent on this.

First, this example does exactly one round trip. The model responds with a tool call, we execute the call, and we stop. A real agent feeds the tool's result back to the model as a `ToolMessage`, gets the next response, executes the next call, and so on until the model returns plain text as the final answer. That loop is what Chapter "LangGraph 1.0 fundamentals" builds on top of LangGraph. Rolling your own loop is fine for a one-off script; for anything durable you want the state machine LangGraph gives you.

Second, tool calling requires a model that supports it. Not every Ollama model does. As of mid-2026 the ones I use for the book examples are `qwen3.5:4b`, `llama3.2:3b`, `gemma3:12b-it-qat`, and `mistral-small`. If you point `bind_tools` at a chat-only model the code will not crash; the model will just respond with prose describing what it *would* do instead of returning a `tool_call`. That is one of the least helpful failure modes in the whole framework, and it is worth committing the list of tool-capable models to memory (or, more realistically, to a `README` in your project).

## What we covered

Six primitives. That is the whole surface area you need for the next dozen chapters:

1. **Chat models** from provider-specific packages, all satisfying the same `BaseChatModel` protocol.
2. **`.invoke()`**, **`.stream()`**, and **`.batch()`** on every `Runnable`.
3. **`ChatPromptTemplate`** with `{placeholder}` variables.
4. **The LCEL `|` operator** to compose Runnables into chains.
5. **`StrOutputParser`** and **`.with_structured_output(PydanticModel)`** to turn model output into useful values.
6. **`.bind_tools([...])`** and the `@tool` decorator to let the model call your Python functions.

Everything else in Part I builds on those six primitives rather than introducing new ones. Chapter "RAG patterns with LangChain" puts them to work in a RAG pipeline. Chapters "LangGraph 1.0 fundamentals" through "Multi-agent supervisor pattern" introduce LangGraph and use it to build stateful, durable, human-in-the-loop agents. If any of the primitives in this chapter feels shaky, run the example, tweak it, break it; the feedback loop with a local model is fast enough that experimenting is basically free.
