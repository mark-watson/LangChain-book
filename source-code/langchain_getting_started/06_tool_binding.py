"""Binding Python functions as tools.

.bind_tools([...]) tells the model which functions it may call. When the
model wants to use one, its response comes back with `tool_calls` populated
instead of `content`. You then execute the call yourself and, in a real
agent, feed the result back to the model.

This example does one round only. Chapters 6 and 7 build a full ReAct loop
on top of LangGraph — this is the primitive that loop is built on.

Note: not every local model supports tool calling. As of mid-2026, qwen3.5:4b,
llama3.2:3b, gemma4:12b-it-qat, and mistral-small work. Chat-only models
will silently return prose instead of a tool_call — the classic gotcha.
"""

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
