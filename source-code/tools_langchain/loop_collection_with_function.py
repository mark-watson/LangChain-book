"""ReAct agent with looping and math tools using LangGraph 1.0.

The old langchain.agents.initialize_agent pattern is replaced by
langgraph.prebuilt.create_react_agent. Uses a local Ollama model.
"""

from typing import Callable

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent


def loop(a_function: Callable, a_collection: list) -> list:
    """Apply a function to each element of a collection."""
    result = []
    for item in a_collection:
        result.append(a_function(item))
    return result


@tool
def loop_sum(input: str) -> str:
    """Sum all integers from start to end. Input should be in the form '10to30'."""
    values = input.split("to")
    start = int(values[0])
    end = int(values[1])
    result = 0
    for i in range(start, end + 1):
        result += i
    return str(result)


@tool
def add(input: str) -> str:
    """Add numbers. Input should be in the form '1 + 2 + 3'."""
    values = [int(x) for x in input.split("+")]
    return str(sum(values))


@tool
def is_prime(input: str) -> str:
    """Check if a number is prime. Input should be a single integer."""
    n = int(input)

    if n <= 1:
        return "no"
    if n <= 3:
        return "yes"
    if n % 2 == 0 or n % 3 == 0:
        return "no"

    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return "no"
        i += 6

    return "yes"


tools = [loop_sum, add, is_prime]

llm = ChatOllama(model="qwen3.5:4b", temperature=0)
agent = create_react_agent(llm, tools)

result = agent.invoke(
    {"messages": [("user", "Loop over the collection [10, 11, 12, 13, 14] and test each for being a prime number. Sum up the prime numbers")]}
)
print(result["messages"][-1].content)

# The loop helper function (for reference, not used by the agent directly)
def foo(x):
    return x + 1

print(loop(foo, [1, 2, 3, 4, 5]))
