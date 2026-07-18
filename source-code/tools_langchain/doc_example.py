"""ReAct agent with custom tools using LangGraph 1.0.

The old langchain.agents.initialize_agent pattern is replaced by
langgraph.prebuilt.create_react_agent, which is the LangChain 1.0 way.
Uses a local Ollama model.
"""

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent


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


tools = [add, is_prime]

llm = ChatOllama(model="qwen3.5:4b", temperature=0)
agent = create_react_agent(llm, tools)

result = agent.invoke({"messages": [("user", "If we add 3, 5, 19, 20 and 24, is the result a prime number?")]})
print(result["messages"][-1].content)
