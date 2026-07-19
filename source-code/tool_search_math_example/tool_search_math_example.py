"""ReAct agent with DuckDuckGo search and a calculator, using LangGraph 1.0.

The old langchain.agents.create_react_agent + AgentExecutor pattern is
replaced by langgraph.prebuilt.create_react_agent, which is the LangChain 1.0
way to build a tool-calling agent.
"""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


@tool
def duckduckgo_search(query: str) -> str:
    """Searches the web using DuckDuckGo and returns the top result."""
    from ddgs import DDGS

    results = DDGS().text(query, max_results=5)
    return " ".join(r["body"] for r in results) if results else "No results found"


@tool
def simple_calculator(expression: str) -> str:
    """Performs simple arithmetic calculations. Input should be a Python expression like '250 * 4'."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error in calculation: {e}"


tools = [duckduckgo_search, simple_calculator]

llm = ChatOllama(model="qwen3.5:4b", temperature=0)

agent = create_react_agent(llm, tools)

# Example 1: web search
result = agent.invoke({"messages": [("user", "search: What is the population of Canada?")]})
print(result["messages"][-1].content)

# Example 2: calculation
result = agent.invoke({"messages": [("user", "calculator: 250 * 4")]})
print(result["messages"][-1].content)
