"""Tavily search agent using LangGraph 1.0.

The old langchain.agents.initialize_agent pattern is replaced by
langgraph.prebuilt.create_react_agent. Requires TAVILY_API_KEY
environment variable.
"""

import os

from langchain_ollama import ChatOllama
from langchain_community.tools.tavily import TavilySearchResults
from langgraph.prebuilt import create_react_agent

if not os.getenv("TAVILY_API_KEY"):
    raise SystemExit("Set TAVILY_API_KEY in your environment to run this example.")

llm = ChatOllama(model="qwen3.5:4b", temperature=0.5)
tavily_tool = TavilySearchResults(max_results=5)

agent = create_react_agent(llm, [tavily_tool])

result = agent.invoke({"messages": [("user", "What are fun things to do in Sedona Arizona?")]})
print(result["messages"][-1].content)
