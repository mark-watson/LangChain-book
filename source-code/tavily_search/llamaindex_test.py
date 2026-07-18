"""Tavily search with LlamaIndex 0.14 using a FunctionAgent.

The old llama_hub.tools.tavily_research + OpenAIAgent pattern is replaced
by the llama-index tools integration + FunctionAgent. Requires TAVILY_API_KEY.
"""

import os

from llama_index.core import Settings
from llama_index.core.agent import FunctionAgent
from llama_index.llms.ollama import Ollama
from llama_index.tools.tavily import TavilyToolSpec

if not os.getenv("TAVILY_API_KEY"):
    raise SystemExit("Set TAVILY_API_KEY in your environment to run this example.")

Settings.llm = Ollama(model="qwen3.5:4b", request_timeout=120.0)

tavily_tool = TavilyToolSpec(api_key=os.environ.get("TAVILY_API_KEY"))

agent = FunctionAgent.from_tools(tavily_tool.to_tool_list(), llm=Settings.llm)

response = agent.chat("What are fun things to do in Sedona Arizona?")
print(response)
