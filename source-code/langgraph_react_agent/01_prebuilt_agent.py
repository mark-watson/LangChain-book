"""A ReAct agent using the prebuilt factory.

`create_react_agent(model, tools)` builds and compiles the exact graph we
construct by hand in `02_react_from_scratch.py`. Use this when the standard
ReAct shape is all you need; drop down to the from-scratch version when you
need to customize routing, add nodes, or change the state schema.
"""

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from _tools import TOOLS

model = ChatOllama(model="qwen3.5:4b", temperature=0)
agent = create_agent(model, TOOLS)

result = agent.invoke(
    {"messages": [HumanMessage(content="What is 137 times 24?")]}
)

for m in result["messages"]:
    print(f"--- {type(m).__name__} ---")
    if getattr(m, "tool_calls", None):
        for call in m.tool_calls:
            print(f"  tool_call: {call['name']}({call['args']})")
    if m.content:
        print(m.content)
