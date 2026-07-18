"""A ReAct agent using the prebuilt FunctionAgent.

`FunctionAgent` is LlamaIndex's equivalent of LangGraph's
`create_react_agent`. It builds the standard "call model, run tools,
loop" agent for you. Under the hood it is a Workflow — you can inspect
its steps if you're curious.
"""

import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

from _tools import multiply, web_search

tools = [
    FunctionTool.from_defaults(fn=multiply),
    FunctionTool.from_defaults(fn=web_search),
]

llm = Ollama(model="qwen3.5:4b", temperature=0, request_timeout=180.0)

agent = FunctionAgent(
    tools=tools,
    llm=llm,
    system_prompt=(
        "You are a helpful assistant. Use the provided tools when they can "
        "answer part of the question. Return a concise final answer."
    ),
)


async def main():
    for q in [
        "What is 137 times 24?",
        "What is the population of Canada, doubled?",
    ]:
        print(f"USER: {q}")
        response = await agent.run(user_msg=q)
        print(f"AGENT: {response}\n")


asyncio.run(main())
