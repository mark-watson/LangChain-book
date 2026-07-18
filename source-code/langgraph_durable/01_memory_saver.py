"""MemorySaver: in-process conversation memory.

`MemorySaver` is the simplest checkpointer. It keeps every checkpoint in a
dict in RAM. When the process exits, the memory is gone. It is the right
thing to use inside a single test script, a notebook, or a request-scoped
web endpoint where the conversation only needs to live for the current
request.

The mechanism to demonstrate: a `thread_id` in the invoke config identifies
a conversation. Invoking the graph twice with the same `thread_id` runs the
second turn on top of the first turn's saved state.
"""

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from _graph import build_graph

checkpointer = MemorySaver()
agent = build_graph().compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "demo"}}

turns = [
    "My name is Mark. What is 12 * 12?",
    "What did I just ask you to compute?",
    "What is my name?",
]

for user_turn in turns:
    print(f"USER: {user_turn}")
    result = agent.invoke({"messages": [HumanMessage(content=user_turn)]}, config=config)
    reply = result["messages"][-1]
    print(f"AGENT: {reply.content.strip()}\n")
