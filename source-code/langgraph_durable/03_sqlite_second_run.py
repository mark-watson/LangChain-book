"""SqliteSaver, part 2: resume the conversation from disk in a fresh process.

Run 02_sqlite_first_run.py first, let it exit, then run this script. The
graph will find the previous turn already in `checkpoints.db` and answer
the new question with full knowledge of what the user said before.

This is the killer feature of durable checkpointing: the same conversation
survives a full Python process restart with no code change.
"""

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from _graph import build_graph

CHECKPOINT_DB = "checkpoints.db"
THREAD_ID = "sqlite-demo"

with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
    agent = build_graph().compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": THREAD_ID}}

    user_turn = "What is my name and where do I live?"
    print(f"USER: {user_turn}")

    result = agent.invoke({"messages": [HumanMessage(content=user_turn)]}, config=config)

    print(f"AGENT: {result['messages'][-1].content.strip()}")
    print(f"\nFull transcript for thread {THREAD_ID!r}:")
    for m in result["messages"]:
        print(f"  {type(m).__name__}: {m.content[:120]}")
