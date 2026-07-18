"""SqliteSaver, part 1: start a conversation and persist it.

Run this script first. It writes checkpoint rows to `checkpoints.db` in the
current directory. When the script exits, the file remains and contains
the full transcript so far.

Run `03_sqlite_second_run.py` afterwards to resume the same thread from
disk in a fresh Python process.
"""

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from _graph import build_graph

CHECKPOINT_DB = "checkpoints.db"
THREAD_ID = "sqlite-demo"

with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
    agent = build_graph().compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": THREAD_ID}}

    user_turn = "My name is Mark and I live in Sedona, Arizona. Please remember that."
    print(f"USER: {user_turn}")

    result = agent.invoke({"messages": [HumanMessage(content=user_turn)]}, config=config)

    print(f"AGENT: {result['messages'][-1].content.strip()}")
    print(f"\nSaved to {CHECKPOINT_DB!r} on thread {THREAD_ID!r}.")
    print("Now run 03_sqlite_second_run.py in a fresh process.")
